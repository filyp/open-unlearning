import logging
import math

from omegaconf import OmegaConf

from lm_eval.models.hf_vlms import HFLM
from lm_eval.tasks import TaskManager
from lm_eval import simple_evaluate

from evals.base import Evaluator


logger = logging.getLogger("evaluator")


class LMEvalEvaluator(Evaluator):
    def __init__(self, eval_cfg, **kwargs):
        self.name = "LMEval"
        self.eval_cfg = eval_cfg
        self.tasks = OmegaConf.to_container(
            self.eval_cfg.tasks, resolve=True, throw_on_missing=True
        )
        self.task_manager = TaskManager()
        _sea = self.eval_cfg.get("simple_evaluate_args", None)
        self.simple_evaluate_args = OmegaConf.to_container(_sea, resolve=True) if _sea is not None else {}
        self.aggregate_key = eval_cfg.get("aggregate_key", None)
        self.only_at_relearn_start = eval_cfg.get("only_at_relearn_start", False)
        self.mode = kwargs.get("mode")

    def prepare_model(self, model, **kwargs):
        """Prepare model for evaluation"""
        model.eval()
        # batch_size must go to HFLM: simple_evaluate ignores it for model instances
        trainer = kwargs.get("trainer")
        if trainer is not None:
            batch_size = trainer.args.per_device_eval_batch_size
        else:
            batch_size = self.simple_evaluate_args.get("batch_size", 1)
        return HFLM(model, tokenizer=kwargs.get("tokenizer", None), batch_size=batch_size)

    def summarize(self, eval_results: dict, task_name: str) -> dict:
        """
        Summarize evaluation metrics from lm_eval.simple_evaluate.
        - If task_name is a group, return only aggregated group-level metrics.
        - If it's a single task, return per-task metrics from 'results'.
        - Always exclude 'alias' entries and strip ',none' suffixes.
        """
        summary = {}

        def clean_metric_key(prefix: str, metric_name: str) -> str | None:
            if metric_name == "alias":
                return None
            base = metric_name.split(",", 1)[0].strip()
            return f"{prefix}/{base}"

        # Check if task is a group (e.g., 'mmlu')
        if task_name in self.task_manager.all_groups:
            group_metrics = eval_results.get("groups", {}).get(task_name, {})
            for metric_name, value in group_metrics.items():
                key = clean_metric_key(task_name, metric_name)
                if key is None:
                    continue
                try:
                    summary[key] = float(value)
                except (TypeError, ValueError):
                    summary[key] = value
        else:
            task_metrics = eval_results.get("results", {}).get(task_name, {})
            for metric_name, value in task_metrics.items():
                key = clean_metric_key(task_name, metric_name)
                if key is None:
                    continue
                try:
                    summary[key] = float(value)
                except (TypeError, ValueError):
                    summary[key] = value

        return summary

    def get_task_name(self, task):
        if isinstance(task, str):
            return task
        elif isinstance(task, dict):
            if "task" in task:
                return task.get("task")
        raise ValueError(f"Invalid task format: {task}")

    def evaluate(self, model, output_dir=None, overwrite=None, **kwargs):
        trainer = kwargs.get("trainer")
        if self.only_at_relearn_start and (self.mode != "relearn" or trainer.state.epoch):
            return {}

        # set flag to overwrite metrics
        overwrite = self.eval_cfg.overwrite if overwrite is None else overwrite

        # Prepare model for evaluation
        kwargs = {"tokenizer": kwargs.get("tokenizer", None), "trainer": trainer}
        model = self.prepare_model(model, **kwargs)

        # Set output_dir and file to store results
        output_dir = output_dir if output_dir else self.eval_cfg.output_dir
        logs_file_path = self.get_logs_file_path(output_dir)
        summary_file_path = self.get_logs_file_path(output_dir, suffix="SUMMARY")

        # Load existing results from file if any.
        logs = self.load_logs_from_file(logs_file_path) if not overwrite else {}
        summary = self.load_logs_from_file(summary_file_path) if not overwrite else {}

        logger.info(f"***** Running {self.name} evaluation suite *****")
        logger.info(f"Fine-grained evaluations will be saved to: {logs_file_path}")
        logger.info(
            f"Aggregated evaluations will be summarised in: {summary_file_path}"
        )

        todo = []
        for task in self.tasks:
            task_name = self.get_task_name(task)
            if not overwrite and task_name in logs and logs[task_name]:
                logger.info(f"Skipping {task_name}, already evaluated.")
                continue
            _ = logs.pop(task_name, None)  # overwriting existing evals if present
            todo.append(task)

        if todo:
            # evaluate all remaining tasks in one call to avoid per-task overhead
            results = simple_evaluate(
                model=model,
                tasks=todo,
                task_manager=self.task_manager,
                **self.simple_evaluate_args,
            )
            for task in todo:
                task_name = self.get_task_name(task)
                if task_name in results["samples"]:
                    logs[task_name] = {task_name: results["samples"][task_name]}
                else:  # a group task: samples are keyed by its subtasks
                    logs[task_name] = results["samples"]
                summary.update(self.summarize(results, task_name))
            self.save_logs(logs, logs_file_path)
            self.save_logs(summary, summary_file_path)

        def _task_samples(task):
            task_name = self.get_task_name(task)
            return logs[task_name].get(task_name, [])  # empty for group tasks

        def _acc_t1(samples):
            # mean probability of the correct choice, as in the few-shot evals
            target_lls = [s["resps"][s["target"]][0][0] for s in samples]
            return sum(math.exp(ll) for ll in target_lls) / len(target_lls)

        for task in self.tasks:
            samples = _task_samples(task)
            if samples:
                summary[f"{self.get_task_name(task)}/acc_t1"] = _acc_t1(samples)

        # micro-averaged accuracy over all tasks, logged under eval_cfg.aggregate_key
        if self.aggregate_key:
            samples = [s for task in self.tasks for s in _task_samples(task)]
            summary[f"{self.aggregate_key}/acc"] = sum(s["acc"] for s in samples) / len(samples)
            summary[f"{self.aggregate_key}/acc_t1"] = _acc_t1(samples)

        self.save_logs(summary, summary_file_path)
        return summary
