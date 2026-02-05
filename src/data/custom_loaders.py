import logging
import os

from datasets import concatenate_datasets, load_dataset, load_from_disk


def load_hf_cached(path, split="train", data_files=None):
    """Load a HuggingFace dataset with local disk caching for fast subsequent loads."""
    cache_dir = ".cache/load_hf/"
    cache_dir += f"{path}_{split}_{data_files}".replace("/", "_")
    if os.path.exists(cache_dir):
        logging.info(f"Loading cached dataset from {cache_dir}")
        return load_from_disk(cache_dir)
    else:
        ds = load_dataset(path, split=split, data_files=data_files)
        os.makedirs(os.path.dirname(cache_dir), exist_ok=True)
        ds.save_to_disk(cache_dir)
        return ds


############## WMDP LOW MI ##############
# todo they don't support chat templates yet - will need to handle that using data.utils


def _tokenize(text, tokenizer, tokenizer_cfg):
    sample = tokenizer(text, return_tensors="pt", **tokenizer_cfg)
    sample = {k: v.squeeze(0) for k, v in sample.items()}
    sample["labels"] = sample["input_ids"].clone()
    return sample


def load_hf_and_tokenize(cfg, tokenizer, **kwargs):
    corpus = load_hf_cached(**cfg.hf_args)
    if "limit" in cfg:
        corpus = corpus.select(range(cfg.limit))
    corpus = corpus.shuffle(seed=42)
    samples = [_tokenize(x["text"], tokenizer, cfg.tokenizer) for x in corpus]
    return {cfg.dataset_name: samples}




def _load_recall_samples(questions, tokenizer_cfg, tokenizer):
    samples = []
    for q in questions:
        question_txt = f"{q['question'].strip()}\nAnswer:"
        answer_txt = q["choices"][q["answer"]]
        full_txt = f"{question_txt} {answer_txt}"

        sample = _tokenize(full_txt, tokenizer, tokenizer_cfg)
        beginning_len = len(tokenizer(question_txt, **tokenizer_cfg)["input_ids"])
        sample["labels"][:beginning_len] = -100
        samples.append(sample)
    return samples


def wmdp_low_mi(cfg, **kwargs):  # handler
    tokenizer = kwargs["tokenizer"]

    T = load_hf_cached(path=f"filypo/wmdp_{cfg.dataset}_T", split="train")
    V = load_hf_cached(path=f"filypo/wmdp_{cfg.dataset}_V", split="train")

    T_and_V = concatenate_datasets([T, V])
    eval_qs = T_and_V if cfg.get("eval_on_all_questions", False) else V
    logging.info(f"{len(T)=}, {len(V)=}, {len(eval_qs)=}")

    training_samples = [
        _tokenize(q["sentences"][idx], tokenizer, cfg.tokenizer)
        for idx in range(cfg.num_examples_per_question)
        for q in T_and_V
    ]

    relearning_samples = [
        _tokenize(q["sentences"][idx], tokenizer, cfg.tokenizer)
        for idx in range(cfg.num_examples_per_question)
        for q in T
    ]

    recall_samples = _load_recall_samples(eval_qs, cfg.tokenizer, tokenizer)

    return dict(
        forget=training_samples,
        relearn=relearning_samples,
        recall=recall_samples,
        eval_qs=eval_qs,
    )


########################### BEAVERTTAILS ###########################


def load_beavertails_samples(
    category_name, tokenizer, tokenizer_cfg, limit=1000, split="330k_train"
):
    # splits: 330k_train, 330k_test, 30k_train, 30k_test
    full_bt = load_hf_cached("PKU-Alignment/BeaverTails", split=split)

    if category_name == "safe":
        texts = full_bt.filter(lambda x: x["is_safe"])
    else:
        texts = full_bt.filter(lambda x: x["category"][category_name])

    samples = []
    for text in texts.select(range(limit)):
        # we don't use "Question:...\nAnswer:..." format, to not have unlearning base too much on these tokens
        question_txt = text['prompt']
        answer_txt = text["response"]
        full_txt = f"{question_txt} {answer_txt}"
        sample = _tokenize(full_txt, tokenizer, tokenizer_cfg)
        beginning_len = len(tokenizer(question_txt, **tokenizer_cfg)["input_ids"])
        sample["labels"][:beginning_len] = -100
        samples.append(sample)
    return samples


def beavertails(cfg, **kwargs):  # handler
    tokenizer = kwargs["tokenizer"]
    target = load_beavertails_samples(
        cfg.category, tokenizer, cfg.tokenizer, cfg.target_limit * 2
    )
    safe = load_beavertails_samples(
        cfg.category, tokenizer, cfg.tokenizer, cfg.eval_limit, split="330k_test"
    )
    retain = load_beavertails_samples(
        "safe", tokenizer, cfg.tokenizer, cfg.retain_limit
    )
    assert len(target) == cfg.target_limit * 2
    assert len(retain) == cfg.retain_limit
    assert len(safe) == cfg.eval_limit

    return dict(
        forget=target[: cfg.target_limit],
        relearn=target[cfg.target_limit :],
        beavertails_safe=safe,
        retain=retain,
    )


########################## OTHER HANDLERS ###########################


# def _load_from_repo(path, repo="filyp/unlearning"):
#     base_url = f"https://raw.githubusercontent.com/{repo}/refs/heads/main"
#     return load_dataset(
#         "json",
#         data_files=[f"{base_url}/{path}"],
#         split="train",
#     )
