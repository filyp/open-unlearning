import copy
import logging

import torch as pt
import torch.nn.functional as F
from datasets import concatenate_datasets, load_dataset

############## WMDP DEDUPED ##############


def _tokenize(text, tokenizer, tokenizer_args):
    batch = tokenizer(text, **tokenizer_args)
    batch["labels"] = copy.deepcopy(batch["input_ids"])
    batch = {k: pt.tensor(v) for k, v in batch.items()}
    return batch


def load_hf(cfg, **kwargs):
    corpus = load_dataset(**cfg.hf_args)
    if "limit" in cfg:
        corpus = corpus.select(range(cfg.limit))
    batches = [
        _tokenize(x["text"], kwargs["tokenizer"], cfg.tokenizer)
        for x in corpus.shuffle(seed=42)
    ]
    return {cfg.load_as: batches}


def _load_from_repo(path):
    base_url = "https://raw.githubusercontent.com/filyp/unlearning/refs/heads/main/data"
    return load_dataset(
        "json",
        data_files=[f"{base_url}/{path}"],
        split="train",
    )


def wikitext(cfg, **kwargs):
    wikitext = _load_from_repo("wikitext_16k.jsonl")
    batches = [
        kwargs["tokenizer"](x["text"], return_tensors="pt", **cfg.tokenizer)
        for x in wikitext.shuffle(seed=42).batch(cfg.wikitext_batch_size)
    ]
    for rb in batches:
        rb["labels"] = rb["input_ids"].clone()
        rb["labels"][rb["attention_mask"] == 0] = -100

    return {cfg.load_as: batches}


def _load_recall_batches(questions, cfg, tokenizer):
    # batch_size 1 if slower but recommended, because it means we will first average
    # loss per answer and then across answers, which is more stable given some answers
    # are shorter than others
    beginnings = []
    fulls = []
    for q in questions:
        beginning = f"""\
    {q["question"].strip()}
    Answer:"""
        ending = q["choices"][q["answer"]]
        full = f"{beginning} {ending}"
        beginnings.append(beginning)
        fulls.append(full)

    batches = []
    for b_txt, f_txt in zip(beginnings, fulls):
        batch = _tokenize(f_txt, tokenizer, cfg.tokenizer)
        beginning_len = len(tokenizer(b_txt, **cfg.tokenizer)["input_ids"])
        batch["labels"][:beginning_len] = -100
        batch = {k: v.reshape(1, -1) for k, v in batch.items()}
        batches.append(batch)

    return batches


def wmdp_bio_deduped(cfg, **kwargs):
    tokenizer = kwargs["tokenizer"]

    is_dev = "dev_" if cfg.use_dev_split else ""
    T = _load_from_repo(f"wmdp_deduped_{cfg.dataset}/{is_dev}T_corpus_simple.jsonl")
    V = _load_from_repo(f"wmdp_deduped_{cfg.dataset}/{is_dev}V_corpus_simple.jsonl")

    if "filter_model_id" in cfg:
        filter_model_id = cfg.filter_model_id
        if filter_model_id in T.column_names:
            logging.info(f"Filtering out texts with {filter_model_id} score < 0.25")
            T = T.filter(lambda x: x[filter_model_id] > 0.25)
            V = V.filter(lambda x: x[filter_model_id] > 0.25)
        else:
            logging.info(f"No {filter_model_id} score in the dataset, not filtering")

    T_and_V = concatenate_datasets([T, V])
    # eval_qs = T_and_V if cfg.get("eval_on_all_questions", False) else V
    eval_qs = V
    logging.info(f"{len(T)=}, {len(V)=}, {len(eval_qs)=}")

    texts = [
        q["sentences"][idx]
        for idx in range(cfg.num_examples_per_question)
        for q in T_and_V
    ]
    training_batches = [_tokenize(txt, tokenizer, cfg.tokenizer) for txt in texts]

    texts = [
        q["sentences"][idx] for idx in range(cfg.num_examples_per_question) for q in T
    ]
    relearning_batches = [_tokenize(txt, tokenizer, cfg.tokenizer) for txt in texts]

    recall_batches = _load_recall_batches(eval_qs, cfg, tokenizer)

    return dict(
        forget=training_batches,
        relearn=relearning_batches,
        recall=recall_batches,
        eval_qs=eval_qs,
    )


##########################################
