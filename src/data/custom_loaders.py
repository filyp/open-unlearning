import copy
import logging
import os

import torch as pt
from datasets import concatenate_datasets, load_dataset, load_from_disk


def load_hf_cached(path, split="train", data_files=None):
    """Load a HuggingFace dataset with local disk caching for fast subsequent loads."""
    cache_dir = f".cache/{path.replace('/', '_')}_{split}_{data_files}"
    if os.path.exists(cache_dir):
        return load_from_disk(cache_dir)
    else:
        ds = load_dataset(path, split=split, data_files=data_files)
        os.makedirs(os.path.dirname(cache_dir), exist_ok=True)
        ds.save_to_disk(cache_dir)
        return ds

############## WMDP DEDUPED ##############
# todo they don't support chat templates yet - will need to handle that using data.utils


def _tokenize(text, tokenizer, tokenizer_args):
    batch = tokenizer(text, **tokenizer_args)
    batch["labels"] = copy.deepcopy(batch["input_ids"])
    batch = {k: pt.tensor(v) for k, v in batch.items()}
    return batch


def load_hf_and_tokenize(cfg, **kwargs):
    corpus = load_hf_cached(**cfg.hf_args)

    if "limit" in cfg:
        corpus = corpus.select(range(cfg.limit))
    corpus = corpus.shuffle(seed=42)
    batches = [_tokenize(x["text"], kwargs["tokenizer"], cfg.tokenizer) for x in corpus]
    return {cfg.load_as: batches}


# def _load_from_repo(path, repo="filyp/unlearning"):
#     base_url = f"https://raw.githubusercontent.com/{repo}/refs/heads/main"
#     return load_dataset(
#         "json",
#         data_files=[f"{base_url}/{path}"],
#         split="train",
#     )


def wikitext(cfg, **kwargs):
    wikitext = load_hf_cached(path="filypo/wikitext_16k", split="train")
    batches = [
        kwargs["tokenizer"](x["text"], return_tensors="pt", **cfg.tokenizer)
        for x in wikitext.shuffle(seed=42).batch(cfg.batch_size).take(cfg.num_batches)
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
    
    T = load_hf_cached(path=f"filypo/wmdp_{cfg.dataset}_T", split="train")
    V = load_hf_cached(path=f"filypo/wmdp_{cfg.dataset}_V", split="train")
    
    T_and_V = concatenate_datasets([T, V])
    eval_qs = T_and_V if cfg.get("eval_on_all_questions", False) else V
    logging.info(f"{len(T)=}, {len(V)=}, {len(eval_qs)=}")

    training_batches = [
        _tokenize(q["sentences"][idx], tokenizer, cfg.tokenizer)
        for idx in range(cfg.num_examples_per_question)
        for q in T_and_V
    ]

    relearning_batches = [
        _tokenize(q["sentences"][idx], tokenizer, cfg.tokenizer)
        for idx in range(cfg.num_examples_per_question)
        for q in T
    ]

    recall_batches = _load_recall_batches(eval_qs, cfg, tokenizer)

    return dict(
        forget=training_batches,
        relearn=relearning_batches,
        recall=recall_batches,
        eval_qs=eval_qs,
    )
