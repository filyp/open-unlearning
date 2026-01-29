import copy
import logging
import os

import torch as pt
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


def _tokenize(text, tokenizer, tokenizer_args):
    sample = tokenizer(text, **tokenizer_args)
    sample["labels"] = copy.deepcopy(sample["input_ids"])
    sample = {k: pt.tensor(v) for k, v in sample.items()}
    return sample


def load_hf_and_tokenize(cfg, **kwargs):
    corpus = load_hf_cached(**cfg.hf_args)

    if "limit" in cfg:
        corpus = corpus.select(range(cfg.limit))
    corpus = corpus.shuffle(seed=42)
    samples = [_tokenize(x["text"], kwargs["tokenizer"], cfg.tokenizer) for x in corpus]
    return {cfg.load_as: samples}


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


def _load_recall_samples(questions, cfg, tokenizer):
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

    samples = []
    for b_txt, f_txt in zip(beginnings, fulls):
        sample = _tokenize(f_txt, tokenizer, cfg.tokenizer)
        beginning_len = len(tokenizer(b_txt, **cfg.tokenizer)["input_ids"])
        sample["labels"][:beginning_len] = -100
        samples.append(sample)

    return samples


def wmdp_low_mi(cfg, **kwargs):
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

    recall_samples = _load_recall_samples(eval_qs, cfg, tokenizer)

    return dict(
        forget=training_samples,
        relearn=relearning_samples,
        recall=recall_samples,
        eval_qs=eval_qs,
    )


# def _load_from_repo(path, repo="filyp/unlearning"):
#     base_url = f"https://raw.githubusercontent.com/{repo}/refs/heads/main"
#     return load_dataset(
#         "json",
#         data_files=[f"{base_url}/{path}"],
#         split="train",
#     )
