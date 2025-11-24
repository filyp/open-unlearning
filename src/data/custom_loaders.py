import copy
import logging

import torch as pt
import torch.nn.functional as F
from datasets import concatenate_datasets, load_dataset

############## WMDP DEDUPED ##############


def load_hf(cfg, **kwargs):
    corpus = load_dataset(**cfg.hf_args)
    if "limit" in cfg:
        corpus = corpus.select(range(cfg.limit))
    
    batches = []
    for x in corpus.shuffle(seed=42):
        batch = kwargs["tokenizer"](x["text"], **cfg.tokenizer)
        batch["labels"] = copy.deepcopy(batch["input_ids"])
        batch = {k: pt.tensor(v) for k, v in batch.items()} 
        batches.append(batch)

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


def _prepare_non_answer_mask(beginning_batch, full_batch):
    long_attn = full_batch["attention_mask"]
    short_attn = beginning_batch["attention_mask"]
    pad_amount = long_attn.shape[1] - short_attn.shape[1]
    short_attn_padded = F.pad(short_attn, (0, pad_amount), value=0)
    non_answer_mask = long_attn == short_attn_padded
    return non_answer_mask


def _load_recall_batches(questions, cfg, tokenizer, batch_size=1):
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
    for i in range(0, len(beginnings), batch_size):
        # calculate the mask
        b_txt = beginnings[i : i + batch_size]
        f_txt = fulls[i : i + batch_size]
        beginning_batch = tokenizer(b_txt, return_tensors="pt", **cfg.tokenizer)
        batch = tokenizer(f_txt, return_tensors="pt", **cfg.tokenizer)
        _mask = _prepare_non_answer_mask(beginning_batch, batch)

        # create labels, with -100 for the non-answer tokens
        batch["labels"] = batch["input_ids"].clone()
        batch["labels"][_mask] = -100
        batch["labels"][batch["attention_mask"] == 0] = -100

        batches.append(batch)

    return batches


def _load_batches_from_simple_set(dataset, cfg, tokenizer, b_size):
    texts = []
    for idx in range(cfg.num_examples_per_question):
        for q in dataset:
            texts.append(q["sentences"][idx])

    batches = []
    for i in range(0, len(texts), b_size):
        txt = texts[i : i + b_size]
        batch = tokenizer(txt, return_tensors="pt", **cfg.tokenizer)
        batch["labels"] = batch["input_ids"].clone()
        batch["labels"][batch["attention_mask"] == 0] = -100
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

    texts = [q["sentences"][idx] for idx in range(cfg.num_examples_per_question) for q in T_and_V]
    training_batches = []
    for txt in texts:
        batch = tokenizer(txt, **cfg.tokenizer)
        batch["labels"] = copy.deepcopy(batch["input_ids"])
        batch = {k: pt.tensor(v) for k, v in batch.items()}
        training_batches.append(batch)

    # todo, process it into single element batches, and let collator join them
    # also consolidate with the code above
    relearning_batches = _load_batches_from_simple_set(T, cfg, tokenizer, cfg.relearn_batch_size)

    recall_batches = _load_recall_batches(eval_qs, cfg, tokenizer, batch_size=1)

    return dict(
        forget=training_batches,
        relearn=relearning_batches,
        recall=recall_batches,
        eval_qs=eval_qs,
    )


##########################################