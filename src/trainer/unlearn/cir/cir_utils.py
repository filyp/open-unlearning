import logging
import random

import numpy as np
import torch as pt
import torch.nn.functional as F
from datasets import Dataset, concatenate_datasets, load_dataset
from transformers import set_seed as set_transformers_seed

################################ data loading #################################


def load_from_repo(path):
    base_url = "https://raw.githubusercontent.com/filyp/unlearning/refs/heads/main/data"
    return load_dataset(
        "json",
        data_files=[f"{base_url}/{path}"],
        split="train",
    )


def load_fineweb_bio_corpus():
    corpus = load_dataset(
        "m-a-p/FineFineWeb",
        split="train",
        # just the smallest parquet file will be enough
        data_files=["biology/biology_000849.jsonl"],
    )
    return corpus.select(range(2_500))


def load_fineweb_tech_corpus():
    corpus = load_dataset(
        "m-a-p/FineFineWeb",
        split="train",
        data_files=[
            "computer_science_and_technology/computer_science_and_technology_000000.jsonl"
        ],
    )
    return corpus.select(range(2_500))


def prepare_answer_mask(beginning_batch, full_batch):
    long_attn = full_batch["attention_mask"]
    short_attn = beginning_batch["attention_mask"]
    pad_amount = long_attn.shape[1] - short_attn.shape[1]
    short_attn_padded = F.pad(short_attn, (0, pad_amount), value=0)
    answer_mask = (long_attn != short_attn_padded).to(pt.int64)
    return answer_mask


def load_recall_batches(questions, cfg, tokenizer, batch_size=1):
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
        b_txt = beginnings[i : i + batch_size]
        f_txt = fulls[i : i + batch_size]
        beginning_batch = tokenizer(b_txt, **cfg.tokenizer)
        full_batch = tokenizer(f_txt, **cfg.tokenizer)
        full_batch["answer_mask"] = prepare_answer_mask(beginning_batch, full_batch)

        batches.append(full_batch)

    return batches


def load_batches_from_simple_set(dataset, cfg, tokenizer):
    texts = []
    for idx in range(cfg.num_examples_per_question):
        for q in dataset:
            texts.append(q["sentences"][idx])

    batches = []
    for i in range(0, len(texts), cfg.train_batch_size):
        txt = texts[i : i + cfg.train_batch_size]
        batch = tokenizer(txt, **cfg.tokenizer)
        batches.append(batch)

    return batches


def load_wmdp_simple_set(cfg, tokenizer):
    if cfg.dataset == "bio":
        retain_set = load_fineweb_bio_corpus()
    elif cfg.dataset == "cyber":
        retain_set = load_fineweb_tech_corpus()

    is_dev = "dev_" if cfg.use_dev_split else ""
    T = load_from_repo(f"wmdp_deduped_{cfg.dataset}/{is_dev}T_corpus_simple.jsonl")
    V = load_from_repo(f"wmdp_deduped_{cfg.dataset}/{is_dev}V_corpus_simple.jsonl")

    # _model_id = cfg.model_id.split("/")[-1]
    filter_model_id = cfg.filter_model_id
    if filter_model_id in T.column_names:
        logging.info(f"Filtering out texts with {filter_model_id} score < 0.25")
        T = T.filter(lambda x: x[filter_model_id] > 0.25)
        V = V.filter(lambda x: x[filter_model_id] > 0.25)
    else:
        logging.info(f"No {filter_model_id} score in the dataset, not filtering")

    T_and_V = concatenate_datasets([T, V])
    eval_qs = T_and_V if cfg.get("eval_on_all_questions", False) else V
    logging.info(f"{len(T)=}, {len(V)=}, {len(eval_qs)=}")

    training_batches = load_batches_from_simple_set(T_and_V, cfg, tokenizer)
    retraining_batches = load_batches_from_simple_set(T, cfg, tokenizer)
    retain_batches = [
        tokenizer(x["text"], **cfg.tokenizer)
        for x in retain_set.shuffle(seed=42).batch(cfg.retain_batch_size)
    ]
    recall_batches = load_recall_batches(eval_qs, cfg, tokenizer, batch_size=1)

    # * follow the format of open-unlearning
    train_dataset = [
        dict(forget=fb, retain=rb) for fb, rb in zip(training_batches, retain_batches)
    ]

    return train_dataset, retraining_batches, recall_batches, eval_qs


################################ torch utils #################################


def trainable_modules(model):
    return [
        (n, m)
        for n, m in model.named_modules()
        if "_proj" in n and m.weight.requires_grad
    ]


def get_update_norm(model):
    """L2 norm of weight.grad, computed across all the trainable weights."""
    return (
        sum(
            m.weight.grad.to(pt.float32).norm() ** 2
            for _, m in trainable_modules(model)
            if m.weight.grad is not None
        )
        ** 0.5
    )


def scale_grads_(model, factor: float):
    for p in model.parameters():
        if p.grad is not None:
            p.grad *= factor


################################ loss functions #################################


def cross_entropy(output, batch, cfg=None, answer_mask=None):
    shifted_logits = output.logits[:, :-1, :].float()
    shifted_ids = batch["input_ids"][:, 1:]

    # mask out the beginning tokens
    if answer_mask is not None:
        # note, that answer_mask is a subset of attention mask
        assert pt.all(batch["attention_mask"] * answer_mask == answer_mask)
        mask = answer_mask
    else:
        mask = batch["attention_mask"]
    mask = mask[:, 1:].bool()

    return pt.nn.functional.cross_entropy(
        shifted_logits[mask],
        shifted_ids[mask],
        # reduction="sum",
    )


def mlp_breaking_loss(model, batch, cfg, answer_mask=None):
    _mask = answer_mask if answer_mask is not None else batch["attention_mask"]
    _mask = _mask.bool().clone()
    _mask[:, :cfg.cut_off_tokens] = False

    loss_acc = 0
    for layer_id in range(*cfg.layer_range):
        out = model.model.layers[layer_id].mlp.cached_out
        out = out[_mask].float()
        org_out = batch["org_mlp_out"][layer_id].to(out.device).float()
        assert out.shape == org_out.shape
        assert len(out.shape) == 2

        org_norm = batch["org_mlp_out_norm"][layer_id].to(out.device)
        dotproducts = pt.einsum("ts,ts->t", out, org_out)
        dotproducts = dotproducts / org_norm ** 2
        # logging.debug(dotproducts)
        loss_acc += dotproducts.clip(min=cfg.mlp_floor).mean()
        # used to also do max=1, but that's catastrophic - stops unlearning but not disruption

    return loss_acc / len(range(*cfg.layer_range))


def cb_retain_loss(output, batch, cfg, answer_mask=None):
    assert answer_mask is None
    _mask = batch["attention_mask"].bool().clone()
    # _mask[:, :cfg.cut_off_tokens] = False  # do not do it! retain everywhere!
    
    loss_acc = 0
    for layer_id in cfg.cb_retaining_layers:
        acts = output.hidden_states[layer_id][_mask].float()
        org_acts = batch["retain_acts"][layer_id].to(acts.device)[_mask].float()
        assert acts.shape == org_acts.shape
        assert len(acts.shape) == 2

        avg_act_norm = org_acts.norm(dim=-1).mean()
        dist = (acts - org_acts).norm(dim=-1).mean() / avg_act_norm

        loss_acc += dist ** cfg.cb_retaining_pow

    return loss_acc / len(cfg.cb_retaining_layers)

