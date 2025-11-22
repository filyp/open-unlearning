import logging

from datasets import concatenate_datasets, load_dataset
import torch.nn.functional as F


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


def _prepare_non_answer_mask(beginning_batch, full_batch):
    long_attn = full_batch["attention_mask"]
    short_attn = beginning_batch["attention_mask"]
    pad_amount = long_attn.shape[1] - short_attn.shape[1]
    short_attn_padded = F.pad(short_attn, (0, pad_amount), value=0)
    non_answer_mask = long_attn == short_attn_padded
    return non_answer_mask


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
        # calculate the mask
        b_txt = beginnings[i : i + batch_size]
        f_txt = fulls[i : i + batch_size]
        beginning_batch = tokenizer(b_txt, **cfg.tokenizer)
        batch = tokenizer(f_txt, **cfg.tokenizer)
        _mask = _prepare_non_answer_mask(beginning_batch, batch)

        # create labels, with -100 for the non-answer tokens
        batch["labels"] = batch["input_ids"].clone()
        batch["labels"][_mask] = -100
        batch["labels"][batch["attention_mask"] == 0] = -100

        batches.append(batch)

    return batches


def load_wikitext_batches(cfg, tokenizer):
    wikitext = load_from_repo("wikitext_16k.jsonl")
    wikitext_batches = [
        tokenizer(x["text"], **cfg.tokenizer)
        for x in wikitext.shuffle(seed=42).batch(cfg.wikitext_batch_size)
    ]
    for wb in wikitext_batches:
        wb["labels"] = wb["input_ids"].clone()
        wb["labels"][wb["attention_mask"] == 0] = -100  # ignore padding tokens
    return wikitext_batches


def load_batches_from_simple_set(dataset, cfg, tokenizer):
    texts = []
    for idx in range(cfg.num_examples_per_question):
        for q in dataset:
            texts.append(q["sentences"][idx])

    batches = []
    for i in range(0, len(texts), cfg.train_batch_size):
        txt = texts[i : i + cfg.train_batch_size]
        batch = tokenizer(txt, **cfg.tokenizer)
        batch["labels"] = batch["input_ids"].clone()
        batch["labels"][batch["attention_mask"] == 0] = -100
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
    for rb in retain_batches:
        rb["labels"] = rb["input_ids"].clone()
        rb["labels"][rb["attention_mask"] == 0] = -100
    recall_batches = load_recall_batches(eval_qs, cfg, tokenizer, batch_size=1)

    # * follow the format of open-unlearning
    train_dataset = [
        dict(forget=fb, retain=rb) for fb, rb in zip(training_batches, retain_batches)
    ]

    wikitext_batches = load_wikitext_batches(cfg, tokenizer)

    return dict(
        train=train_dataset,
        retrain=retraining_batches,
        recall=recall_batches,
        eval_qs=eval_qs,
        wikitext=wikitext_batches,
    )
