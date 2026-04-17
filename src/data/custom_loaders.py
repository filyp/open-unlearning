import logging
import os

from datasets import concatenate_datasets, load_dataset, load_from_disk


DATE_STRING = "10 Apr 2025"


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


def _tokenize(text, tokenizer, tokenizer_cfg):
    sample = tokenizer(text, return_tensors="pt", **tokenizer_cfg)
    sample = {k: v.squeeze(0) for k, v in sample.items()}
    sample["labels"] = sample["input_ids"].clone()
    return sample


def _apply_chat_template_pair(prompt, response, tokenizer, tokenizer_cfg):
    # note that we skip the system prompt
    chat = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": response},
    ]
    sample = tokenizer.apply_chat_template(
        chat,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        date_string=DATE_STRING,
        **tokenizer_cfg,
    )
    sample = {k: v.squeeze(0) for k, v in sample.items()}
    sample["labels"] = sample["input_ids"].clone()

    beginning_encoding = tokenizer.apply_chat_template(
        chat[:-1],
        tokenize=True,
        return_dict=True,
        add_generation_prompt=True,
        date_string=DATE_STRING,
    )
    beginning_len = len(beginning_encoding["input_ids"])
    return sample, beginning_len


def load_hf_and_tokenize(cfg, tokenizer, **kwargs):
    # note that we don't use a chat template, even with chat models,
    # since the texts are not in a form of chat
    corpus = load_hf_cached(**cfg.hf_args)
    if "range" in cfg:
        corpus = corpus.select(range(*cfg.range))
    corpus = corpus.shuffle(seed=42)
    samples = [_tokenize(x["text"], tokenizer, cfg.tokenizer) for x in corpus]
    return {cfg.dataset_name: samples}


############## WMDP LOW MI ##############


def _load_recall_samples(questions, tokenizer_cfg, tokenizer):
    samples = []
    for q in questions:
        prompt = q["question"].strip()
        response = q["choices"][q["answer"]]

        if tokenizer.chat_template is None:
            beginning_text = f"{prompt}\nAnswer:"
            full_txt = f"{beginning_text} {response}"
            sample = _tokenize(full_txt, tokenizer, tokenizer_cfg)
            beginning_len = len(tokenizer(beginning_text, **tokenizer_cfg)["input_ids"])
        else:
            sample, beginning_len = _apply_chat_template_pair(
                prompt, response, tokenizer, tokenizer_cfg
            )

        sample["labels"][:beginning_len] = -100
        samples.append(sample)
    return samples


def wmdp_low_mi(cfg, tokenizer, **kwargs):
    # note that we don't use a chat template, even with chat models,
    # since the texts are not in a form of chat
    T = load_hf_cached(path=f"filypo/wmdp_{cfg.dataset}_T", split="train")
    V = load_hf_cached(path=f"filypo/wmdp_{cfg.dataset}_V", split="train")

    full = concatenate_datasets([T, V])
    mid = len(full) // 2
    split1 = full.select(range(mid))
    split2 = full.select(range(mid, len(full)))
    logging.info(f"{len(full)=}, {len(split1)=}, {len(split2)=}")

    training_samples = [
        _tokenize(q["sentences"][idx], tokenizer, cfg.tokenizer)
        for idx in range(cfg.num_examples_per_question)
        for q in full
    ]

    relearning_samples = [
        _tokenize(q["sentences"][idx], tokenizer, cfg.tokenizer)
        for idx in range(cfg.num_examples_per_question)
        for q in split1
    ]

    recall_samples = _load_recall_samples(split2, cfg.tokenizer, tokenizer)

    # # TEMP sanity check: print formatting of one sample from each split
    # print("\n===== wmdp_low_mi sample formatting sanity check =====")
    # print(f"[forget] full text:\n{tokenizer.decode(training_samples[0]['input_ids'])}")
    # print(f"[relearn] full text:\n{tokenizer.decode(relearning_samples[0]['input_ids'])}")
    # rs = recall_samples[0]
    # unmasked_ids = rs["input_ids"][rs["labels"] != -100]
    # masked_ids = rs["input_ids"][rs["labels"] == -100]
    # print(f"[recall] full text:\n{tokenizer.decode(rs['input_ids'])}")
    # print(f"[recall] masked prefix (labels=-100):\n{tokenizer.decode(masked_ids)}")
    # print(f"[recall] unmasked target (loss computed here):\n{tokenizer.decode(unmasked_ids)}")
    # print("======================================================\n")

    return dict(
        forget=training_samples,
        relearn=relearning_samples,
        recall=recall_samples,
        eval_qs=split2,
    )


########################### BEAVERTTAILS ###########################


def _get_beavertails_sample(prompt, response, tokenizer, cfg):
    if tokenizer.chat_template is None:
        # we don't use "Question:...\nAnswer:..." format, to not have unlearning base too much on these tokens
        beginning_text = prompt
        full_txt = f"{prompt} {response}"
        sample = _tokenize(full_txt, tokenizer, cfg.tokenizer)
        beginning_len = len(tokenizer(beginning_text, **cfg.tokenizer)["input_ids"])
    else:
        sample, beginning_len = _apply_chat_template_pair(
            prompt, response, tokenizer, cfg.tokenizer
        )

    sample["labels"][:beginning_len] = -100
    return sample


# def beavertails(cfg, tokenizer, **kwargs):
#     # splits: 330k_train, 330k_test, 30k_train, 30k_test
#     full_bt = load_hf_cached("PKU-Alignment/BeaverTails", split=cfg.split)

#     if cfg.category == "safe":
#         texts = full_bt.filter(lambda x: x["is_safe"])
#     else:
#         texts = full_bt.filter(lambda x: x["category"][cfg.category])

#     len_ = cfg.range[1] - cfg.range[0]
#     logging.info(f"{cfg.dataset_name} {len_}/{len(texts)}")
#     samples = []
#     for text in texts.select(range(*cfg.range)):
#         samples.append(
#             _get_beavertails_sample(text["prompt"], text["response"], tokenizer, cfg)
#         )

#     assert len(samples) == len_
#     return {cfg.dataset_name: samples}


# def beavertails_curated(cfg, tokenizer, **kwargs):
#     # splits: animal_abuse, terrorism_organized_crime, safe
#     ds = load_hf_cached("filypo/beavertails-curated", split=cfg.split)
#     texts = ds.filter(lambda x: x["label_correct"])
#     len_ = cfg.range[1] - cfg.range[0]
#     texts = texts.select(range(*cfg.range))
#     logging.info(f"{cfg.dataset_name} {len_}/{len(ds)} (filtered by label_correct)")

#     samples = []
#     for text in texts:
#         samples.append(
#             _get_beavertails_sample(text["prompt"], text["response"], tokenizer, cfg)
#         )

#     assert len(samples) == len_
#     return {cfg.dataset_name: samples}


def beavertails_contrast(cfg, tokenizer, **kwargs):
    # filypo/beavertails-contrast — contrast (retain) pairs, index-aligned with beavertails-curated
    # Do NOT shuffle: ordering must be preserved for index alignment with forget set
    texts = load_hf_cached("filypo/beavertails-contrast", split=cfg.split)
    len_ = cfg.range[1] - cfg.range[0]
    texts = texts.select(range(*cfg.range))
    logging.info(f"{cfg.dataset_name} {len_}/{len(texts)} (contrast set)")

    samples = []
    for text in texts:
        if cfg.original:
            prompt, response = text["original_prompt"], text["original_response"]
        else:
            prompt, response = text["retain_prompt"], text["retain_response"]
        samples.append(_get_beavertails_sample(prompt, response, tokenizer, cfg))

    assert len(samples) == len_

    # # TEMP sanity check: print formatting of the first sample
    # s = samples[0]
    # unmasked_ids = s["input_ids"][s["labels"] != -100]
    # masked_ids = s["input_ids"][s["labels"] == -100]
    # print(f"\n===== beavertails_contrast[{cfg.dataset_name}] sanity check =====")
    # print(f"full text:\n{tokenizer.decode(s['input_ids'])}")
    # print(f"masked prefix (labels=-100):\n{tokenizer.decode(masked_ids)}")
    # print(f"unmasked target (loss computed here):\n{tokenizer.decode(unmasked_ids)}")
    # print("======================================================\n")

    return {cfg.dataset_name: samples}


# def _load_from_repo(path, repo="filyp/unlearning"):
#     base_url = f"https://raw.githubusercontent.com/{repo}/refs/heads/main"
#     return load_dataset(
#         "json",
#         data_files=[f"{base_url}/{path}"],
#         split="train",
#     )
