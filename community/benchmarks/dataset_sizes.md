# Dataset sizes

Sample counts and token counts for every benchmark, as loaded by `src/data/custom_loaders.py` with the default experiment configs.
Token lengths are the mean of real tokenized lengths (Llama-3 tokenizer, after truncation at the config's `max_length`), not the cap.
Totals are approximate (count × mean length).
To rederive, load each dataset with its default config and average `len(sample["input_ids"])` per split.

All benchmarks use `max_length: 128`, except sycophancy which uses 512 (its chats are median ~296 tokens, so 128 would truncate the entire scored response).

## Training and attack sets

| benchmark | forget | relearn | retain |
|---|---|---|---|
| WMDP-Bio | 567 × 23 ≈ 13k | 282 × 23 ≈ 6.5k | 1000 × ~128 ≈ 128k |
| WMDP-Cyber | 894 × 20 ≈ 18k | 447 × 20 ≈ 9k | 1000 × ~128 ≈ 128k |
| RWKU | 801 × 83 ≈ 66k | 400 × 87 ≈ 35k | 658 × 86 ≈ 57k |
| Animal Abuse | 371 × 94 ≈ 35k | 371 × 93 ≈ 35k | 371 × 94 ≈ 35k |
| Sycophancy | 250 × 265 ≈ 66k | 250 × 267 ≈ 67k | 250 × 291 ≈ 73k |

Notes:

- WMDP sentences average only ~20–23 tokens, far below the 128 cap, so these configs are much lighter than the cap suggests.
- The WMDP retain set is FineFineWeb web text, which typically fills the 128-token cap (mean not measured exactly).
- RWKU relearn is half of forget (held-out passages of the same targets).
- Sycophancy relearn is a disjoint split of misaligned pairs, same count as forget.
- Sycophancy retain uses the same prompts as forget with the non-sycophantic (normal) responses.

## Eval sets (scored every relearn epoch)

| benchmark | recall probes | other probes | retain_eval (KL) | wikitext (KL) |
|---|---|---|---|---|
| WMDP-Bio | 95 × 33 ≈ 3k | — | 64 × ~128 ≈ 8k | 128 × 128 ≈ 16k |
| WMDP-Cyber | 149 × 33 ≈ 5k | — | 64 × ~128 ≈ 8k | 128 × 128 ≈ 16k |
| RWKU | cloze: 164 × 14 ≈ 2.3k | QA: 131 × 19 ≈ 2.5k, neighbor: 275 × 21 ≈ 6k | 64 × 90 ≈ 6k | 128 × 128 ≈ 16k |
| Animal Abuse | 128 × 97 ≈ 12k | — | (safe_eval split) | 128 × 128 ≈ 16k |
| Sycophancy | 200 × 262 ≈ 52k | normal: 200 × 290 ≈ 58k | 64 × 286 ≈ 18k | 128 × 128 ≈ 16k |

The "recall probes" column is the robustness metric of each benchmark: `recall_prob` (WMDP, Sycophancy), `recall_cloze_prob` (RWKU), `holdout_harmful_prob` (Animal Abuse).

Sycophancy's probe sets are the heaviest of any benchmark (~110k tokens per eval), but they are forward-only passes and the probe count (200) is worth keeping.

## One-time attack evals (relearn epoch 0 only)

- Few-shot attack (k=5): WMDP (MCQ format), Animal Abuse and Sycophancy (chat pairs); RWKU has none.
- Sycophancy few-shot contexts are ~1.6k tokens (5 demos + eval example), hence its dedicated `batch_size: 2`.
- MMLU `general_caps` (6 category subsets) is identical across all benchmarks.
