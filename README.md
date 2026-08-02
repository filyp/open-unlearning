<div align="center">

# RepSelect: Robust LLM Unlearning via Representation Selectivity

**Filip Sondej**<sup>\*</sup> (Independent) &nbsp;·&nbsp; **Yushi Yang**<sup>\*</sup> (University of Oxford) &nbsp;·&nbsp; **Adam Mahdi** (University of Oxford)

<sub><sup>\*</sup>Equal contribution, author order alphabetical.</sub>

[![Paper](https://img.shields.io/badge/arXiv-2606.17168-b31b1b?logo=arxiv&logoColor=white)](https://arxiv.org/abs/2606.17168)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue)](LICENSE)
[![Built on OpenUnlearning](https://img.shields.io/badge/built%20on-OpenUnlearning-informational)](https://github.com/locuslab/open-unlearning)

</div>

---

## 📖 Overview

Making large language models (LLMs) forget specific knowledge without sacrificing general
capabilities remains a central challenge in machine unlearning. Despite progress, unlearned
knowledge can often be recovered through **fine-tuning** or **few-shot** attacks.

We identify the root cause: existing methods target representations that are **not specific** to
the forget set, which makes unlearning both disruptive to general capabilities and easy to reverse.

We propose **RepSelect** (Representation Selectivity), which isolates representations specific to
the forget set by **collapsing the top principal components of activations and output gradients
before each update**. This leaves general capabilities intact and limits what an attacker can
recover. Prior to unlearning, RepSelect also trains a LoRA on the forget set to elicit harmful
representations, so the method can target them more effectively.

### Key results

- Evaluated on five datasets spanning both **knowledge unlearning** — **WMDP-Bio**, **WMDP-Cyber**,
  and **RWKU** (real-world knowledge about famous people) — and **tendency unlearning** —
  **BeaverTails animal abuse** and **sycophancy**.
- Covers **three model families** spanning dense and Mixture-of-Experts architectures:
  **Llama 3.1 8B, Qwen 3.5 9B, DeepSeek V2 Lite**.
- Compared to five popular baselines (**GradDiff, NPO, SimNPO, RMU, UNDIAL**), RepSelect reduces
  **post-relearning answer probability 4–50× more**, and achieves **near-perfect robustness to
  few-shot attacks**, while retaining matched general capability.

> Selective representation targeting is thus an essential factor for robust LLM unlearning.

This repository is built on the [OpenUnlearning](https://github.com/locuslab/open-unlearning)
framework (see [Built on OpenUnlearning](#-built-on-openunlearning) for credit and the upstream
documentation). RepSelect is implemented as a trainer within that framework.

---

## 🧩 Where RepSelect lives in this repo

| Component | Path |
|-----------|------|
| Method implementation | [`src/trainer/unlearn/repselect_simple.py`](src/trainer/unlearn/repselect_simple.py) (multi-epoch variant: [`src/trainer/unlearn/repselect/`](src/trainer/unlearn/repselect/)) |
| Trainer configs | [`configs/trainer/RepSelectSimple.yaml`](configs/trainer/RepSelectSimple.yaml) (multi-epoch: [`configs/trainer/RepSelect.yaml`](configs/trainer/RepSelect.yaml)) |
| Hyperparameter sweeps | [`configs/hydra/sweeper/RepSelectSimple.yaml`](configs/hydra/sweeper/RepSelectSimple.yaml), [`RepSelectSimpleMoE.yaml`](configs/hydra/sweeper/RepSelectSimpleMoE.yaml) (for MoE models) |
| Paper experiment scripts | [`community/benchmarks/{wmdp_low_mi,beavertails,rwku,sycophancy}/run2.sh`](community/benchmarks/) |
| Unlearn + relearn driver | [`src/unlearn_relearn.py`](src/unlearn_relearn.py) |
| Few-shot / MMLU eval runners | [`scripts/runs/`](scripts/runs/) |
| Representation analysis (PCA, subspace attacks, plots) | [`scripts/interpretability/`](scripts/interpretability/) |

---

## ⚡ Setup

```bash
# Environment
conda create -n unlearning python=3.11
conda activate unlearning
pip install ".[lm-eval]"
pip install --no-build-isolation flash-attn==2.8.3
# Or, to avoid building flash-attn:
pip install "https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.7.16/flash_attn-2.8.3+cu128torch2.9-cp311-cp311-linux_x86_64.whl"

# Evaluation data / reference logs
python setup_data.py --eval
# WMDP and other datasets are supported; see: python setup_data.py --help
```

> **⚠️ Qwen 3.5 note:** to reproduce the Qwen 3.5 experiments, install the fast causal-conv1d
> kernel (`pip install causal_conv1d`). Without it, Qwen 3.5 falls back to a pure-PyTorch
> implementation that can OOM even on an 80 GB GPU.

A [Docker image](https://hub.docker.com/r/filyp/open-unlearning) with the environment preinstalled
is also available.

---

## 🔬 Reproducing the paper

The full set of runs (RepSelect, all baselines, and ablations) for each benchmark is scripted in
the `community/benchmarks/*/run2.sh` files. Each script takes the model name as its argument:

```bash
# WMDP (biohazardous / cybersecurity knowledge; edit wmdp_domain in the script for bio vs cyber)
bash community/benchmarks/wmdp_low_mi/run2.sh Llama-3.1-8B

# RWKU (real-world knowledge)
bash community/benchmarks/rwku/run2.sh Llama-3.1-8B

# BeaverTails (abusive tendencies)
bash community/benchmarks/beavertails/run2.sh Llama-3.1-8B

# Sycophancy
bash community/benchmarks/sycophancy/run2.sh Llama-3.1-8B
```

Supported models: `Llama-3.1-8B`, `Qwen3.5-9B`, `DeepSeek-V2-Lite`. The scripts dispatch each run
through a `run()` wrapper (by default a remote GPU runner); edit it to run locally or on your own
infrastructure.

To run **just RepSelect** on a single setting (this is the main command from those scripts):

```bash
python src/unlearn_relearn.py --config-name=unlearn.yaml --multirun \
  experiment=unlearn/wmdp_low_mi/default \
  model=Llama-3.1-8B wmdp_domain=bio \
  trainer=RepSelectSimple hydra/sweeper=RepSelectSimple \
  task_name=demo_RepSelect
```

- `trainer=RepSelectSimple` — loads [`configs/trainer/RepSelectSimple.yaml`](configs/trainer/RepSelectSimple.yaml), whose `handler` resolves to the trainer in [`src/trainer/unlearn/repselect_simple.py`](src/trainer/unlearn/repselect_simple.py).
- `hydra/sweeper=RepSelectSimple` — the hyperparameter sweep used in the paper (`RepSelectSimpleMoE` for MoE models).
- Ablations are flags on the same trainer: `trainer.method_args.use_lora=false` (no LoRA adversary), `trainer.method_args.distribution=retain|none` (collapse retain PCs / no collapse).
- `src/unlearn_relearn.py` — unlearns, then runs the relearning (fine-tuning) attack to measure robustness.

Swap `trainer=` for `GradDiff | NPO | SimNPO | RMU | UNDIAL` (with the matching `hydra/sweeper=`)
to reproduce the baselines under the same matched-disruption budget.

For general framework usage (configs, distributed training, adding methods/datasets/metrics), see
the upstream OpenUnlearning docs under [`docs/`](docs/).

---

## 📝 Citing this work

```bibtex
@misc{sondej2026repselect,
  title     = {{RepSelect}: Robust {LLM} Unlearning via Representation Selectivity},
  author    = {Sondej, Filip and Yang, Yushi and Mahdi, Adam},
  year      = {2026},
  publisher = {arXiv},
  doi       = {10.48550/arXiv.2606.17168},
  url       = {https://arxiv.org/abs/2606.17168},
}
```

---

## 🤝 Built on OpenUnlearning

This repository is a fork of and built on top of
[**OpenUnlearning**](https://github.com/locuslab/open-unlearning), an easily extensible framework
unifying LLM unlearning benchmarks and methods. RepSelect reuses its benchmark harness, baseline
implementations, and evaluation pipeline. We are grateful to its authors and maintainers.

If you use the underlying framework, please also cite the OpenUnlearning technical report:

```bibtex
@article{openunlearning2025,
  title={{OpenUnlearning}: Accelerating {LLM} Unlearning via Unified Benchmarking of Methods and Metrics},
  author={Dorna, Vineeth and Mekala, Anmol and Zhao, Wenlong and McCallum, Andrew and Lipton, Zachary C and Kolter, J Zico and Maini, Pratyush},
  journal={arXiv preprint arXiv:2506.12618},
  year={2025},
  url={https://arxiv.org/abs/2506.12618}
}
```

---

## 📄 License

Released under the [MIT License](LICENSE), consistent with the upstream OpenUnlearning project.
