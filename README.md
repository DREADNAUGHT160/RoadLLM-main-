# RoadLLM

A LLaVA-NeXT based vision-language model for road scene understanding and inspection, built on **Qwen3-8B** as the LLM backbone and **google/siglip-so400m-patch14-384** as the vision encoder.

## Architecture

| Component | Model |
|---|---|
| Language model | `Qwen/Qwen3-8B` (and 0.6B / 1.7B / 4B variants) |
| Vision encoder | `google/siglip-so400m-patch14-384` |
| Connector | MLP 2× GELU (`mlp2x_gelu`) |
| Model class | `LlavaQwen3ForCausalLM` |

## Installation

```bash
git clone https://github.com/DREADNAUGHT160/RoadLLM-main-.git
cd RoadLLM-main-

conda create -n roadllm python=3.10 -y
conda activate roadllm
pip install --upgrade pip
pip install -e ".[train]"
pip install lmms-eval
pip install av --prefer-binary   # pre-built wheel; avoids FFmpeg compilation
```

> **HPC / offline note:** Set `TRANSFORMERS_CACHE` (not `HF_HOME`) to your local HuggingFace hub cache.
> Add `HF_HUB_OFFLINE=1` to disable all network requests on compute nodes.

## Preparing the Datasets

See [docs/dataset.md](./docs/dataset.md) for full download and preprocessing instructions.

Datasets used: Waymo, BDD100k, nuImages, Cityscapes, Mapillary, R2S100k, RDD2022.

## Training

### Stage 1 — Projector pretraining

Only the MLP connector is trained; the LLM and vision encoder are frozen.

```bash
# Single GPU / local
bash local/pretrain_qwen3.bash

# Multi-GPU on HPC (LSF)
bash hpc/job_multi_gpus.bash <num_gpus> <num_cpus> Qwen/Qwen3-8B <epochs>
```

The output is saved to:

```
checkpoints/projectors/roadllm-llava-<vision>-<llm>-mlp2x_gelu-pretrain-full-<N>gpus-<E>epoches/
├── config.json         ← LlavaQwen3Config (mm_* fields)
└── mm_projector.bin    ← trained MLP connector weights only
```

> **Important:** This directory does **not** contain full model weights.
> See the [Evaluation](#evaluation) section for how to run inference from it.

### Stage 2 — Full fine-tuning *(planned)*

Coming soon.

## Evaluation

There are two ways to run evaluation depending on your setup.

### Option A — Direct inference (no merge needed)

`load_pretrained_model` in `llava/model/builder.py` automatically detects a projector-only checkpoint, resolves the base LLM from the HuggingFace cache, loads the model, and injects the projector weights — no extra steps required.

```bash
python -m llava.serve.evaluate_roadllm \
    --conv-mode qwen_3 \
    --model-path ./checkpoints/projectors/roadllm-llava-google_siglip-so400m-patch14-384-Qwen_Qwen3-8B-mlp2x_gelu-pretrain-full-4gpus-5epoches \
    --image-file test_images/urban.jpg \
    --prompt "Describe the image" \
    --output-file results.json
```

On HPC (offline, LSF):

```bash
bsub -Is -q gpu -gpu "num=1:j_exclusive=yes:gmem=40G" -R "select[type==X64LIN]" \
  eval "TRANSFORMERS_CACHE=/path/to/hf/cache HF_HUB_OFFLINE=1 DS_SKIP_CUDA_CHECK=1 \
        python /path/to/RoadLLM/test_bench.py"
```

### Option B — Merge then evaluate (required for lmms-eval)

`lmms-eval` calls `from_pretrained(model_path)` directly and expects a complete checkpoint. Use `merge_model.py` to produce one:

**Step 1 — merge once:**

```bash
python merge_model.py \
    --projector-path ./checkpoints/projectors/roadllm-llava-google_siglip-so400m-patch14-384-Qwen_Qwen3-8B-mlp2x_gelu-pretrain-full-4gpus-5epoches \
    --model-base Qwen/Qwen3-8B \
    --output-path ./checkpoints/merged/roadllm-full
```

**Step 2 — run all benchmarks:**

```bash
bash run_eval.sh
```

This evaluates `gqa`, `mmbench_en_dev`, `pope`, `seedbench`, `vqav2_val` and saves results to `./test_results/`.

Run merge + eval in a single command (merge is skipped if the checkpoint already exists):

```bash
bash run_eval.sh
```

Override defaults via environment variables:

```bash
PROJECTOR_PATH=./checkpoints/projectors/<run> \
MODEL_BASE=Qwen/Qwen3-8B \
MERGED_PATH=./checkpoints/merged/roadllm-full \
RESULTS_DIR=./test_results \
bash run_eval.sh
```

### Benchmark multiple model sizes

```bash
bash hpc/run_benchmark.bash
```

Runs `evaluate_roadllm.py` for the 0.6B, 1.7B, 4B, and 8B SigLIP models and saves a comparison JSON.

## Repository Structure

```
RoadLLM/
├── llava/
│   ├── model/
│   │   ├── builder.py                  ← load_pretrained_model (auto-detects projector-only)
│   │   ├── language_model/
│   │   │   └── llava_qwen3.py          ← LlavaQwen3ForCausalLM
│   │   ├── multimodal_encoder/
│   │   │   └── siglip_encoder.py
│   │   └── multimodal_projector/
│   │       └── builder.py              ← mlp2x_gelu projector
│   ├── serve/
│   │   ├── cli_roadllm.py              ← interactive CLI
│   │   └── evaluate_roadllm.py         ← single-image evaluation script
│   └── train/
│       └── train.py
├── merge_model.py                      ← merge projector + base LLM → safetensors
├── run_eval.sh                         ← full merge + lmms-eval pipeline
├── scripts/
│   └── merge_projector_weights.py      ← generalised merge utility
├── hpc/
│   ├── job_multi_gpus.bash             ← LSF multi-GPU training job
│   ├── job_cli.bash                    ← LSF interactive inference job
│   └── run_benchmark.bash              ← multi-model evaluation script
├── local/
│   └── pretrain_qwen3.bash             ← local single-GPU training
├── dataset_utils/                      ← dataset download & preprocessing
├── configs/
│   └── dataset_paths.json
└── docs/
    └── dataset.md
```

## Known Issues & Fixes

| Issue | Fix |
|---|---|
| `av` fails to install (no FFmpeg) | `pip install av --prefer-binary` |
| DeepSpeed `CUDA_HOME does not exist` | `export DS_SKIP_CUDA_CHECK=1` |
| HuggingFace network timeout on compute node | `HF_HUB_OFFLINE=1` + `TRANSFORMERS_CACHE=<path>` |
| `export` fails inside `bsub` string | Use inline env vars: `eval "VAR=val python script.py"` |
| Projector checkpoint has no tokenizer | Fixed in `builder.py` — auto-resolves base model from HF cache |
| `OSError: no pytorch_model.bin found` in projector dir | Fixed in `builder.py` — loads LLM weights from cached base model, injects projector |

## Acknowledgement

This project is based on [LLaVA-NeXT](https://github.com/LLaVA-VL/LLaVA-NeXT).
