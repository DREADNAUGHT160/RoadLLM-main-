#!/usr/bin/env bash
# =============================================================================
# run_eval.sh — RoadLLM full evaluation pipeline
# =============================================================================
#
# Workflow:
#   1. Merges the projector-only checkpoint with Qwen3-8B base weights
#      (skipped if ./checkpoints/merged/roadllm-full already exists)
#   2. Runs lmms-eval on five benchmark tasks and saves results
#
# Tasks evaluated:
#   gqa, mmbench_en_dev, pope, seedbench, vqav2_val
#
# Requirements:
#   pip install lmms-eval
#   HuggingFace access to Qwen/Qwen3-8B and google/siglip-so400m-patch14-384
#
# Usage (from repo root):
#   bash run_eval.sh
#
# Override any default via env vars:
#   PROJECTOR_PATH=<path> MODEL_BASE=Qwen/Qwen3-8B MERGED_PATH=<path> bash run_eval.sh
# =============================================================================

set -euo pipefail

# ── Configurable via environment variables ────────────────────────────────────
# Projector lives in the original RoadLLM repo; this script runs from RoadLLM-main-.
# MODEL_BASE uses the local HF cache snapshot path (required when HF_HUB_OFFLINE=1).
PROJECTOR_PATH="${PROJECTOR_PATH:-/home/phd_li/git_repo/RoadLLM/checkpoints/projectors/roadllm-llava-openai_clip-vit-large-patch14-336-Qwen_Qwen3-8B-mlp2x_gelu-pretrain-full-4gpus-5epoches}"
MODEL_BASE="${MODEL_BASE:-/home/phd_li/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218}"
MERGED_PATH="${MERGED_PATH:-/home/phd_li/git_repo/RoadLLM-main-/checkpoints/merged/roadllm-full}"
RESULTS_DIR="${RESULTS_DIR:-./test_results}"
DTYPE="${DTYPE:-bfloat16}"
BATCH_SIZE="${BATCH_SIZE:-1}"

# Tasks — comma-separated lmms-eval task identifiers
# Adjust task names if your lmms-eval version uses different identifiers.
TASKS="${TASKS:-gqa,mmbench_en_dev,pope,seedbench,vqav2_val}"

# NOTE on model_name:
#   lmms-eval derives model_name from the last component of MERGED_PATH
#   ("roadllm-full"), which does not contain "qwen" or "llava".
#   builder.py routes model loading based on keywords in the model name:
#     - "llava" or is_multimodal=True  → multimodal branch
#     - "qwen3"                        → LlavaQwen3ForCausalLM branch
#   We override model_name so builder.py takes the correct path.
MODEL_NAME_OVERRIDE="roadllm-llava-qwen3-8b"
# ─────────────────────────────────────────────────────────────────────────────

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$SCRIPT_DIR"

# Ensure `import llava` resolves to this repo
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

# ── Banner ────────────────────────────────────────────────────────────────────
echo ""
echo "=================================================================="
echo "  RoadLLM Evaluation Pipeline"
echo "=================================================================="
echo "  Projector dir : $PROJECTOR_PATH"
echo "  Base LLM      : $MODEL_BASE"
echo "  Merged output : $MERGED_PATH"
echo "  Results dir   : $RESULTS_DIR"
echo "  Tasks         : $TASKS"
echo "  Dtype         : $DTYPE"
echo "  Batch size    : $BATCH_SIZE"
echo "=================================================================="

# ── Step 1: Merge projector + base LLM ───────────────────────────────────────
echo ""
echo "────────────────────────────────────────────────────────────────"
echo "  STEP 1/2 — Merge projector weights with base LLM"
echo "────────────────────────────────────────────────────────────────"

# Check for both config.json AND at least one .safetensors shard
SAFETENSORS_COUNT=$(find "$MERGED_PATH" -name "*.safetensors" 2>/dev/null | wc -l || echo 0)

if [ -f "$MERGED_PATH/config.json" ] && [ "$SAFETENSORS_COUNT" -gt 0 ]; then
    echo ""
    echo "  Merged checkpoint already exists — skipping."
    echo "  $MERGED_PATH  ($SAFETENSORS_COUNT shard(s))"
else
    echo ""
    if [ ! -d "$PROJECTOR_PATH" ]; then
        echo "  ERROR: Projector checkpoint directory not found:"
        echo "         $PROJECTOR_PATH"
        echo ""
        echo "  Set the PROJECTOR_PATH environment variable or confirm"
        echo "  the training completed successfully."
        exit 1
    fi

    if [ ! -f "$PROJECTOR_PATH/mm_projector.bin" ]; then
        echo "  ERROR: mm_projector.bin not found in:"
        echo "         $PROJECTOR_PATH"
        echo ""
        echo "  The checkpoint must be produced by LLaVA pretrain with"
        echo "  --mm_tunable_parts=mm_mlp_adapter"
        exit 1
    fi

    echo "  Running merge_model.py …"
    echo ""
    python "$REPO_ROOT/merge_model.py" \
        --projector-path "$PROJECTOR_PATH" \
        --model-base     "$MODEL_BASE" \
        --output-path    "$MERGED_PATH" \
        --dtype          "$DTYPE"

    echo ""
    echo "  Merge finished: $MERGED_PATH"
fi

# ── Step 2: lmms-eval ─────────────────────────────────────────────────────────
echo ""
echo "────────────────────────────────────────────────────────────────"
echo "  STEP 2/2 — Running lmms-eval"
echo "────────────────────────────────────────────────────────────────"
echo ""

# Check lmms-eval is importable
if ! python -c "import lmms_eval" 2>/dev/null; then
    echo "  ERROR: lmms-eval is not installed."
    echo "         Install with: pip install lmms-eval"
    exit 1
fi

LMMS_EVAL_VERSION=$(python -c "import lmms_eval; print(lmms_eval.__version__)" 2>/dev/null || echo "unknown")
echo "  lmms-eval version : $LMMS_EVAL_VERSION"
echo "  Model args        : pretrained=$MERGED_PATH,model_name=$MODEL_NAME_OVERRIDE,conv_template=qwen_3"
echo ""

mkdir -p "$RESULTS_DIR"

# Build model_args string
# - pretrained   : path to the merged checkpoint
# - model_name   : override so builder.py routes to LlavaQwen3ForCausalLM
# - conv_template: qwen_3 (matches conv_qwen_3 in llava/conversation.py)
# - attn_implementation: sdpa (matches training; change to flash_attention_2
#                         if you have flash-attn installed)
MODEL_ARGS="pretrained=${MERGED_PATH},model_name=${MODEL_NAME_OVERRIDE},conv_template=qwen_3,attn_implementation=sdpa"

python -m lmms_eval \
    --model      llava \
    --model_args "$MODEL_ARGS" \
    --tasks      "$TASKS" \
    --batch_size "$BATCH_SIZE" \
    --log_samples \
    --log_samples_suffix roadllm_eval \
    --output_path "$RESULTS_DIR"

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "=================================================================="
echo "  DONE"
echo "=================================================================="
echo "  Results written to: $RESULTS_DIR"
echo ""
echo "  Result files:"
find "$RESULTS_DIR" -name "*.json" 2>/dev/null | sort | while read -r f; do
    echo "    $f"
done
echo "=================================================================="
