#!/bin/bash

# Configuration
IMAGE_FILE="/home/phd_li/git_repo/RoadLLM/test_images/urban.jpg" # Or change to "image.jpg" as you requested
PROMPT="Describe the image"
OUTPUT_JSON="comparison_results.json"

# ---------------------------------------------------------------------------
# NOTE: The checkpoint directories under checkpoints/projectors/ contain only
# the trained MLP projector (mm_projector.bin) + config.json — NOT the full
# LLM weights.  Two ways to run evaluation:
#
# Option A (this script) — two-path loading via evaluate_roadllm.py:
#   Pass both --model-base (base LLM) and --model-path (projector dir).
#   builder.py handles combining them at load time.
#
# Option B — merged checkpoint for lmms-eval (one-path loading):
#   First create a self-contained checkpoint with:
#     python scripts/merge_projector_weights.py \
#       --projector-path ./checkpoints/projectors/<run_name> \
#       --model-base Qwen/Qwen3-8B \
#       --output-path ./checkpoints/merged/<run_name>-merged
#   Then run lmms-eval pointing to --output-path (no model_base needed).
# ---------------------------------------------------------------------------

# Clear previous results
if [ -f "$OUTPUT_JSON" ]; then
    rm "$OUTPUT_JSON"
fi

echo "Starting Evaluation..."
echo "Image: $IMAGE_FILE"
echo "Prompt: $PROMPT"
echo "------------------------------------------------"

# --- Define Models (Extracted from job_cli.bash) ---

# 0.6B Model
echo "Running 0.6B Model..."
TRANSFORMERS_CACHE=/home/phd_li/.cache/huggingface/hub HF_HUB_OFFLINE=1 python -m llava.serve.evaluate_roadllm \
    --conv-mode qwen_3 \
    --model-alias "0.6B" \
    --model-base "/home/phd_li/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca" \
    --model-path "/home/phd_li/git_repo/RoadLLM/checkpoints/projectors/roadllm-llava-openai_clip-vit-large-patch14-336-Qwen_Qwen3-0.6B-mlp2x_gelu-pretrain-full-2gpus-5epoches" \
    --image-file "$IMAGE_FILE" \
    --prompt "$PROMPT" \
    --output-file "$OUTPUT_JSON"

# 1.7B Model
echo "Running 1.7B Model..."
TRANSFORMERS_CACHE=/home/phd_li/.cache/huggingface/hub HF_HUB_OFFLINE=1 python -m llava.serve.evaluate_roadllm \
    --conv-mode qwen_3 \
    --model-alias "1.7B" \
    --model-base "/home/phd_li/.cache/huggingface/hub/models--Qwen--Qwen3-1.7B/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e" \
    --model-path "/home/phd_li/git_repo/RoadLLM/checkpoints/projectors/roadllm-llava-openai_clip-vit-large-patch14-336-Qwen_Qwen3-1.7B-mlp2x_gelu-pretrain-full-2gpus-5epoches" \
    --image-file "$IMAGE_FILE" \
    --prompt "$PROMPT" \
    --output-file "$OUTPUT_JSON"

# 4B Model
echo "Running 4B Model..."
TRANSFORMERS_CACHE=/home/phd_li/.cache/huggingface/hub HF_HUB_OFFLINE=1 python -m llava.serve.evaluate_roadllm \
    --conv-mode qwen_3 \
    --model-alias "4B" \
    --model-base "/home/phd_li/.cache/huggingface/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c" \
    --model-path "/home/phd_li/git_repo/RoadLLM/checkpoints/projectors/roadllm-llava-openai_clip-vit-large-patch14-336-Qwen_Qwen3-4B-mlp2x_gelu-pretrain-full-4gpus-5epoches" \
    --image-file "$IMAGE_FILE" \
    --prompt "$PROMPT" \
    --output-file "$OUTPUT_JSON"

# 8B Model (CLIP, Option A: two-path loading)
echo "Running 8B Model..."
TRANSFORMERS_CACHE=/home/phd_li/.cache/huggingface/hub HF_HUB_OFFLINE=1 python -m llava.serve.evaluate_roadllm \
    --conv-mode qwen_3 \
    --model-alias "8B" \
    --model-base "/home/phd_li/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218" \
    --model-path "/home/phd_li/git_repo/RoadLLM/checkpoints/projectors/roadllm-llava-openai_clip-vit-large-patch14-336-Qwen_Qwen3-8B-mlp2x_gelu-pretrain-full-4gpus-5epoches" \
    --image-file "$IMAGE_FILE" \
    --prompt "$PROMPT" \
    --output-file "$OUTPUT_JSON"

# 8B SigLIP Model (Option A: two-path loading via evaluate_roadllm.py)
# The projector dir only has mm_projector.bin; model-base supplies the LLM weights.
echo "Running 8B SigLIP Model (two-path)..."
TRANSFORMERS_CACHE=/home/phd_li/.cache/huggingface/hub HF_HUB_OFFLINE=1 python -m llava.serve.evaluate_roadllm \
    --conv-mode qwen_3 \
    --model-alias "8B-siglip" \
    --model-base "/home/phd_li/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218" \
    --model-path "/home/phd_li/git_repo/RoadLLM/checkpoints/projectors/roadllm-llava-google_siglip-so400m-patch14-384-Qwen_Qwen3-8B-mlp2x_gelu-pretrain-full-4gpus-5epoches" \
    --image-file "$IMAGE_FILE" \
    --prompt "$PROMPT" \
    --output-file "$OUTPUT_JSON"

# ---------------------------------------------------------------------------
# Option B: lmms-eval with a merged checkpoint (one-path loading).
#
# Step 1 — create the merged checkpoint once:
#   python scripts/merge_projector_weights.py \
#     --projector-path /home/phd_li/git_repo/RoadLLM/checkpoints/projectors/roadllm-llava-google_siglip-so400m-patch14-384-Qwen_Qwen3-8B-mlp2x_gelu-pretrain-full-4gpus-5epoches \
#     --model-base Qwen/Qwen3-8B \
#     --output-path /home/phd_li/git_repo/RoadLLM/checkpoints/merged/roadllm-llava-qwen3-8b-siglip-merged
#
# Step 2 — run lmms-eval pointing at the merged dir (no model_base needed):
#   lmms-eval \
#     --model llava \
#     --model_args pretrained=/home/phd_li/git_repo/RoadLLM/checkpoints/merged/roadllm-llava-qwen3-8b-siglip-merged,conv_template=qwen_3 \
#     --tasks <your_task> \
#     --batch_size 1 \
#     --output_path ./lmms_results
# ---------------------------------------------------------------------------

echo "Done! Results saved to $OUTPUT_JSON"