#!/usr/bin/env python3
"""
Merge a projector-only LLaVA checkpoint with its base LLM to produce a
full, self-contained LlavaQwen3ForCausalLM checkpoint loadable by lmms-eval
(or any HuggingFace from_pretrained call without a separate model_base).

Background
----------
Training with --mm_tunable_parts="mm_mlp_adapter" only saves the MLP
connector (mm_projector.bin) in the checkpoint directory, NOT the full
LLM weights.  lmms-eval calls from_pretrained(model_path) directly, so
it fails with:

    OSError: Error no file named pytorch_model.bin, model.safetensors...

This script combines:
  1. Base LLM weights  (e.g. Qwen/Qwen3-8B from HuggingFace)
  2. Trained projector (mm_projector.bin from the checkpoint directory)

and writes a single directory that lmms-eval can load with no model_base.

Usage
-----
python scripts/merge_projector_weights.py \\
    --projector-path ./checkpoints/projectors/roadllm-llava-google_siglip-so400m-patch14-384-Qwen_Qwen3-8B-mlp2x_gelu-pretrain-full-4gpus-5epoches \\
    --model-base Qwen/Qwen3-8B \\
    --output-path ./checkpoints/merged/roadllm-llava-qwen3-8b-siglip-merged

Then run lmms-eval pointing to --output-path (see bottom of this file).
"""

import argparse
import os
import sys

import torch

# Ensure the repo root is on the path so `llava` is importable
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from transformers import AutoTokenizer
from llava.model.language_model.llava_qwen3 import LlavaQwen3Config, LlavaQwen3ForCausalLM


def merge(args: argparse.Namespace) -> None:
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    projector_bin = os.path.join(args.projector_path, "mm_projector.bin")

    # ------------------------------------------------------------------
    # 1. Load the LlavaQwen3Config from the projector checkpoint.
    #    This config contains all multimodal fields:
    #      mm_vision_tower, mm_projector_type, mm_hidden_size, etc.
    # ------------------------------------------------------------------
    print(f"[1/5] Reading LlavaQwen3Config from: {args.projector_path}")
    llava_cfg = LlavaQwen3Config.from_pretrained(args.projector_path)
    print(f"      mm_vision_tower  : {getattr(llava_cfg, 'mm_vision_tower', 'NOT SET')}")
    print(f"      mm_projector_type: {getattr(llava_cfg, 'mm_projector_type', 'NOT SET')}")
    print(f"      mm_hidden_size   : {getattr(llava_cfg, 'mm_hidden_size', 'NOT SET')}")

    # Use delay_load=True so we don't download / initialise the vision
    # tower weights during the merge — they are not part of mm_projector.bin
    # and will be fetched from HuggingFace at inference time anyway.
    llava_cfg.delay_load = True

    # ------------------------------------------------------------------
    # 2. Load the base LLM with the LlavaQwen3 config.
    #    This initialises the full Qwen3 transformer + an empty mm_projector.
    # ------------------------------------------------------------------
    print(f"\n[2/5] Loading base LLM from: {args.model_base}")
    model = LlavaQwen3ForCausalLM.from_pretrained(
        args.model_base,
        config=llava_cfg,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )
    print(f"      Model class: {model.__class__.__name__}")
    print(f"      Dtype      : {dtype}")

    # ------------------------------------------------------------------
    # 3. Load and apply the trained projector weights.
    #    mm_projector.bin keys look like:
    #      "model.mm_projector.0.weight", "model.mm_projector.0.bias", ...
    # ------------------------------------------------------------------
    if not os.path.exists(projector_bin):
        raise FileNotFoundError(
            f"mm_projector.bin not found in {args.projector_path}.\n"
            "Make sure --projector-path points to the directory that contains it."
        )

    print(f"\n[3/5] Loading projector weights from: {projector_bin}")
    mm_projector_weights = torch.load(projector_bin, map_location="cpu")
    mm_projector_weights = {k: v.to(dtype) for k, v in mm_projector_weights.items()}

    # Print a sample of keys for diagnostics
    sample_keys = list(mm_projector_weights.keys())[:5]
    print(f"      Sample keys: {sample_keys}")

    incompatible = model.load_state_dict(mm_projector_weights, strict=False)
    if incompatible.missing_keys:
        print(f"      [WARN] Missing keys (not in projector .bin): {incompatible.missing_keys}")
    if incompatible.unexpected_keys:
        print(f"      [WARN] Unexpected keys (ignored): {incompatible.unexpected_keys}")
    print("      Projector weights applied successfully.")

    # ------------------------------------------------------------------
    # 4. Save tokenizer from the base model.
    # ------------------------------------------------------------------
    print(f"\n[4/5] Saving tokenizer from: {args.model_base}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_base)

    # ------------------------------------------------------------------
    # 5. Save the merged model.
    #    Reset delay_load so the vision tower is built at inference time.
    # ------------------------------------------------------------------
    llava_cfg.delay_load = False
    model.config.delay_load = False

    os.makedirs(args.output_path, exist_ok=True)
    print(f"\n[5/5] Saving merged model to: {args.output_path}")
    print("      (this may take a few minutes for large models)")
    model.save_pretrained(args.output_path, safe_serialization=True)
    tokenizer.save_pretrained(args.output_path)

    # ------------------------------------------------------------------
    # Done — print the updated eval command
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Merge complete!")
    print(f"Merged checkpoint: {args.output_path}")
    print("=" * 70)
    print("\nTo run lmms-eval on the merged checkpoint:")
    print(
        f"""
lmms-eval \\
    --model llava \\
    --model_args pretrained={args.output_path},conv_template=qwen_3 \\
    --tasks <your_task> \\
    --batch_size 1 \\
    --output_path ./results
"""
    )
    print("Or with evaluate_roadllm.py (no --model-base needed now):")
    print(
        f"""
python -m llava.serve.evaluate_roadllm \\
    --conv-mode qwen_3 \\
    --model-path {args.output_path} \\
    --image-file <your_image> \\
    --prompt "Describe the image" \\
    --output-file results.json
"""
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Merge a projector-only LLaVA checkpoint with its base LLM."
    )
    parser.add_argument(
        "--projector-path",
        type=str,
        required=True,
        help=(
            "Path to the projector-only checkpoint directory "
            "(must contain mm_projector.bin and config.json)."
        ),
    )
    parser.add_argument(
        "--model-base",
        type=str,
        required=True,
        help="HuggingFace model ID or local path of the base LLM (e.g. Qwen/Qwen3-8B).",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        required=True,
        help="Output directory for the merged, self-contained checkpoint.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["bfloat16", "float16"],
        help="Weight dtype used when loading and saving (default: bfloat16).",
    )
    args = parser.parse_args()
    merge(args)
