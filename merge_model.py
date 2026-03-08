#!/usr/bin/env python3
"""
merge_model.py — Merge a projector-only LLaVA checkpoint with Qwen3-8B into
a full, self-contained LlavaQwen3ForCausalLM checkpoint usable by lmms-eval.

Why this is needed
------------------
Training with --mm_tunable_parts="mm_mlp_adapter" saves only the MLP connector
(mm_projector.bin) + config.json to the checkpoint directory; the base LLM
weights are never written there.  lmms-eval calls from_pretrained(model_path)
directly and fails with:

    OSError: No file named pytorch_model.bin, model.safetensors ... found

This script produces a single directory containing:
  - model.safetensors  (Qwen3-8B LLM weights + trained MLP projector)
  - config.json        (LlavaQwen3Config with all mm_* fields)
  - tokenizer files    (copied from Qwen/Qwen3-8B)

lmms-eval can then load it with a single path and no model_base.

Usage
-----
# With defaults (RoadLLM 8B CLIP checkpoint from original RoadLLM repo):
python merge_model.py

# Or with explicit paths:
python merge_model.py \\
    --projector-path /home/phd_li/git_repo/RoadLLM/checkpoints/projectors/roadllm-llava-openai_clip-vit-large-patch14-336-Qwen_Qwen3-8B-mlp2x_gelu-pretrain-full-4gpus-5epoches \\
    --model-base /home/phd_li/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218 \\
    --output-path /home/phd_li/git_repo/RoadLLM-main-/checkpoints/merged/roadllm-full \\
    --dtype bfloat16
"""

import argparse
import os
import sys
from pathlib import Path

import torch

# ── Ensure the repo root is on sys.path so `import llava` resolves ──────────
_REPO_ROOT = Path(__file__).resolve().parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from transformers import AutoTokenizer
from llava.model.language_model.llava_qwen3 import LlavaQwen3Config, LlavaQwen3ForCausalLM

# ── Defaults for the RoadLLM 8B CLIP run ─────────────────────────────────────
# Projector lives in the original RoadLLM repo; this script runs from RoadLLM-main-.
# Model base points to the local HF cache snapshot (required when HF_HUB_OFFLINE=1).
_DEFAULT_PROJECTOR = (
    "/home/phd_li/git_repo/RoadLLM/checkpoints/projectors/"
    "roadllm-llava-openai_clip-vit-large-patch14-336-"
    "Qwen_Qwen3-8B-mlp2x_gelu-pretrain-full-4gpus-5epoches"
)
_DEFAULT_MODEL_BASE = (
    "/home/phd_li/.cache/huggingface/hub/"
    "models--Qwen--Qwen3-8B/snapshots/"
    "b968826d9c46dd6066d109eabc6255188de91218"
)
_DEFAULT_OUTPUT = "/home/phd_li/git_repo/RoadLLM-main-/checkpoints/merged/roadllm-full"


# ─────────────────────────────────────────────────────────────────────────────
def _banner(msg: str) -> None:
    print(f"\n{'─' * 60}")
    print(f"  {msg}")
    print(f"{'─' * 60}")


def merge(args: argparse.Namespace) -> None:
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    projector_bin = os.path.join(args.projector_path, "mm_projector.bin")

    # ── 1. Validate inputs ───────────────────────────────────────────────────
    _banner("Step 1 / 5 — Validating inputs")

    if not os.path.isdir(args.projector_path):
        raise FileNotFoundError(
            f"Projector checkpoint directory not found:\n  {args.projector_path}\n"
            "Pass --projector-path or set the PROJECTOR_PATH env var."
        )
    if not os.path.exists(projector_bin):
        raise FileNotFoundError(
            f"mm_projector.bin not found in:\n  {args.projector_path}\n"
            "The directory must be a LLaVA pretrain output with "
            "--mm_tunable_parts=mm_mlp_adapter."
        )
    if not os.path.exists(os.path.join(args.projector_path, "config.json")):
        raise FileNotFoundError(
            f"config.json not found in:\n  {args.projector_path}"
        )

    print(f"  Projector path : {args.projector_path}")
    print(f"  Base model     : {args.model_base}")
    print(f"  Output path    : {args.output_path}")
    print(f"  Weight dtype   : {args.dtype}")

    # ── 2. Load LlavaQwen3Config from the projector checkpoint ───────────────
    _banner("Step 2 / 5 — Loading LlavaQwen3Config")

    config = LlavaQwen3Config.from_pretrained(args.projector_path)
    print(f"  model_type        : {config.model_type}")
    print(f"  mm_vision_tower   : {getattr(config, 'mm_vision_tower', 'NOT SET')}")
    print(f"  mm_projector_type : {getattr(config, 'mm_projector_type', 'NOT SET')}")
    print(f"  mm_hidden_size    : {getattr(config, 'mm_hidden_size', 'NOT SET')}")
    print(f"  hidden_size       : {config.hidden_size}")

    # delay_load=True → LlavaMetaModel.__init__ skips downloading the vision
    # tower weights during this merge step; the tower loads lazily from
    # HuggingFace at inference time.
    config.delay_load = True

    # ── 3. Load base LLM with the LlavaQwen3 config ─────────────────────────
    _banner("Step 3 / 5 — Loading base LLM (Qwen3-8B)")
    print("  This downloads ~16 GB of weights if not already cached.\n")

    model = LlavaQwen3ForCausalLM.from_pretrained(
        args.model_base,
        config=config,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )
    print(f"\n  Loaded : {model.__class__.__name__}")
    n_params = sum(p.numel() for p in model.parameters()) / 1e9
    print(f"  Params : {n_params:.1f} B  (base LLM only, no projector yet)")

    # ── 4. Load and inject projector weights ─────────────────────────────────
    _banner("Step 4 / 5 — Injecting mm_projector weights")

    projector_weights = torch.load(projector_bin, map_location="cpu")
    projector_weights = {k: v.to(dtype) for k, v in projector_weights.items()}

    # Keys look like: "model.mm_projector.0.weight", "model.mm_projector.0.bias" …
    sample_keys = list(projector_weights.keys())[:6]
    print(f"  Tensors in mm_projector.bin : {len(projector_weights)}")
    print(f"  Sample keys : {sample_keys}")

    result = model.load_state_dict(projector_weights, strict=False)

    # missing_keys = LLM weights (expected — they come from base model)
    # unexpected_keys = truly unrecognised keys (should be empty)
    projector_keys_found = [
        k for k in projector_weights if k not in result.missing_keys
    ]
    print(f"\n  Projector tensors matched : {len(projector_keys_found)} / {len(projector_weights)}")
    if result.unexpected_keys:
        print(f"  [WARN] Unexpected keys (ignored) : {result.unexpected_keys}")

    # ── 5. Save merged checkpoint ────────────────────────────────────────────
    _banner("Step 5 / 5 — Saving merged checkpoint")

    # Reset delay_load so the merged model rebuilds the vision tower from
    # HuggingFace when loaded at inference time.
    config.delay_load = False
    model.config.delay_load = False

    os.makedirs(args.output_path, exist_ok=True)
    print(f"  Writing model.safetensors to: {args.output_path}")
    print("  (may take a few minutes for a ~16 GB model) …\n")
    model.save_pretrained(args.output_path, safe_serialization=True)

    print(f"  Saving tokenizer from: {args.model_base}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_base)
    tokenizer.save_pretrained(args.output_path)

    # ── Done ─────────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  MERGE COMPLETE")
    print("=" * 60)
    print(f"  Merged checkpoint : {args.output_path}")
    print()
    print("  Files written:")
    for f in sorted(os.listdir(args.output_path)):
        size_mb = os.path.getsize(os.path.join(args.output_path, f)) / 1e6
        print(f"    {f:<45}  {size_mb:>8.1f} MB")
    print()
    print("  Next step → run_eval.sh (or see below):")
    print(f"""
  python -m lmms_eval \\
      --model llava \\
      --model_args "pretrained={args.output_path},model_name=roadllm-llava-qwen3-8b,conv_template=qwen_3" \\
      --tasks gqa,mmbench_en_dev,pope,seedbench,vqav2_val \\
      --batch_size 1 \\
      --output_path ./test_results
""")


# ─────────────────────────────────────────────────────────────────────────────
def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Merge a projector-only LLaVA checkpoint with its base LLM.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--projector-path",
        default=os.environ.get("PROJECTOR_PATH", _DEFAULT_PROJECTOR),
        help=(
            "Directory containing mm_projector.bin + config.json "
            f"(default: {_DEFAULT_PROJECTOR})"
        ),
    )
    p.add_argument(
        "--model-base",
        default=os.environ.get("MODEL_BASE", _DEFAULT_MODEL_BASE),
        help=f"Base LLM — HuggingFace ID or local path (default: {_DEFAULT_MODEL_BASE})",
    )
    p.add_argument(
        "--output-path",
        default=os.environ.get("MERGED_PATH", _DEFAULT_OUTPUT),
        help=f"Output directory for the merged checkpoint (default: {_DEFAULT_OUTPUT})",
    )
    p.add_argument(
        "--dtype",
        default="bfloat16",
        choices=["bfloat16", "float16"],
        help="Weight dtype for loading and saving (default: bfloat16)",
    )
    return p.parse_args()


if __name__ == "__main__":
    merge(_parse_args())
