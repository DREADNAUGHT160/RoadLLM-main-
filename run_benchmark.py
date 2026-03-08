#!/usr/bin/env python3
"""
RoadLLM VQA Benchmark Runner
=============================
Run standard VQA benchmarks on a trained RoadLLM / LLaVA-NeXT checkpoint.

Usage:
    python run_benchmark.py
    python run_benchmark.py --tasks vqav2_val,pope
    python run_benchmark.py --load_in_4bit
    python run_benchmark.py --checkpoint ./checkpoints/merged/roadllm-full

The checkpoint must be a fully-merged directory produced by merge_model.py.
Projector-only checkpoints (mm_projector.bin) are not supported here —
use run_eval.sh which calls merge_model.py first, or evaluate_roadllm.py
for single-image inference.
"""
import argparse
import os
import subprocess
import sys
import json
import glob
from datetime import datetime

# ──────────────────────────────────────────────────────────────
# CONFIGURATION — edit these before running
# ──────────────────────────────────────────────────────────────

# Path to the fully-merged RoadLLM checkpoint (produced by merge_model.py).
# NOT the base LLM HuggingFace ID ("Qwen/Qwen3-8B") — that won't work
# with HF_HUB_OFFLINE=1 and has no projector weights.
CHECKPOINT = "./checkpoints/merged/roadllm-full"

# model_name override passed to lmms-eval so builder.py routes to
# LlavaQwen3ForCausalLM (the merged dir is named "roadllm-full" which lacks
# the "qwen3"/"llava" keywords that builder.py uses for routing).
MODEL_NAME_OVERRIDE = "roadllm-llava-qwen3-8b"

# Conversation template — must match llava/conversation.py conv_qwen_3.
CONV_TEMPLATE = "qwen_3"

# Attention implementation — sdpa matches training; use flash_attention_2
# if flash-attn is installed.
ATTN_IMPL = "sdpa"

# Confirmed working cache path from your bsub command.
TRANSFORMERS_CACHE = "/home/phd_li/.cache/huggingface/hub"

# ──────────────────────────────────────────────────────────────

DEFAULT_TASKS = [
    "vqav2_val",        # VQAv2       — general visual question answering
    "gqa",              # GQA         — spatial & compositional reasoning
    "mmbench_en_dev",   # MMBench     — comprehensive multi-modal understanding
    "pope",             # POPE        — hallucination evaluation
    "seedbench",        # SeedBench   — perception & cognition
]
OUTPUT_BASE = "./results"


def parse_args():
    parser = argparse.ArgumentParser(description="RoadLLM VQA Benchmark Runner")
    parser.add_argument(
        "--checkpoint", type=str, default=CHECKPOINT,
        help="Path to fully-merged RoadLLM checkpoint (produced by merge_model.py)",
    )
    parser.add_argument(
        "--tasks", type=str, default=",".join(DEFAULT_TASKS),
        help="Comma-separated benchmark tasks",
    )
    parser.add_argument(
        "--batch_size", type=int, default=1,
        help="Inference batch size (default: 1)",
    )
    parser.add_argument(
        "--num_gpus", type=int, default=1,
        help="Number of GPUs (default: 1)",
    )
    parser.add_argument(
        "--output_path", type=str, default=None,
        help="Results output directory",
    )
    parser.add_argument(
        "--load_in_4bit", action="store_true",
        help="4-bit quantization (use if VRAM is tight)",
    )
    parser.add_argument(
        "--log_samples", action="store_true", default=True,
        help="Save per-sample outputs",
    )
    return parser.parse_args()


def setup_environment():
    """
    Set all required environment variables for offline HPC use.
    Matches the pattern from your bsub command:
      TRANSFORMERS_CACHE=... HF_HUB_OFFLINE=1
    """
    os.environ["TRANSFORMERS_CACHE"]    = TRANSFORMERS_CACHE
    os.environ["HF_DATASETS_CACHE"]     = os.path.join(TRANSFORMERS_CACHE, "datasets")
    os.environ["HUGGINGFACE_HUB_CACHE"] = TRANSFORMERS_CACHE
    os.environ["HF_HUB_OFFLINE"]        = "1"
    os.environ["TRANSFORMERS_OFFLINE"]  = "1"
    os.environ["HF_DATASETS_OFFLINE"]   = "1"
    os.environ["DS_SKIP_CUDA_CHECK"]    = "1"
    print(f"  TRANSFORMERS_CACHE  = {TRANSFORMERS_CACHE}")
    print(f"  HF_HUB_OFFLINE      = 1")
    print(f"  DS_SKIP_CUDA_CHECK  = 1")


def check_checkpoint(checkpoint_path):
    """Validate the checkpoint is a local merged directory (not a HF model ID)."""
    if not os.path.isdir(checkpoint_path):
        print(f"\n  ERROR: Checkpoint directory not found: {checkpoint_path}")
        print()
        print("  The checkpoint must be a fully-merged local directory produced by:")
        print("    python merge_model.py \\")
        print("        --projector-path ./checkpoints/projectors/<run> \\")
        print("        --model-base Qwen/Qwen3-8B \\")
        print(f"        --output-path {checkpoint_path}")
        print()
        print("  Or run the full pipeline (merge + eval) with:")
        print("    bash run_eval.sh")
        sys.exit(1)

    config_path = os.path.join(checkpoint_path, "config.json")
    if not os.path.exists(config_path):
        print(f"  WARNING: No config.json in {checkpoint_path} — is this a valid checkpoint?")
    else:
        # Sanity-check: confirm it's a LlavaQwen3 config, not the raw base LLM
        try:
            with open(config_path) as f:
                cfg = json.load(f)
            model_type = cfg.get("model_type", "")
            if model_type not in ("llava_qwen3", "llava"):
                print(f"  WARNING: config.json has model_type={model_type!r}.")
                print("  Expected llava_qwen3. Make sure this is a merged RoadLLM checkpoint.")
        except Exception:
            pass
        print(f"  Checkpoint OK: {checkpoint_path}")


def check_lmms_eval():
    try:
        import lmms_eval  # noqa: F401
        print("  lmms-eval installed")
    except ImportError:
        print("  ERROR: lmms-eval not installed. Run:")
        print("      pip install av --prefer-binary")
        print("      cd ~/git_repo/lmms-eval && pip install -e .")
        sys.exit(1)


def check_cache():
    """Check the HF cache exists and list cached models."""
    if not os.path.isdir(TRANSFORMERS_CACHE):
        print(f"  WARNING: Cache dir not found: {TRANSFORMERS_CACHE}")
        return
    cached = os.listdir(TRANSFORMERS_CACHE)
    model_dirs = [d for d in cached if d.startswith("models--")]
    print(f"  HF cache: {len(model_dirs)} model(s) cached")
    for m in model_dirs:
        print(f"    {m}")


def build_command(args, output_path, tasks_str):
    # Build model_args to match run_eval.sh:
    #   pretrained, model_name, conv_template, attn_implementation
    # model_name is required so builder.py routes to LlavaQwen3ForCausalLM.
    model_args = (
        f"pretrained={args.checkpoint}"
        f",model_name={MODEL_NAME_OVERRIDE}"
        f",conv_template={CONV_TEMPLATE}"
        f",attn_implementation={ATTN_IMPL}"
    )
    if args.load_in_4bit:
        model_args += ",load_in_4bit=True"

    if args.num_gpus > 1:
        cmd = [
            "accelerate", "launch",
            f"--num_processes={args.num_gpus}",
            "-m", "lmms_eval",
        ]
    else:
        cmd = [sys.executable, "-m", "lmms_eval"]

    cmd += [
        "--model",       "llava",
        "--model_args",  model_args,
        "--tasks",       tasks_str,
        "--batch_size",  str(args.batch_size),
        "--output_path", output_path,
    ]
    if args.log_samples:
        cmd += ["--log_samples", "--log_samples_suffix", "roadllm_eval"]
    return cmd


def print_results_summary(output_path):
    print("\n" + "=" * 65)
    print("  BENCHMARK RESULTS SUMMARY")
    print("=" * 65)

    result_files = glob.glob(os.path.join(output_path, "**", "*.json"), recursive=True)
    result_files = [f for f in result_files if "samples_" not in os.path.basename(f)]

    all_results = {}
    for f in result_files:
        try:
            with open(f) as fp:
                data = json.load(fp)
            if "results" in data:
                all_results.update(data["results"])
        except Exception:
            continue

    if not all_results:
        print(f"  Results saved to: {output_path}")
        return

    print(f"  {'Task':<25} {'Metric':<30} {'Score':>10}")
    print("  " + "-" * 65)
    for task, metrics in sorted(all_results.items()):
        for metric, value in metrics.items():
            if isinstance(value, float):
                print(f"  {task:<25} {metric:<30} {value*100:>9.2f}%")
            elif isinstance(value, (int, str)):
                print(f"  {task:<25} {metric:<30} {str(value):>10}")
    print("=" * 65)
    print(f"  Full results: {output_path}")


def main():
    args   = parse_args()
    ts     = datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = args.output_path or os.path.join(OUTPUT_BASE, f"run_{ts}")
    tasks  = ",".join(t.strip() for t in args.tasks.split(","))

    os.makedirs(outdir, exist_ok=True)

    print("\n" + "=" * 65)
    print("  RoadLLM VQA Benchmark Runner")
    print("=" * 65)
    print(f"  Checkpoint  : {args.checkpoint}")
    print(f"  Model name  : {MODEL_NAME_OVERRIDE}")
    print(f"  Conv tmpl   : {CONV_TEMPLATE}")
    print(f"  Tasks       : {tasks}")
    print(f"  Batch size  : {args.batch_size}")
    print(f"  GPUs        : {args.num_gpus}")
    print(f"  4-bit       : {args.load_in_4bit}")
    print(f"  Output      : {outdir}")
    print("=" * 65 + "\n")

    setup_environment()
    check_cache()
    check_checkpoint(args.checkpoint)
    check_lmms_eval()

    cmd = build_command(args, outdir, tasks)
    print("\n  Running:")
    print("    " + " \\\n      ".join(cmd) + "\n")

    # Save run metadata
    with open(os.path.join(outdir, "run_command.txt"), "w") as f:
        f.write(" \\\n  ".join(cmd) + "\n\n")
        f.write(f"# {ts}\n# Checkpoint: {args.checkpoint}\n# Tasks: {tasks}\n")

    try:
        subprocess.run(cmd, check=True, env=os.environ.copy())
    except subprocess.CalledProcessError as e:
        print(f"\n  Failed with exit code {e.returncode}")
        sys.exit(e.returncode)
    except KeyboardInterrupt:
        print("\n  Interrupted.")
        sys.exit(1)

    print_results_summary(outdir)


if __name__ == "__main__":
    main()
