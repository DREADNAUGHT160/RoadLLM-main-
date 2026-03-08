#!/usr/bin/env python3
"""
road_eval.py — RoadLLM evaluation on local road datasets
=========================================================
Runs inference and computes accuracy on datasets already on disk.
No internet, no merging, no lmms-eval required.

Supported datasets
------------------
  rdd2022     Road Damage Detection 2022 — binary damaged/undamaged
               + damage type (crack / pothole)
  bdd100k     BDD100K — time-of-day, weather, scene classification
  cityscapes  Cityscapes — open-ended scene description (qualitative)
  mapillary   Mapillary Vistas — open-ended scene description (qualitative)
  r2s100k     R2S100K — binary road-surface defect detection

Usage
-----
  python road_eval.py --dataset rdd2022 --data-root /path/to/RDD2022
  python road_eval.py --dataset bdd100k --data-root /path/to/bdd100k
  python road_eval.py --dataset rdd2022 --max-samples 200 --output results/rdd.json

Two-path loading (no merge needed):
  --model-path  projector-only checkpoint  (mm_projector.bin + config.json)
  --model-base  local Qwen3-8B snapshot path
"""

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from xml.etree import ElementTree as ET

# ── Force offline before HuggingFace imports ─────────────────────────────────
os.environ["HF_HUB_OFFLINE"]       = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"]  = "1"
os.environ["DS_SKIP_CUDA_CHECK"]   = "1"
# ─────────────────────────────────────────────────────────────────────────────

import torch
from PIL import Image

_REPO_ROOT = Path(__file__).resolve().parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from transformers import AutoTokenizer
from llava.model.builder import load_pretrained_model
from llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates

# ── HPC defaults ──────────────────────────────────────────────────────────────
_DEFAULT_MODEL_PATH = (
    "/home/phd_li/git_repo/RoadLLM/checkpoints/projectors/"
    "roadllm-llava-openai_clip-vit-large-patch14-336-"
    "Qwen_Qwen3-8B-mlp2x_gelu-pretrain-full-4gpus-5epoches"
)
_DEFAULT_MODEL_BASE = (
    "/home/phd_li/.cache/huggingface/hub/"
    "models--Qwen--Qwen3-8B/snapshots/"
    "b968826d9c46dd6066d109eabc6255188de91218"
)
_TRANSFORMERS_CACHE = "/home/phd_li/.cache/huggingface/hub"
os.environ.setdefault("TRANSFORMERS_CACHE", _TRANSFORMERS_CACHE)


# =============================================================================
# Dataset loaders — each returns a list of dicts:
#   {"image": Path, "question": str, "answer": str|None, "meta": dict}
# answer=None means qualitative (no accuracy computed)
# =============================================================================

def load_rdd2022(data_root: Path, max_samples: int, seed: int):
    """
    RDD2022 COCO layout:
      <data_root>/
        annotations/   ← JSON files (instances_train2017.json etc.)
        train2017/     ← training images
        val2017/       ← validation images

    Also handles Pascal VOC XML layout (per-image .xml files).

    Damage categories: D00 longitudinal crack, D10 transverse crack,
                       D20 alligator crack,    D40 pothole
    """
    _TYPE_MAP = {
        "D00": "longitudinal crack",
        "D10": "transverse crack",
        "D20": "alligator crack",
        "D40": "pothole",
    }

    # ── Try COCO JSON annotations first ──────────────────────────────────────
    ann_dir = data_root / "annotations"
    json_files = sorted(ann_dir.glob("*.json")) if ann_dir.exists() else []

    # image_id → set of damage class names
    img_damage: dict = {}
    # image filename → image_id
    fname_to_id: dict = {}

    for jf in json_files:
        try:
            with open(jf) as f:
                coco = json.load(f)
            # Build category id → name map
            cat_map = {c["id"]: c["name"] for c in coco.get("categories", [])}
            for img in coco.get("images", []):
                fname_to_id[img["file_name"]] = img["id"]
                img_damage.setdefault(img["id"], set())
            for ann in coco.get("annotations", []):
                iid = ann["image_id"]
                cname = cat_map.get(ann["category_id"], "")
                if cname:
                    img_damage.setdefault(iid, set()).add(cname)
        except Exception:
            pass

    # ── Collect all images ────────────────────────────────────────────────────
    img_dirs = [data_root / "train2017", data_root / "val2017"]
    # Fall back to searching entire tree if the expected dirs don't exist
    if not any(d.exists() for d in img_dirs):
        all_images = sorted(data_root.rglob("*.jpg")) + sorted(data_root.rglob("*.JPG"))
    else:
        all_images = []
        for d in img_dirs:
            if d.exists():
                all_images += sorted(d.glob("*.jpg")) + sorted(d.glob("*.JPG"))

    items = []
    for img_path in all_images:
        # ── Get damage labels ─────────────────────────────────────────────────
        damage_classes: set = set()

        # 1. COCO JSON
        iid = fname_to_id.get(img_path.name)
        if iid is not None:
            damage_classes = img_damage.get(iid, set())
        else:
            # 2. Fall back to Pascal VOC XML (same dir or annotations/)
            xml_candidates = [
                img_path.with_suffix(".xml"),
                ann_dir / img_path.with_suffix(".xml").name,
            ]
            xml_path = next((p for p in xml_candidates if p.exists()), None)
            if xml_path:
                try:
                    tree = ET.parse(xml_path)
                    for obj in tree.findall(".//object"):
                        name = obj.findtext("name", "").strip()
                        if name:
                            damage_classes.add(name)
                except Exception:
                    pass

        has_damage = len(damage_classes) > 0

        # Binary yes/no
        items.append({
            "image":    img_path,
            "question": "Is there visible road damage in this image? Answer with yes or no only.",
            "answer":   "yes" if has_damage else "no",
            "meta":     {"damage": list(damage_classes)},
            "task":     "binary",
        })

        # Damage type (only when we have labels)
        if has_damage:
            readable = [_TYPE_MAP.get(c, c) for c in sorted(damage_classes)]
            items.append({
                "image":    img_path,
                "question": (
                    "What type of road damage is visible in this image? "
                    "Choose the best answer: longitudinal crack, transverse crack, "
                    "alligator crack, pothole, or other."
                ),
                "answer":   readable[0],
                "meta":     {"all_damage": list(damage_classes)},
                "task":     "damage_type",
            })

    random.seed(seed)
    random.shuffle(items)
    return items[:max_samples]


def load_bdd100k(data_root: Path, max_samples: int, seed: int):
    """
    BDD100K directory structure:
      bdd100k/
        images/100k/val/  (or train/)
        labels/det_20/det_val.json  — detection labels with attributes
      OR
        labels/bdd100k_labels_images_val.json — full attribute labels
    """
    label_candidates = [
        data_root / "labels" / "bdd100k_labels_images_val.json",
        data_root / "labels" / "bdd100k_labels_images_train.json",
        data_root / "labels" / "det_20" / "det_val.json",
        data_root / "labels" / "det_20" / "det_train.json",
    ]
    label_file = next((f for f in label_candidates if f.exists()), None)
    if label_file is None:
        # Try any json in labels/
        found = list(data_root.glob("labels/**/*.json"))
        label_file = found[0] if found else None

    img_dirs = [
        data_root / "images" / "100k" / "val",
        data_root / "images" / "100k" / "train",
        data_root / "images" / "val",
        data_root / "images" / "train",
    ]
    img_dir = next((d for d in img_dirs if d.exists()), None)

    items = []

    if label_file and img_dir:
        with open(label_file) as f:
            labels = json.load(f)
        # Support both list-of-dicts and {"frames": [...]} formats
        frames = labels if isinstance(labels, list) else labels.get("frames", [])
        for frame in frames:
            img_name = frame.get("name", "")
            img_path = img_dir / img_name
            if not img_path.exists():
                continue
            attrs = frame.get("attributes", {})
            timeofday = attrs.get("timeofday", "")
            weather   = attrs.get("weather", "")
            scene     = attrs.get("scene", "")

            if timeofday:
                is_night = "night" in timeofday.lower() or "dawn/dusk" in timeofday.lower()
                items.append({
                    "image":    img_path,
                    "question": "Is this image taken at night or in low-light conditions? Answer with yes or no only.",
                    "answer":   "yes" if is_night else "no",
                    "meta":     {"timeofday": timeofday},
                    "task":     "timeofday",
                })
            if weather:
                items.append({
                    "image":    img_path,
                    "question": (
                        "What is the weather condition in this driving image? "
                        "Choose: clear, overcast, rainy, snowy, foggy, or partly cloudy."
                    ),
                    "answer":   weather.lower(),
                    "meta":     {"weather": weather},
                    "task":     "weather",
                })
            if scene:
                items.append({
                    "image":    img_path,
                    "question": (
                        "What type of road environment is shown? "
                        "Choose: city street, highway, residential, tunnel, or parking lot."
                    ),
                    "answer":   scene.lower(),
                    "meta":     {"scene": scene},
                    "task":     "scene",
                })
    else:
        # No labels found — qualitative only
        for img_path in sorted((img_dir or data_root).rglob("*.jpg"))[:max_samples]:
            items.append({
                "image":    img_path,
                "question": "Describe the driving scene and road conditions in this image.",
                "answer":   None,
                "meta":     {},
                "task":     "description",
            })

    random.seed(seed)
    random.shuffle(items)
    return items[:max_samples]


def load_r2s100k(data_root: Path, max_samples: int, seed: int):
    """
    R2S100K — Road to Scene. Tries common layouts:
      images/ + labels/ or annotations/
    Falls back to binary defect detection from filename conventions.
    """
    items = []
    img_dirs = [data_root / "images", data_root / "imgs", data_root]
    img_dir  = next((d for d in img_dirs if d.is_dir() and list(d.glob("*.jpg"))), None) or data_root

    for img_path in sorted(img_dir.rglob("*.jpg")):
        # Infer defect presence from directory name (common convention)
        parent = img_path.parent.name.lower()
        has_defect = any(k in parent for k in ["defect", "damage", "crack", "pothole", "positive"])
        no_defect  = any(k in parent for k in ["normal", "negative", "clean", "good"])
        answer = "yes" if has_defect else ("no" if no_defect else None)
        items.append({
            "image":    img_path,
            "question": "Is there a road surface defect or damage visible in this image? Answer with yes or no only.",
            "answer":   answer,
            "meta":     {"dir": parent},
            "task":     "binary",
        })

    random.seed(seed)
    random.shuffle(items)
    return items[:max_samples]


def load_qualitative(data_root: Path, max_samples: int, seed: int, dataset_name: str):
    """Generic loader for Cityscapes / Mapillary — no ground-truth labels needed."""
    imgs = sorted(data_root.rglob("*.jpg")) + sorted(data_root.rglob("*.png"))
    random.seed(seed)
    random.shuffle(imgs)
    questions = [
        "Describe the road and traffic conditions visible in this image.",
        "What potential hazards or obstacles are visible on or near the road?",
        "Is the road surface in good condition? Describe any issues you see.",
    ]
    items = []
    for i, img_path in enumerate(imgs[:max_samples]):
        items.append({
            "image":    img_path,
            "question": questions[i % len(questions)],
            "answer":   None,
            "meta":     {},
            "task":     "description",
        })
    return items


# =============================================================================
# Inference
# =============================================================================

def run_inference(model, tokenizer, image_processor, item: dict, conv_mode: str, device: str) -> str:
    img = Image.open(item["image"]).convert("RGB")
    image_tensor = process_images([img], image_processor, model.config)
    image_tensor = [t.to(device=device, dtype=torch.float16) for t in image_tensor]

    question = DEFAULT_IMAGE_TOKEN + "\n" + item["question"]
    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], question)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
    input_ids = input_ids.unsqueeze(0).to(device)

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_tensor,
            image_sizes=[img.size],
            do_sample=False,
            max_new_tokens=64,
            use_cache=True,
        )
    return tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()


def normalize(text: str) -> str:
    """Normalise model output for answer matching."""
    t = text.lower().strip().rstrip(".,!?")
    # For yes/no questions, grab first word
    first = t.split()[0] if t else ""
    if first in ("yes", "no"):
        return first
    return t


def match(prediction: str, answer: str) -> bool:
    pred = normalize(prediction)
    gt   = answer.lower().strip()
    return gt in pred or pred.startswith(gt)


# =============================================================================
# Main
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(description="RoadLLM road dataset evaluation")
    p.add_argument("--dataset",    required=True,
                   choices=["rdd2022", "bdd100k", "cityscapes", "mapillary", "r2s100k"],
                   help="Dataset to evaluate on")
    p.add_argument("--data-root",  required=True, type=Path,
                   help="Root directory of the dataset")
    p.add_argument("--model-path", default=_DEFAULT_MODEL_PATH,
                   help="Projector-only checkpoint directory")
    p.add_argument("--model-base", default=_DEFAULT_MODEL_BASE,
                   help="Local Qwen3-8B snapshot path")
    p.add_argument("--conv-mode",  default="qwen_3",
                   help="Conversation template (default: qwen_3)")
    p.add_argument("--max-samples", type=int, default=500,
                   help="Max images to evaluate (default: 500)")
    p.add_argument("--seed",        type=int, default=42)
    p.add_argument("--output",      type=Path, default=None,
                   help="Save results JSON to this path")
    p.add_argument("--device",      default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main():
    args = parse_args()

    print("\n" + "=" * 60)
    print("  RoadLLM Road Dataset Evaluation")
    print("=" * 60)
    print(f"  Dataset     : {args.dataset}")
    print(f"  Data root   : {args.data_root}")
    print(f"  Model path  : {args.model_path}")
    print(f"  Model base  : {args.model_base}")
    print(f"  Max samples : {args.max_samples}")
    print(f"  Device      : {args.device}")
    print("=" * 60 + "\n")

    # ── Load dataset ──────────────────────────────────────────────────────────
    loaders = {
        "rdd2022":    load_rdd2022,
        "bdd100k":    load_bdd100k,
        "r2s100k":    load_r2s100k,
        "cityscapes": lambda r, n, s: load_qualitative(r, n, s, "cityscapes"),
        "mapillary":  lambda r, n, s: load_qualitative(r, n, s, "mapillary"),
    }
    print("  Loading dataset …")
    items = loaders[args.dataset](args.data_root, args.max_samples, args.seed)
    if not items:
        print(f"  ERROR: No images found under {args.data_root}")
        sys.exit(1)
    print(f"  Loaded {len(items)} samples\n")

    # ── Load model (two-path, no merge needed) ────────────────────────────────
    print("  Loading model (two-path, projector + base LLM) …")
    model_name = get_model_name_from_path(args.model_path)
    # Override model_name to ensure builder.py routes to LlavaQwen3ForCausalLM
    if "qwen3" not in model_name.lower() and "qwen" not in model_name.lower():
        model_name = "roadllm-llava-qwen3-8b"

    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path  = args.model_path,
        model_base  = args.model_base,
        model_name  = model_name,
        device_map  = args.device,
        torch_dtype = "float16",
        attn_implementation = "sdpa",
    )
    model.eval()
    print("  Model loaded\n")

    # ── Run evaluation ────────────────────────────────────────────────────────
    results_by_task = {}
    all_results = []
    t0 = time.time()

    for i, item in enumerate(items):
        try:
            pred = run_inference(model, tokenizer, image_processor, item,
                                 args.conv_mode, args.device)
        except Exception as e:
            pred = f"ERROR: {e}"

        task = item["task"]
        correct = None
        if item["answer"] is not None:
            correct = match(pred, item["answer"])
            task_stats = results_by_task.setdefault(task, {"correct": 0, "total": 0})
            task_stats["total"] += 1
            if correct:
                task_stats["correct"] += 1

        all_results.append({
            "image":    str(item["image"]),
            "task":     task,
            "question": item["question"],
            "answer":   item["answer"],
            "pred":     pred,
            "correct":  correct,
            "meta":     item["meta"],
        })

        elapsed = time.time() - t0
        eta = elapsed / (i + 1) * (len(items) - i - 1)
        print(
            f"  [{i+1:>4}/{len(items)}]  {task:<15}  "
            f"pred={pred[:30]!r:<33}  gt={str(item['answer']):<20}  "
            f"{'OK' if correct else ('--' if correct is None else 'WRONG')}  "
            f"ETA {eta/60:.1f}m"
        )

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  RESULTS SUMMARY")
    print("=" * 60)
    overall_correct = overall_total = 0
    for task, stats in sorted(results_by_task.items()):
        acc = stats["correct"] / stats["total"] * 100
        print(f"  {task:<20} {stats['correct']:>4} / {stats['total']:<4}  {acc:>6.1f}%")
        overall_correct += stats["correct"]
        overall_total   += stats["total"]
    if overall_total:
        print(f"  {'OVERALL':<20} {overall_correct:>4} / {overall_total:<4}  "
              f"{overall_correct/overall_total*100:>6.1f}%")
    qual = sum(1 for r in all_results if r["answer"] is None)
    if qual:
        print(f"\n  {qual} qualitative samples (no accuracy — see output file for predictions)")
    print("=" * 60)

    # ── Save ──────────────────────────────────────────────────────────────────
    output_path = args.output or Path(f"./results/{args.dataset}_eval.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump({
            "dataset":   args.dataset,
            "model":     args.model_path,
            "n_samples": len(items),
            "by_task":   results_by_task,
            "overall":   {
                "correct": overall_correct,
                "total":   overall_total,
                "accuracy": overall_correct / overall_total if overall_total else None,
            },
            "samples": all_results,
        }, f, indent=2)
    print(f"\n  Full results → {output_path}")


if __name__ == "__main__":
    main()
