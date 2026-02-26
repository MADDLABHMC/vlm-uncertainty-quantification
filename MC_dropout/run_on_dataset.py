"""
Run MC Dropout on an entire dataset (e.g., Semantic Drone).

Processes all image/mask pairs, computes per-image metrics, and aggregates
dataset-level statistics.

Usage:
    python run_on_dataset.py --dataset-path /path/to/semantic_drone --output-dir outputs
    python run_on_dataset.py --dataset-path /path/to/semantic_drone --limit 10  # test on 10 images
    python run_on_dataset.py --dataset-path /path/to/semantic_drone --classes paved-area dirt grass --indices 1 2 3
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

from src.model import load_model
from src.data_utils import (
    load_image_and_mask,
    iter_dataset_pairs,
    load_dataset_classes,
    get_indices_for_classes,
)
from src.inference import mc_dropout_predict
from src.metrics import compute_predictions_and_entropy, compute_accuracy


def _get_device():
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def run_single_image(
    model,
    processor,
    image_path: Path,
    mask_path: Path,
    class_names: list[str],
    class_indices: list[int],
    n_samples: int,
    device: str,
) -> dict:
    """Run MC Dropout on a single image, return metrics dict."""
    image, ground_truth, _, _ = load_image_and_mask(
        str(image_path),
        str(mask_path),
        class_names=class_names,
        class_indices=class_indices,
    )

    mean_probs, std_probs, _ = mc_dropout_predict(
        model, processor, image, class_names,
        n_samples=n_samples, device=device, verbose=False
    )

    predictions, _, normalized_entropy = compute_predictions_and_entropy(mean_probs)
    mean_iou, per_class_iou = compute_accuracy(
        predictions, ground_truth, class_names, verbose=False
    )

    # Flatten per-class IoU to {class_name: iou} (skip classes with no pixels)
    per_class = {
        name: iou for name, (iou, count) in per_class_iou.items()
        if iou is not None
    }

    return {
        "image": image_path.name,
        "mean_iou": float(mean_iou),
        "mean_uncertainty": float(normalized_entropy.mean()),
        "per_class_iou": per_class,
    }


def run_dataset_evaluation(
    dataset_path: str | Path,
    n_samples: int = 30,
    dropout_rate: float = 0.1,
    limit: int | None = None,
    device: str | None = None,
    class_names: list[str] | None = None,
    class_indices: list[int] | None = None,
    verbose: bool = True,
) -> tuple[dict, list[dict]]:
    """
    Run MC Dropout on dataset and return aggregate + per_image results.

    Returns:
        (aggregate dict, per_image results list)
    """
    from tqdm import tqdm

    device = device or _get_device()
    dataset_path = Path(dataset_path)

    if verbose:
        print(f"\n  Dropout rate: {dropout_rate}")

    pairs = iter_dataset_pairs(dataset_path)
    if limit is not None:
        pairs = pairs[:limit]

    full_class_names, full_indices = load_dataset_classes(dataset_path)
    if class_names is not None:
        if class_indices is None:
            class_indices = get_indices_for_classes(full_class_names, class_names)
    else:
        class_names = full_class_names
        class_indices = full_indices

    model, processor = load_model(dropout_rate=dropout_rate)
    model.to(device)

    per_image_results = []
    total_start = time.time()

    for img_path, mask_path in tqdm(pairs, desc=f"  Images (p={dropout_rate})", disable=not verbose):
        start = time.time()
        result = run_single_image(
            model, processor,
            img_path, mask_path,
            class_names, class_indices,
            n_samples, device,
        )
        result["inference_time"] = time.time() - start
        per_image_results.append(result)

    total_time = time.time() - total_start

    mean_ious = [r["mean_iou"] for r in per_image_results]
    mean_uncertainties = [r["mean_uncertainty"] for r in per_image_results]

    per_class_ious: dict[str, list[float]] = {name: [] for name in class_names}
    for r in per_image_results:
        for name, iou in r.get("per_class_iou", {}).items():
            if iou is not None:
                per_class_ious.setdefault(name, []).append(iou)
    per_class_mean = {
        name: float(np.mean(ious)) if ious else None
        for name, ious in per_class_ious.items()
    }
    per_class_std = {
        name: float(np.std(ious)) if len(ious) > 1 else 0.0
        for name, ious in per_class_ious.items()
    }

    aggregate = {
        "dataset_path": str(dataset_path),
        "classes": class_names,
        "n_images": len(per_image_results),
        "n_samples": n_samples,
        "dropout_rate": dropout_rate,
        "device": device,
        "total_time_seconds": total_time,
        "time_per_image_seconds": total_time / len(per_image_results) if per_image_results else 0,
        "mean_iou": float(np.mean(mean_ious)),
        "std_iou": float(np.std(mean_ious)),
        "mean_uncertainty": float(np.mean(mean_uncertainties)),
        "std_uncertainty": float(np.std(mean_uncertainties)),
        "per_class_iou": {
            name: {"mean": per_class_mean.get(name), "std": per_class_std.get(name)}
            for name in class_names
            if per_class_mean.get(name) is not None
        },
    }

    return aggregate, per_image_results


def main(
    dataset_path: str,
    output_dir: str = "outputs",
    n_samples: int = 30,
    dropout_rate: float = 0.1,
    limit: int | None = None,
    device: str | None = None,
    save_per_image: bool = True,
    class_names: list[str] | None = None,
    class_indices: list[int] | None = None,
):
    """Run MC Dropout on entire dataset."""
    device = device or _get_device()
    dataset_path = Path(dataset_path)
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)

    print("=" * 60)
    print("MC DROPOUT - FULL DATASET EVALUATION")
    print("=" * 60)

    print("\n[1/5] Discovering dataset...")
    pairs = iter_dataset_pairs(dataset_path)
    if limit is not None:
        pairs = pairs[:limit]
        print(f"  Using subset: {len(pairs)} images (--limit {limit})")
    else:
        print(f"  Found {len(pairs)} image/mask pairs")

    full_class_names, full_indices = load_dataset_classes(dataset_path)
    if class_names is not None:
        if class_indices is None:
            class_indices = get_indices_for_classes(full_class_names, class_names)
        print(f"  Classes (subset): {len(class_names)} — {class_names}")
    else:
        class_names = full_class_names
        class_indices = full_indices
        print(f"  Classes: {len(class_names)} (all from dataset)")

    print("\n[2/5] Loading model...")
    print("\n[3/5] Running MC Dropout on dataset...")
    aggregate, per_image_results = run_dataset_evaluation(
        dataset_path=dataset_path,
        n_samples=n_samples,
        dropout_rate=dropout_rate,
        limit=limit,
        device=device,
        class_names=class_names,
        class_indices=class_indices,
        verbose=True,
    )

    if save_per_image:
        print("\n[4/5] Saving per-image results...")
        per_img_dir = output_path / "per_image"
        per_img_dir.mkdir(exist_ok=True)
        for r in per_image_results:
            img_stem = Path(r["image"]).stem
            with open(per_img_dir / f"{img_stem}.json", "w") as f:
                json.dump(r, f, indent=2)

    print("\n[5/5] Saving results...")
    results_path = output_path / "dataset_results.json"
    with open(results_path, "w") as f:
        json.dump({
            "aggregate": aggregate,
            "per_image": per_image_results,
        }, f, indent=2)
    print(f"  Saved: {results_path}")

    print("\n" + "=" * 60)
    print("DATASET SUMMARY")
    print("=" * 60)
    print(f"Images processed:    {aggregate['n_images']}")
    print(f"Total time:          {aggregate['total_time_seconds']:.1f}s")
    print(f"Time per image:      {aggregate['time_per_image_seconds']:.2f}s")
    print(f"Mean IoU:            {aggregate['mean_iou']:.4f} ± {aggregate['std_iou']:.4f}")
    print(f"Mean uncertainty:    {aggregate['mean_uncertainty']:.6f} ± {aggregate['std_uncertainty']:.6f}")
    if aggregate.get("per_class_iou"):
        print("\nPer-class IoU:")
        for name, stats in aggregate["per_class_iou"].items():
            m, s = stats["mean"], stats.get("std", 0) or 0
            print(f"  {name:15s}: {m:.4f} ± {s:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run MC Dropout on entire dataset (e.g., Semantic Drone)"
    )
    parser.add_argument("--dataset-path", type=str, required=True, help="Path to dataset root")
    parser.add_argument("--output-dir", type=str, default="outputs", help="Output directory")
    parser.add_argument("--n-samples", type=int, default=30, help="MC Dropout samples per image")
    parser.add_argument("--dropout-rate", type=float, default=0.1, help="Encoder dropout rate")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of images (for testing)")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda/mps/cpu)")
    parser.add_argument("--no-per-image", action="store_true", help="Skip saving per-image JSON files")
    parser.add_argument(
        "--classes",
        type=str,
        nargs="+",
        default=None,
        help="Subset of class names (e.g. paved-area dirt grass). Indices auto-derived from classes.csv, or use --indices.",
    )
    parser.add_argument(
        "--indices",
        type=int,
        nargs="+",
        default=None,
        help="Optional: mask channel indices for --classes. If omitted, indices are looked up from dataset classes.csv.",
    )
    args = parser.parse_args()

    if args.classes is not None and args.indices is not None and len(args.classes) != len(args.indices):
        parser.error("--classes and --indices must have the same length when both are specified")

    main(
        dataset_path=args.dataset_path,
        output_dir=args.output_dir,
        n_samples=args.n_samples,
        dropout_rate=args.dropout_rate,
        limit=args.limit,
        device=args.device,
        save_per_image=not args.no_per_image,
        class_names=args.classes,
        class_indices=args.indices,
    )
