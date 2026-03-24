"""
Run Temperature-Scaled MC Dropout on an entire dataset.

Calibrates T on a validation split, then evaluates on all images.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
from tqdm import tqdm

from src.model import load_model
from src.data_utils import (
    load_image_and_mask,
    iter_dataset_pairs,
    load_dataset_classes,
    get_indices_for_classes,
    create_train_val_split,
)
from src.inference import mc_temperature_predict
from src.calibration import calibrate_temperature
from src.metrics import compute_predictions_and_entropy, compute_accuracy


def _get_device():
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _safe_mean(x):
    return float(np.mean(x)) if x else None


def _safe_std(x):
    return float(np.std(x)) if x else None


def run_single_image(
    model,
    processor,
    image_path: Path,
    mask_path: Path,
    class_names: list[str],
    class_indices: list[int],
    temperature: float,
    n_samples: int,
    device: str,
) -> dict:
    """Run Temperature-Scaled MC Dropout on a single image."""
    image, ground_truth, _, _ = load_image_and_mask(
        str(image_path), str(mask_path),
        class_names=class_names,
        class_indices=class_indices,
    )

    mean_probs, std_probs, normalized_entropy = mc_temperature_predict(
        model, processor, image, class_names,
        temperature=temperature,
        n_samples=n_samples,
        device=device,
        verbose=False,
    )

    predictions, _, _ = compute_predictions_and_entropy(mean_probs)
    pixel_accuracy, per_class_acc = compute_accuracy(
        predictions, ground_truth, class_names, verbose=False
    )

    per_class_pixel_accuracy = {
        name: acc for name, (acc, count) in per_class_acc.items()
        if acc is not None
    }
    per_class_uncertainty = {}
    for i, name in enumerate(class_names):
        class_mask = ground_truth == i
        if class_mask.sum() > 0:
            per_class_uncertainty[name] = {
                "mean_normalized_entropy": float(normalized_entropy[class_mask].mean()),
            }

    return {
        "image": image_path.name,
        "pixel_accuracy": float(pixel_accuracy),
        "mean_uncertainty": float(normalized_entropy.mean()),
        "per_class_pixel_accuracy": per_class_pixel_accuracy,
        "per_class_uncertainty": per_class_uncertainty,
    }


def main(
    dataset_path: str,
    output_dir: str = "outputs",
    n_samples: int = 25,
    val_images_min: int = 30,
    max_val_images: int = 50,
    limit: int | None = None,
    device: str | None = None,
    class_names: list[str] | None = None,
    class_indices: list[int] | None = None,
):
    """Run full dataset evaluation with temperature calibration."""
    device = device or _get_device()
    dataset_path = Path(dataset_path)
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)

    print("=" * 60)
    print("TEMPERATURE-SCALED MC DROPOUT - FULL DATASET")
    print("=" * 60)

    pairs = list(iter_dataset_pairs(dataset_path))
    if limit:
        pairs = pairs[:limit]
    print(f"\n[1/4] Found {len(pairs)} image/mask pairs")

    full_class_names, full_indices = load_dataset_classes(dataset_path)
    if class_names:
        class_indices = class_indices or get_indices_for_classes(full_class_names, class_names)
    else:
        class_names = full_class_names
        class_indices = full_indices
    print(f"  Classes: {len(class_names)}")

    print("\n[2/4] Loading model and calibrating temperature...")
    model, processor = load_model(dropout_rate=0.3)
    model.to(device)

    temperature = calibrate_temperature(
        model, processor, dataset_path,
        n_mc_samples=n_samples,
        val_images_min=val_images_min,
        max_val_images=max_val_images,
        device=device,
        class_names=class_names,
        class_indices=class_indices,
        verbose=True,
    )
    print(f"  Calibrated T = {temperature:.4f}")

    print("\n[3/4] Running on full dataset...")
    per_image_results = []
    start = time.time()
    for img_path, mask_path in tqdm(pairs, desc="  Images"):
        result = run_single_image(
            model, processor,
            img_path, mask_path,
            class_names, class_indices,
            temperature, n_samples, device,
        )
        per_image_results.append(result)
    total_time = time.time() - start

    print("\n[4/4] Aggregating results...")
    accs = [r["pixel_accuracy"] for r in per_image_results]
    uncs = [r["mean_uncertainty"] for r in per_image_results]

    aggregate = {
        "method": "Temperature-Scaled MC Dropout",
        "dataset_path": str(dataset_path),
        "n_images": len(per_image_results),
        "n_samples": n_samples,
        "temperature": temperature,
        "total_time_seconds": total_time,
        "time_per_image_seconds": total_time / len(per_image_results) if per_image_results else 0,
        "pixel_accuracy_mean": _safe_mean(accs),
        "pixel_accuracy_std": _safe_std(accs),
        "mean_uncertainty": _safe_mean(uncs),
        "std_uncertainty": _safe_std(uncs),
    }

    results_path = output_path / "dataset_results.json"
    with open(results_path, "w") as f:
        json.dump({"aggregate": aggregate, "per_image": per_image_results}, f, indent=2)
    print(f"  Saved: {results_path}")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Pixel accuracy: {aggregate['pixel_accuracy_mean']:.4f} ± {aggregate['pixel_accuracy_std'] or 0:.4f}")
    print(f"Mean uncertainty: {aggregate['mean_uncertainty']:.6f}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument("--n-samples", type=int, default=25)
    parser.add_argument("--val-images-min", type=int, default=30)
    parser.add_argument("--max-val-images", type=int, default=50)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--classes", type=str, nargs="+", default=None)
    parser.add_argument("--indices", type=int, nargs="+", default=None)
    args = parser.parse_args()

    main(
        dataset_path=args.dataset_path,
        output_dir=args.output_dir,
        n_samples=args.n_samples,
        val_images_min=args.val_images_min,
        max_val_images=args.max_val_images,
        limit=args.limit,
        device=args.device,
        class_names=args.classes,
        class_indices=args.indices,
    )
