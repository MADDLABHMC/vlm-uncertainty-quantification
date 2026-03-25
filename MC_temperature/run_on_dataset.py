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


def aggregate_from_per_image_results(
    per_image_results: list[dict],
    class_names: list[str],
    dataset_path: Path,
    n_samples: int,
    temperature: float,
    total_time: float,
    dropout_rate: float,
) -> dict:
    """
    Build aggregate dict from per-image result dicts (same schema as run_single_image output).
    """
    accs = [r["pixel_accuracy"] for r in per_image_results]
    uncs = [r["mean_uncertainty"] for r in per_image_results]
    mean_std_probs = [
        r.get("mean_std_prob") for r in per_image_results if r.get("mean_std_prob") is not None
    ]
    std_std_probs = [
        r.get("std_std_prob") for r in per_image_results if r.get("std_std_prob") is not None
    ]

    per_class_accs: dict[str, list[float]] = {name: [] for name in class_names}
    per_class_entropy_means: dict[str, list[float]] = {name: [] for name in class_names}
    per_class_entropy_std_of_pixels: dict[str, list[float]] = {
        name: [] for name in class_names
    }
    per_class_stdprob_means: dict[str, list[float]] = {name: [] for name in class_names}
    for r in per_image_results:
        for name, acc in r.get("per_class_pixel_accuracy", {}).items():
            if acc is not None:
                per_class_accs.setdefault(name, []).append(acc)
        for name, stats in r.get("per_class_uncertainty", {}).items():
            m = stats.get("mean_normalized_entropy")
            if m is not None:
                per_class_entropy_means.setdefault(name, []).append(m)
            sp = stats.get("std_normalized_entropy")
            if sp is not None:
                per_class_entropy_std_of_pixels.setdefault(name, []).append(sp)
        for name, stats in r.get("per_class_channel_std_probs", {}).items():
            m = stats.get("mean")
            if m is not None:
                per_class_stdprob_means.setdefault(name, []).append(m)

    per_class = {}
    for name in class_names:
        accs_c = per_class_accs.get(name, [])
        ents_c = per_class_entropy_means.get(name, [])
        ent_stds_c = per_class_entropy_std_of_pixels.get(name, [])
        stdps_c = per_class_stdprob_means.get(name, [])
        if not (accs_c or ents_c or ent_stds_c or stdps_c):
            continue
        per_class[name] = {
            "pixel_accuracy": {
                "mean": float(np.mean(accs_c)) if accs_c else None,
                "std": float(np.std(accs_c)) if accs_c else None,
            },
            "mean_normalized_entropy": {
                "mean": float(np.mean(ents_c)) if ents_c else None,
                "std": float(np.std(ents_c)) if ents_c else None,
            },
            "std_normalized_entropy": {
                "mean": float(np.mean(ent_stds_c)) if ent_stds_c else None,
                "std": float(np.std(ent_stds_c)) if ent_stds_c else None,
            },
            "mean_std_prob": {
                "mean": float(np.mean(stdps_c)) if stdps_c else None,
                "std": float(np.std(stdps_c)) if stdps_c else None,
            },
        }

    return {
        "method": "Temperature-Scaled MC Dropout",
        "dataset_path": str(dataset_path),
        "n_images": len(per_image_results),
        "n_samples": n_samples,
        "temperature": temperature,
        "dropout_rate": dropout_rate,
        "total_time_seconds": total_time,
        "time_per_image_seconds": total_time / len(per_image_results) if per_image_results else 0,
        "pixel_accuracy_mean": _safe_mean(accs),
        "pixel_accuracy_std": _safe_std(accs),
        "mean_uncertainty": _safe_mean(uncs),
        "std_uncertainty": _safe_std(uncs),
        "mean_std_prob": _safe_mean(mean_std_probs),
        "std_std_prob": _safe_std(mean_std_probs),
        "mean_of_std_prob_std": _safe_mean(std_std_probs),
        "std_of_std_prob_std": _safe_std(std_std_probs),
        "per_class": per_class,
    }


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
    transform_name: str | None = None,
    transform_kwargs: dict | None = None,
    image_index: int = 0,
) -> dict:
    """Run Temperature-Scaled MC Dropout on a single image."""
    image, ground_truth, _, _ = load_image_and_mask(
        str(image_path), str(mask_path),
        class_names=class_names,
        class_indices=class_indices,
    )

    if transform_name:
        from src.transforms import apply_transform

        kwargs = dict(transform_kwargs or {})
        if transform_name in ("occlusions", "smoke") and "seed" not in kwargs:
            kwargs["seed"] = image_index
        image = apply_transform(image, transform_name, **kwargs)

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
    per_class_channel_std_probs = {}
    for i, name in enumerate(class_names):
        class_mask = ground_truth == i
        n_class_pixels = int(class_mask.sum())
        if n_class_pixels <= 0:
            continue

        class_entropy = normalized_entropy[class_mask]
        class_std_prob = std_probs[:, :, i][class_mask]
        per_class_uncertainty[name] = {
            "mean_normalized_entropy": float(class_entropy.mean()),
            "std_normalized_entropy": float(class_entropy.std()),
        }
        per_class_channel_std_probs[name] = {
            "mean": float(class_std_prob.mean()),
            "std": float(class_std_prob.std()),
        }

    return {
        "image": image_path.name,
        "pixel_accuracy": float(pixel_accuracy),
        "mean_uncertainty": float(normalized_entropy.mean()),
        "per_class_pixel_accuracy": per_class_pixel_accuracy,
        "per_class_uncertainty": per_class_uncertainty,
        "per_class_channel_std_probs": per_class_channel_std_probs,
    }


def main(
    dataset_path: str,
    output_dir: str = "outputs",
    n_samples: int = 25,
    dropout_rate: float = 0.3,
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
    model, processor = load_model(dropout_rate=dropout_rate)
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
        # Global mean/std over std_probs for this image
        if "per_class_channel_std_probs" in result and result["per_class_channel_std_probs"]:
            all_std_vals = []
            for stats in result["per_class_channel_std_probs"].values():
                m = stats.get("mean")
                if m is not None:
                    all_std_vals.append(m)
            if all_std_vals:
                result["mean_std_prob"] = float(np.mean(all_std_vals))
                result["std_std_prob"] = float(np.std(all_std_vals))
        per_image_results.append(result)
    total_time = time.time() - start

    print("\n[4/4] Aggregating results...")
    aggregate = aggregate_from_per_image_results(
        per_image_results,
        class_names,
        dataset_path,
        n_samples,
        temperature,
        total_time,
        dropout_rate,
    )

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
    parser.add_argument("--dropout-rate", type=float, default=0.3)
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
        dropout_rate=args.dropout_rate,
        val_images_min=args.val_images_min,
        max_val_images=args.max_val_images,
        limit=args.limit,
        device=args.device,
        class_names=args.classes,
        class_indices=args.indices,
    )
