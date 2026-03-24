"""
Run MC Dropout on an entire dataset.

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
    pixel_accuracy, per_class_accuracy = compute_accuracy(
        predictions, ground_truth, class_names, verbose=False
    )

    per_class_pixel_accuracy = {
        name: acc for name, (acc, count) in per_class_accuracy.items()
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

    uncertainty_map_file = None
    if uncertainty_maps_dir is not None:
        uncertainty_maps_dir.mkdir(parents=True, exist_ok=True)
        uncertainty_map_file = uncertainty_maps_dir / f"{image_path.stem}_uncertainty.npz"
        np.savez_compressed(
            uncertainty_map_file,
            std_probs=std_probs.astype(np.float32),
            normalized_entropy=normalized_entropy.astype(np.float32),
            predictions=predictions.astype(np.int16),
        )

    return {
        "image": image_path.name,
        "pixel_accuracy": float(pixel_accuracy),
        "mean_uncertainty": float(normalized_entropy.mean()),
        "std_uncertainty": float(normalized_entropy.std()),
        "mean_std_prob": float(std_probs.mean()),
        "std_std_prob": float(std_probs.std()),
        "per_class_pixel_accuracy": per_class_pixel_accuracy,
        "per_class_uncertainty": per_class_uncertainty,
        "per_class_channel_std_probs": per_class_channel_std_probs,
        "uncertainty_map_file": str(uncertainty_map_file) if uncertainty_map_file else None,
    }


def run_dataset_evaluation(
    dataset_path: str | Path,
    n_samples: int = 30,
    dropout_rate: float = 0.1,
    limit: int | None = None,
    device: str | None = None,
    class_names: list[str] | None = None,
    class_indices: list[int] | None = None,
    uncertainty_maps_dir: Path | None = None,
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
            uncertainty_maps_dir=uncertainty_maps_dir,
        )
        result["inference_time"] = time.time() - start
        per_image_results.append(result)

    total_time = time.time() - total_start

    pixel_accuracies = [r["pixel_accuracy"] for r in per_image_results]
    mean_uncertainties = [r["mean_uncertainty"] for r in per_image_results]
    std_uncertainties = [r["std_uncertainty"] for r in per_image_results]
    mean_std_probs = [r["mean_std_prob"] for r in per_image_results]
    std_std_probs = [r["std_std_prob"] for r in per_image_results]

    per_class_accs: dict[str, list[float]] = {name: [] for name in class_names}
    per_class_entropy_means: dict[str, list[float]] = {name: [] for name in class_names}
    per_class_stdprob_means: dict[str, list[float]] = {name: [] for name in class_names}
    for r in per_image_results:
        for name, acc in r.get("per_class_pixel_accuracy", {}).items():
            if acc is not None:
                per_class_accs.setdefault(name, []).append(acc)
        for name, stats in r.get("per_class_uncertainty", {}).items():
            m = stats.get("mean_normalized_entropy")
            if m is not None:
                per_class_entropy_means.setdefault(name, []).append(m)
        for name, stats in r.get("per_class_channel_std_probs", {}).items():
            m = stats.get("mean")
            if m is not None:
                per_class_stdprob_means.setdefault(name, []).append(m)

    per_class = {}
    for name in class_names:
        accs = per_class_accs.get(name, [])
        ents = per_class_entropy_means.get(name, [])
        stdps = per_class_stdprob_means.get(name, [])
        if not (accs or ents or stdps):
            continue
        per_class[name] = {
            "pixel_accuracy": {
                "mean": float(np.mean(accs)) if accs else None,
                "std": float(np.std(accs)) if accs else None,
            },
            "mean_normalized_entropy": {
                "mean": float(np.mean(ents)) if ents else None,
                "std": float(np.std(ents)) if ents else None,
            },
            "mean_std_prob": {
                "mean": float(np.mean(stdps)) if stdps else None,
                "std": float(np.std(stdps)) if stdps else None,
            },
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
        "pixel_accuracy_mean": _safe_mean(pixel_accuracies),
        "pixel_accuracy_std": _safe_std(pixel_accuracies),
        "mean_uncertainty": _safe_mean(mean_uncertainties),
        "std_uncertainty": _safe_std(mean_uncertainties),
        "mean_of_std_uncertainty": _safe_mean(std_uncertainties),
        "std_of_std_uncertainty": _safe_std(std_uncertainties),
        "mean_std_prob": _safe_mean(mean_std_probs),
        "std_std_prob": _safe_std(mean_std_probs),
        "mean_of_std_prob_std": _safe_mean(std_std_probs),
        "std_of_std_prob_std": _safe_std(std_std_probs),
        "per_class": per_class,
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
    save_uncertainty_maps: bool = True,
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
    uncertainty_maps_dir = output_path / "uncertainty_maps" if save_uncertainty_maps else None
    aggregate, per_image_results = run_dataset_evaluation(
        dataset_path=dataset_path,
        n_samples=n_samples,
        dropout_rate=dropout_rate,
        limit=limit,
        device=device,
        class_names=class_names,
        class_indices=class_indices,
        uncertainty_maps_dir=uncertainty_maps_dir,
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
    pa_mean = aggregate["pixel_accuracy_mean"]
    pa_std = aggregate["pixel_accuracy_std"]
    mu_mean = aggregate["mean_uncertainty"]
    mu_std = aggregate["std_uncertainty"]
    sp_mean = aggregate["mean_std_prob"]
    sp_std = aggregate["std_std_prob"]
    print(f"Pixel accuracy:      {pa_mean:.4f} ± {pa_std:.4f}" if pa_mean is not None else "Pixel accuracy:      N/A")
    print(f"Mean uncertainty:    {mu_mean:.6f} ± {mu_std:.6f}" if mu_mean is not None else "Mean uncertainty:    N/A")
    print(f"Mean std_prob:       {sp_mean:.6f} ± {sp_std:.6f}" if sp_mean is not None else "Mean std_prob:       N/A")
    if aggregate.get("per_class"):
        print("\nPer-class summary:")
        for name, stats in aggregate["per_class"].items():
            acc = stats["pixel_accuracy"]["mean"]
            acc_std = stats["pixel_accuracy"]["std"] or 0.0
            ent = stats["mean_normalized_entropy"]["mean"]
            stp = stats["mean_std_prob"]["mean"]
            acc_str = f"{acc:.4f} ± {acc_std:.4f}" if acc is not None else "N/A"
            ent_str = f"{ent:.6f}" if ent is not None else "N/A"
            stp_str = f"{stp:.6f}" if stp is not None else "N/A"
            print(f"  {name:15s}: acc={acc_str}, entropy={ent_str}, std_prob={stp_str}")
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
        "--no-save-uncertainty-maps",
        action="store_true",
        help="Skip saving per-image uncertainty maps (.npz with std_probs and normalized_entropy)",
    )
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
        save_uncertainty_maps=not args.no_save_uncertainty_maps,
        class_names=args.classes,
        class_indices=args.indices,
    )
