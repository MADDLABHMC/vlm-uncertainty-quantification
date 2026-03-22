"""
Compare Temperature-Scaled MC Dropout across decoder dropout rates with full metric reporting.

This mirrors MC_dropout/run_dropout_comparison.py, but uses the calibrated
temperature pipeline from MC_temperature/run_on_dataset.py.

Usage:
    python run_temperature_comparison.py --dataset-path /path/to/semantic_drone --output-dir outputs_T
    python run_temperature_comparison.py --dataset-path /path/to/semantic_drone --limit 5  # quick test
"""
import argparse
import json
from pathlib import Path

import numpy as np

from run_on_dataset import main as run_dataset_main


def _safe_mean(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def _safe_std(values: list[float]) -> float | None:
    return float(np.std(values)) if values else None


def main(
    dataset_path: str,
    output_dir: str = "outputs_T",
    n_samples: int = 150,
    dropout_rates: list[float] | None = None,
    val_images_min: int = 30,
    max_val_images: int = 50,
    limit: int | None = None,
    device: str | None = None,
    class_names: list[str] | None = None,
    class_indices: list[int] | None = None,
):
    """
    Run Temperature-Scaled MC Dropout for multiple decoder dropout rates,
    aggregate full metrics, and report best/worst classes.
    """
    dropout_rates = dropout_rates or [0.3, 0.7]
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)

    print("=" * 70)
    print("TEMPERATURE-SCALED MC DROPOUT - DROPOUT RATE COMPARISON")
    print("=" * 70)
    print(f"\nDataset: {dataset_path}")
    print(f"Decoder dropout rates: {dropout_rates}")
    print(f"Classes: {class_names if class_names is not None else 'all (from dataset)'}")
    print(f"MC samples: {n_samples}")
    print(f"Device: {device or 'auto'}")
    if limit:
        print(f"Limit: {limit} images (for testing)")
    print()

    # Run evaluation for each decoder dropout rate
    results_by_dropout = {}
    for p in dropout_rates:
        print(f"\n{'='*70}")
        print(f"Running with decoder dropout rate = {p}")
        print("=" * 70)

        run_dir = output_path / f"p{p}"
        run_dir.mkdir(exist_ok=True, parents=True)

        # Run dataset evaluation (writes dataset_results.json)
        run_dataset_main(
            dataset_path=dataset_path,
            output_dir=str(run_dir),
            n_samples=n_samples,
            dropout_rate=p,
            val_images_min=val_images_min,
            max_val_images=max_val_images,
            limit=limit,
            device=device,
            class_names=class_names,
            class_indices=class_indices,
        )

        # Load aggregate metrics from this run
        results_file = run_dir / "dataset_results.json"
        with open(results_file, "r") as f:
            data = json.load(f)
        aggregate = data["aggregate"]
        results_by_dropout[p] = aggregate

    # Aggregate per-class metrics across dropout rates
    all_classes = set()
    for agg in results_by_dropout.values():
        all_classes.update(agg.get("per_class", {}).keys())

    per_class_acc_across_dropouts: dict[str, list[float]] = {c: [] for c in all_classes}
    per_class_entropy_across_dropouts: dict[str, list[float]] = {c: [] for c in all_classes}
    per_class_stdprob_across_dropouts: dict[str, list[float]] = {c: [] for c in all_classes}
    for p, agg in results_by_dropout.items():
        for class_name, stats in agg.get("per_class", {}).items():
            acc = stats.get("pixel_accuracy", {}).get("mean")
            entropy = stats.get("mean_normalized_entropy", {}).get("mean")
            std_prob = stats.get("mean_std_prob", {}).get("mean")
            if acc is not None:
                per_class_acc_across_dropouts[class_name].append(acc)
            if entropy is not None:
                per_class_entropy_across_dropouts[class_name].append(entropy)
            if std_prob is not None:
                per_class_stdprob_across_dropouts[class_name].append(std_prob)

    overall_per_class = {}
    for class_name in sorted(all_classes):
        accs = per_class_acc_across_dropouts[class_name]
        ents = per_class_entropy_across_dropouts[class_name]
        stdps = per_class_stdprob_across_dropouts[class_name]
        if not (accs or ents or stdps):
            continue
        overall_per_class[class_name] = {
            "pixel_accuracy": {
                "mean": float(np.mean(accs)) if accs else None,
                "std": float(np.std(accs)) if accs else None,
                "by_dropout": {
                    str(p): results_by_dropout[p]
                    .get("per_class", {})
                    .get(class_name, {})
                    .get("pixel_accuracy", {})
                    .get("mean")
                    for p in dropout_rates
                },
            },
            "mean_normalized_entropy": {
                "mean": float(np.mean(ents)) if ents else None,
                "std": float(np.std(ents)) if ents else None,
                "by_dropout": {
                    str(p): results_by_dropout[p]
                    .get("per_class", {})
                    .get(class_name, {})
                    .get("mean_normalized_entropy", {})
                    .get("mean")
                    for p in dropout_rates
                },
            },
            "mean_std_prob": {
                "mean": float(np.mean(stdps)) if stdps else None,
                "std": float(np.std(stdps)) if stdps else None,
                "by_dropout": {
                    str(p): results_by_dropout[p]
                    .get("per_class", {})
                    .get(class_name, {})
                    .get("mean_std_prob", {})
                    .get("mean")
                    for p in dropout_rates
                },
            },
        }

    # Find best and worst classes by mean pixel accuracy
    valid = {
        k: v["pixel_accuracy"]["mean"]
        for k, v in overall_per_class.items()
        if v["pixel_accuracy"]["mean"] is not None
    }
    best_class = max(valid, key=valid.get) if valid else None
    worst_class = min(valid, key=valid.get) if valid else None

    # Save results
    first_key = dropout_rates[0]
    summary = {
        "dataset_path": str(dataset_path),
        "dropout_rates": dropout_rates,
        "n_samples": n_samples,
        "n_images": results_by_dropout[first_key]["n_images"],
        "best_class": best_class,
        "best_pixel_accuracy": float(valid[best_class]) if best_class else None,
        "worst_class": worst_class,
        "worst_pixel_accuracy": float(valid[worst_class]) if worst_class else None,
        "per_class_overall": overall_per_class,
        "results_by_dropout": {
            str(p): {
                "pixel_accuracy_mean": agg["pixel_accuracy_mean"],
                "pixel_accuracy_std": agg["pixel_accuracy_std"],
                "mean_uncertainty": agg["mean_uncertainty"],
                "std_uncertainty": agg["std_uncertainty"],
                "mean_std_prob": agg.get("mean_std_prob"),
                "std_std_prob": agg.get("std_std_prob"),
                "time_per_image_seconds": agg["time_per_image_seconds"],
                "total_time_seconds": agg["total_time_seconds"],
                "temperature": agg.get("temperature"),
                "per_class": agg.get("per_class"),
            }
            for p, agg in results_by_dropout.items()
        },
        "overall": {
            "pixel_accuracy_mean": _safe_mean(
                [agg["pixel_accuracy_mean"] for agg in results_by_dropout.values() if agg["pixel_accuracy_mean"] is not None]
            ),
            "pixel_accuracy_std": _safe_std(
                [agg["pixel_accuracy_mean"] for agg in results_by_dropout.values() if agg["pixel_accuracy_mean"] is not None]
            ),
            "mean_uncertainty": _safe_mean(
                [agg["mean_uncertainty"] for agg in results_by_dropout.values() if agg["mean_uncertainty"] is not None]
            ),
            "std_uncertainty": _safe_std(
                [agg["mean_uncertainty"] for agg in results_by_dropout.values() if agg["mean_uncertainty"] is not None]
            ),
            "mean_std_prob": _safe_mean(
                [agg.get("mean_std_prob") for agg in results_by_dropout.values() if agg.get("mean_std_prob") is not None]
            ),
            "std_std_prob": _safe_std(
                [agg.get("mean_std_prob") for agg in results_by_dropout.values() if agg.get("mean_std_prob") is not None]
            ),
            "time_per_image_seconds_mean": _safe_mean(
                [agg["time_per_image_seconds"] for agg in results_by_dropout.values()]
            ),
        },
    }

    out_file = output_path / "temperature_comparison_results.json"
    with open(out_file, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved: {out_file}")

    # Print summary
    print("\n" + "=" * 70)
    print("TEMPERATURE-SCALED MC DROPOUT - CLASS SUMMARY")
    print("=" * 70)
    print(f"\nBest class:  {best_class} (pixel accuracy: {valid.get(best_class, 0):.4f})")
    print(f"Worst class: {worst_class} (pixel accuracy: {valid.get(worst_class, 0):.4f})")

    print("\nPer-class pixel accuracy (across selected dropout rates):")
    print("-" * 50)
    sorted_classes = sorted(
        overall_per_class.items(),
        key=lambda x: x[1]["pixel_accuracy"]["mean"] if x[1]["pixel_accuracy"]["mean"] is not None else -1,
        reverse=True,
    )
    for name, stats in sorted_classes:
        pa = stats["pixel_accuracy"]["mean"]
        pas = stats["pixel_accuracy"]["std"] or 0.0
        en = stats["mean_normalized_entropy"]["mean"]
        sp = stats.get("mean_std_prob", {}).get("mean")
        pa_str = f"{pa:.4f} ± {pas:.4f}" if pa is not None else "N/A"
        en_str = f"{en:.6f}" if en is not None else "N/A"
        sp_str = f"{sp:.6f}" if sp is not None else "N/A"
        print(f"  {name:20s}: acc={pa_str}, entropy={en_str}, std_prob={sp_str}")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare Temperature-Scaled MC Dropout across decoder dropout rates"
    )
    parser.add_argument("--dataset-path", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="outputs_T")
    parser.add_argument("--n-samples", type=int, default=150)
    parser.add_argument(
        "--dropout-rates",
        type=float,
        nargs="+",
        default=[0.3, 0.7],
        help="Decoder dropout rates before the final layer",
    )
    parser.add_argument("--val-images-min", type=int, default=30)
    parser.add_argument("--max-val-images", type=int, default=50)
    parser.add_argument("--limit", type=int, default=None, help="Limit images (for testing)")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--classes",
        type=str,
        nargs="+",
        default=[
            "paved-area",
            "grass",
            "roof",
            "car",
            "tree",
            "ar-marker",
            "obstacle",
        ],
        help="Class names to evaluate (default matches MC_dropout/run_dropout_comparison.py)",
    )
    parser.add_argument(
        "--indices",
        type=int,
        nargs="+",
        default=None,
        help="Optional mask channel indices matching --classes",
    )
    args = parser.parse_args()

    if args.classes is not None and args.indices is not None and len(args.classes) != len(args.indices):
        parser.error("--classes and --indices must have the same length when both are specified")

    main(
        dataset_path=args.dataset_path,
        output_dir=args.output_dir,
        n_samples=args.n_samples,
        dropout_rates=args.dropout_rates,
        val_images_min=args.val_images_min,
        max_val_images=args.max_val_images,
        limit=args.limit,
        device=args.device,
        class_names=args.classes,
        class_indices=args.indices,
    )

