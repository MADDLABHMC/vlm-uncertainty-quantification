"""
Compare MC Dropout across multiple dropout rates on full dataset with all classes.

Runs the model with dropout rates 0.3, 0.5, and 0.7 on all 24 classes across the
entire dataset, then reports the overall best and worst performing classes
(averaged across dropout rates).

Usage:
    python run_dropout_comparison.py --dataset-path /path/to/semantic_drone --output-dir outputs
    python run_dropout_comparison.py --dataset-path /path/to/semantic_drone --limit 5  # quick test
"""
import argparse
import json
from pathlib import Path

import numpy as np

from run_on_dataset import run_dataset_evaluation, _get_device


def main(
    dataset_path: str,
    output_dir: str = "outputs",
    n_samples: int = 30,
    dropout_rates: list[float] | None = None,
    limit: int | None = None,
    device: str | None = None,
):
    """Run MC Dropout for multiple dropout rates, aggregate per-class IoU, report best/worst."""
    dropout_rates = dropout_rates or [0.3, 0.5, 0.7]
    device = device or _get_device()
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)

    print("=" * 70)
    print("MC DROPOUT - DROPOUT RATE COMPARISON (Best/Worst Classes)")
    print("=" * 70)
    print(f"\nDataset: {dataset_path}")
    print(f"Dropout rates: {dropout_rates}")
    print(f"Classes: all 24 (from dataset)")
    print(f"Device: {device}")
    if limit:
        print(f"Limit: {limit} images (for testing)")
    print()

    # Run evaluation for each dropout rate
    results_by_dropout = {}
    for p in dropout_rates:
        print(f"\n{'='*70}")
        print(f"Running with dropout rate = {p}")
        print("=" * 70)
        aggregate, _ = run_dataset_evaluation(
            dataset_path=dataset_path,
            n_samples=n_samples,
            dropout_rate=p,
            limit=limit,
            device=device,
            class_names=None,  # use all 24 classes
            class_indices=None,
            verbose=True,
        )
        results_by_dropout[p] = aggregate

    # Aggregate per-class IoU across dropout rates
    # For each class: average the mean IoU from each dropout run
    all_classes = set()
    for agg in results_by_dropout.values():
        all_classes.update(agg.get("per_class_iou", {}).keys())

    per_class_across_dropouts: dict[str, list[float]] = {c: [] for c in all_classes}
    for p, agg in results_by_dropout.items():
        for class_name, stats in agg.get("per_class_iou", {}).items():
            m = stats.get("mean")
            if m is not None:
                per_class_across_dropouts[class_name].append(m)

    overall_per_class = {}
    for class_name in sorted(per_class_across_dropouts.keys()):
        ious = per_class_across_dropouts[class_name]
        if ious:
            overall_per_class[class_name] = {
                "mean": float(np.mean(ious)),
                "std": float(np.std(ious)) if len(ious) > 1 else 0.0,
                "by_dropout": {
                    str(p): results_by_dropout[p]["per_class_iou"].get(class_name, {}).get("mean")
                    for p in dropout_rates
                    if class_name in results_by_dropout[p].get("per_class_iou", {})
                },
            }

    # Find best and worst
    valid = {k: v["mean"] for k, v in overall_per_class.items() if v["mean"] is not None}
    best_class = max(valid, key=valid.get) if valid else None
    worst_class = min(valid, key=valid.get) if valid else None

    # Save results
    summary = {
        "dataset_path": str(dataset_path),
        "dropout_rates": dropout_rates,
        "n_samples": n_samples,
        "n_images": results_by_dropout[dropout_rates[0]]["n_images"],
        "best_class": best_class,
        "best_mean_iou": float(valid[best_class]) if best_class else None,
        "worst_class": worst_class,
        "worst_mean_iou": float(valid[worst_class]) if worst_class else None,
        "per_class_overall": overall_per_class,
        "results_by_dropout": {
            str(p): {
                "mean_iou": agg["mean_iou"],
                "per_class_iou": agg.get("per_class_iou"),
            }
            for p, agg in results_by_dropout.items()
        },
    }

    out_file = output_path / "dropout_comparison_results.json"
    with open(out_file, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved: {out_file}")

    # Print summary
    print("\n" + "=" * 70)
    print("OVERALL BEST AND WORST CLASSES (averaged across dropout rates)")
    print("=" * 70)
    print(f"\nBest class:  {best_class} (mean IoU: {valid.get(best_class, 0):.4f})")
    print(f"Worst class: {worst_class} (mean IoU: {valid.get(worst_class, 0):.4f})")

    print("\nPer-class mean IoU (across dropout rates 0.3, 0.5, 0.7):")
    print("-" * 50)
    sorted_classes = sorted(overall_per_class.items(), key=lambda x: x[1]["mean"], reverse=True)
    for name, stats in sorted_classes:
        print(f"  {name:20s}: {stats['mean']:.4f} ± {stats['std']:.4f}")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare MC Dropout across dropout rates, find best/worst classes"
    )
    parser.add_argument("--dataset-path", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument("--n-samples", type=int, default=30)
    parser.add_argument(
        "--dropout-rates",
        type=float,
        nargs="+",
        default=[0.3, 0.5, 0.7],
        help="Dropout rates to compare",
    )
    parser.add_argument("--limit", type=int, default=None, help="Limit images (for testing)")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    main(
        dataset_path=args.dataset_path,
        output_dir=args.output_dir,
        n_samples=args.n_samples,
        dropout_rates=args.dropout_rates,
        limit=args.limit,
        device=args.device,
    )
