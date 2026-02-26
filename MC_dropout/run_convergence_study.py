"""
Convergence study: run MC Dropout with different sample sizes to find optimal n_samples.

Usage:
    python run_convergence_study.py --image path/to/image.jpg --mask path/to/mask.tiff --output-dir outputs
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.model import load_model
from src.data_utils import load_image_and_mask
from src.inference import mc_dropout_predict
from src.metrics import (
    compute_predictions_and_entropy,
    compute_accuracy,
    compute_uncertainty_stats,
)
from src.visualization import visualize_convergence


def main(
    image_path: str,
    mask_path: str,
    class_names: list[str] | None = None,
    class_indices: list[int] | None = None,
    sample_sizes: list[int] | None = None,
    dropout_rate: float = 0.1,
    output_dir: str = "outputs",
    device: str | None = None,
):
    """Run convergence study across different MC sample counts."""
    import torch

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    sample_sizes = sample_sizes or [5, 10, 20, 30, 50, 100]

    print("=" * 70)
    print("MC DROPOUT CONVERGENCE STUDY")
    print("=" * 70)

    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)

    # Load data
    print("\nLoading data...")
    image, ground_truth, class_names, class_indices = load_image_and_mask(
        image_path, mask_path, class_names, class_indices
    )
    if ground_truth is None:
        raise ValueError("Convergence study requires a mask for evaluation.")

    # Load model
    print("Loading model...")
    model, processor = load_model(dropout_rate=dropout_rate)

    # Run for each sample size
    convergence_results = []
    for n in sample_sizes:
        print(f"\n{'='*50}")
        print(f"Testing n_samples = {n}")
        print(f"{'='*50}")

        start_time = time.time()
        mean_probs, std_probs, _ = mc_dropout_predict(
            model, processor, image, class_names, n_samples=n, device=device
        )
        elapsed = time.time() - start_time

        predictions, _, normalized_entropy = compute_predictions_and_entropy(mean_probs)
        mean_iou, _ = compute_accuracy(
            predictions, ground_truth, class_names, verbose=False
        )

        mean_uncertainty = float(normalized_entropy.mean())

        stats = compute_uncertainty_stats(
            normalized_entropy, predictions, ground_truth, class_names, verbose=False
        )

        result = {
            "n_samples": n,
            "mean_iou": mean_iou,
            "mean_uncertainty": mean_uncertainty,
            "std_uncertainty": float(normalized_entropy.std()),
            "uncertainty_when_correct": stats.get("uncertainty_when_correct"),
            "uncertainty_when_wrong": stats.get("uncertainty_when_wrong"),
            "uncertainty_separation": stats.get("uncertainty_separation"),
            "time_seconds": elapsed,
        }
        convergence_results.append(result)

        print(f"  Mean IoU: {mean_iou:.4f}")
        print(f"  Mean uncertainty: {mean_uncertainty:.6f}")
        print(f"  Time: {elapsed:.2f}s")

    df = pd.DataFrame(convergence_results)

    # Save results
    csv_path = output_path / "convergence_study.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to {csv_path}")

    # Visualize
    fig = visualize_convergence(df, save_path=output_path / "convergence_study.png")

    # Summary
    print("\n" + "=" * 70)
    print("CONVERGENCE STUDY SUMMARY")
    print("=" * 70)
    print(df.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--mask", required=True)
    parser.add_argument("--classes", nargs="+", default=None)
    parser.add_argument("--indices", type=int, nargs="+", default=None)
    parser.add_argument("--sample-sizes", type=int, nargs="+", default=[5, 10, 20, 30, 50, 100])
    parser.add_argument("--dropout-rate", type=float, default=0.1)
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    main(
        image_path=args.image,
        mask_path=args.mask,
        class_names=args.classes,
        class_indices=args.indices,
        sample_sizes=args.sample_sizes,
        dropout_rate=args.dropout_rate,
        output_dir=args.output_dir,
        device=args.device,
    )
