"""
Main entry point for MC Dropout uncertainty quantification with CLIPSeg.

Usage:
    python main.py --image path/to/image.jpg --output-dir outputs
    python main.py --image path/to/image.jpg --mask path/to/mask.tiff --classes paved-area dirt grass --indices 1 2 3
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

from src.model import load_model
from src.data_utils import load_image_and_mask, DEFAULT_CLASSES
from src.inference import mc_dropout_predict
from src.metrics import (
    compute_predictions_and_entropy,
    compute_accuracy,
    compute_uncertainty_stats,
)
from src.visualization import visualize_results, print_example_predictions


def main(
    image_path: str,
    mask_path: str | None = None,
    class_names: list[str] | None = None,
    class_indices: list[int] | None = None,
    n_samples: int = 30,
    dropout_rate: float = 0.1,
    output_dir: str = "outputs",
    device: str | None = None,
    model_name: str = "CIDAS/clipseg-rd64-refined",
):
    """Run MC Dropout pipeline for CLIPSeg uncertainty quantification."""
    import torch

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print("MC DROPOUT FOR CLIPSEG UNCERTAINTY QUANTIFICATION")
    print("=" * 60)

    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)

    # Load data
    print("\n[1/6] Loading data...")
    image, ground_truth, class_names, class_indices = load_image_and_mask(
        image_path,
        mask_path,
        class_names=class_names,
        class_indices=class_indices,
    )
    print(f"  Image size: {list(image.size)}")
    print(f"  Classes: {class_names}")

    H, W = 352, 352
    if ground_truth is not None:
        print(f"  Ground truth shape: {ground_truth.shape}")
    else:
        ground_truth = -np.ones((H, W), dtype=np.int64)  # Placeholder for no mask
        print("  No mask provided - skipping accuracy evaluation")

    # Load model
    print("\n[2/6] Loading model...")
    model, processor = load_model(model_name=model_name, dropout_rate=dropout_rate)
    print(f"  Model: {model_name}")
    print(f"  Dropout rate: {dropout_rate}")

    # Run MC Dropout
    print("\n[3/6] Running MC Dropout inference...")
    start_time = time.time()
    mean_probs, std_probs, _ = mc_dropout_predict(
        model,
        processor,
        image,
        class_names,
        n_samples=n_samples,
        device=device,
    )
    inference_time = time.time() - start_time
    print(f"  Completed in {inference_time:.2f}s")

    # Compute predictions and metrics
    print("\n[4/6] Computing predictions and metrics...")
    predictions, _, normalized_entropy = compute_predictions_and_entropy(mean_probs)

    mean_iou = 0.0
    stats = {}
    if ground_truth is not None and (ground_truth >= 0).any():
        mean_iou, _ = compute_accuracy(
            predictions, ground_truth, class_names, verbose=True
        )
        stats = compute_uncertainty_stats(
            normalized_entropy, predictions, ground_truth, class_names, verbose=True
        )
    else:
        print("  Skipping IoU evaluation (no ground truth)")

    # Visualize
    print("\n[5/6] Creating visualizations...")
    fig = visualize_results(
        image,
        ground_truth if ground_truth is not None else np.zeros_like(predictions),
        predictions,
        normalized_entropy,
        mean_iou,
        class_names,
        save_path=output_path / "mc_dropout_results.png",
    )

    # Example predictions
    print("\n[6/6] Example predictions...")
    if ground_truth is not None and (ground_truth >= 0).any():
        print_example_predictions(
            mean_probs,
            std_probs,
            predictions,
            ground_truth,
            normalized_entropy,
            class_names,
        )

    # Save results
    results = {
        "method": "MC Dropout (Encoder)",
        "model": model_name,
        "dropout_rate": dropout_rate,
        "n_samples": n_samples,
        "mean_iou": float(mean_iou),
        "inference_time": inference_time,
        "mean_uncertainty": float(normalized_entropy.mean()),
        **{k: v for k, v in stats.items() if v is not None},
    }
    results_path = output_path / "mc_dropout_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    print("\n" + "=" * 60)
    print("DONE!")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="MC Dropout for CLIPSeg Uncertainty Quantification"
    )
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--mask", type=str, default=None, help="Path to ground truth mask (TIFF)")
    parser.add_argument("--classes", type=str, nargs="+", default=None, help="Class names")
    parser.add_argument("--indices", type=int, nargs="+", default=None, help="Mask channel indices")
    parser.add_argument("--n-samples", type=int, default=30, help="Number of MC samples")
    parser.add_argument("--dropout-rate", type=float, default=0.1, help="Encoder dropout rate")
    parser.add_argument("--output-dir", type=str, default="outputs", help="Output directory")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda/cpu)")
    parser.add_argument("--model", type=str, default="CIDAS/clipseg-rd64-refined")

    args = parser.parse_args()

    main(
        image_path=args.image,
        mask_path=args.mask,
        class_names=args.classes,
        class_indices=args.indices,
        n_samples=args.n_samples,
        dropout_rate=args.dropout_rate,
        output_dir=args.output_dir,
        device=args.device,
        model_name=args.model,
    )
