"""
Main entry point for Temperature-Scaled MC Dropout with CLIPSeg.

Usage:
    # Single image (uses default T=1.0 if no calibration)
    python main.py --image path/to/image.jpg --output-dir outputs

    # Calibrate T on dataset, then run on image
    python main.py --image image.jpg --mask mask.tiff --calibrate-on /path/to/dataset --output-dir outputs

    # With custom temperature and uncertainty rejection
    python main.py --image image.jpg --mask mask.tiff --temperature 1.5 --H-max 0.8
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

from src.model import load_model
from src.data_utils import load_image_and_mask, DEFAULT_CLASSES
from src.inference import mc_temperature_predict, predict_with_rejection
from src.calibration import calibrate_temperature
from src.metrics import compute_predictions_and_entropy, compute_accuracy, compute_uncertainty_stats
from src.visualization import visualize_results


def main(
    image_path: str,
    mask_path: str | None = None,
    class_names: list[str] | None = None,
    class_indices: list[int] | None = None,
    temperature: float = 1.0,
    calibrate_on: str | None = None,
    n_samples: int = 25,
    H_max: float | None = None,
    dropout_rate: float = 0.3,
    output_dir: str = "outputs",
    device: str | None = None,
    model_name: str = "CIDAS/clipseg-rd64-refined",
):
    """Run Temperature-Scaled MC Dropout pipeline for CLIPSeg."""
    import torch

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print("TEMPERATURE-SCALED MC DROPOUT FOR CLIPSEG")
    print("=" * 60)

    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)

    # Load data
    print("\n[1/7] Loading data...")
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
        ground_truth = -np.ones((H, W), dtype=np.int64)
        print("  No mask provided - skipping accuracy evaluation")

    # Load model
    print("\n[2/7] Loading model...")
    model, processor = load_model(model_name=model_name, dropout_rate=dropout_rate)
    print(f"  Model: {model_name}")
    print(f"  Dropout rate: {dropout_rate}")

    # Calibrate temperature if requested
    if calibrate_on:
        print("\n[3/7] Calibrating temperature on validation set...")
        temperature = calibrate_temperature(
            model, processor, calibrate_on,
            n_mc_samples=n_samples,
            device=device,
            class_names=class_names,
            class_indices=class_indices,
            verbose=True,
        )
        print(f"  Calibrated T = {temperature:.4f}")
    else:
        print(f"\n[3/7] Using temperature T = {temperature}")

    # Run MC Temperature inference
    print("\n[4/7] Running Temperature-Scaled MC Dropout inference...")
    start_time = time.time()
    mean_probs, std_probs, normalized_entropy = mc_temperature_predict(
        model, processor, image, class_names,
        temperature=temperature,
        n_samples=n_samples,
        device=device,
    )
    inference_time = time.time() - start_time
    print(f"  Completed in {inference_time:.2f}s (N={n_samples})")

    # Optional rejection by uncertainty threshold
    rejected = np.zeros_like(mean_probs[:, :, 0], dtype=bool)  # placeholder
    if H_max is not None:
        predictions, rejected = predict_with_rejection(
            mean_probs, normalized_entropy, H_max=H_max
        )
        print(f"  Rejected {rejected.sum()} pixels (H > {H_max})")
    else:
        predictions = mean_probs.argmax(axis=-1)

    # Compute metrics
    print("\n[5/7] Computing metrics...")
    pixel_accuracy = 0.0
    stats = {}
    if ground_truth is not None and (ground_truth >= 0).any():
        gt_eval = ground_truth.copy()
        if H_max is not None and rejected.any():
            gt_eval[rejected] = -1  # exclude rejected from accuracy
        pixel_accuracy, _ = compute_accuracy(
            predictions, gt_eval, class_names, verbose=True
        )
        valid = gt_eval >= 0
        if valid.any():
            stats = compute_uncertainty_stats(
                normalized_entropy[valid],
                predictions[valid],
                gt_eval[valid],
            )

    # Visualize
    print("\n[6/7] Creating visualizations...")
    gt_display = ground_truth if ground_truth is not None else np.zeros_like(predictions)
    if (ground_truth < 0).all():
        gt_display = np.zeros_like(predictions)
    fig = visualize_results(
        image, gt_display, predictions, normalized_entropy,
        pixel_accuracy, class_names,
        save_path=output_path / "mc_temperature_results.png",
    )

    # Save results
    print("\n[7/7] Saving results...")
    results = {
        "method": "Temperature-Scaled MC Dropout",
        "model": model_name,
        "temperature": temperature,
        "n_samples": n_samples,
        "H_max": H_max,
        "dropout_rate": dropout_rate,
        "pixel_accuracy": float(pixel_accuracy),
        "inference_time": inference_time,
        "mean_uncertainty": float(normalized_entropy.mean()),
        **stats,
    }
    results_path = output_path / "mc_temperature_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved: {results_path}")

    print("\n" + "=" * 60)
    print("DONE!")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Temperature-Scaled MC Dropout for CLIPSeg"
    )
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--mask", type=str, default=None, help="Path to ground truth mask (TIFF)")
    parser.add_argument("--classes", type=str, nargs="+", default=None)
    parser.add_argument("--indices", type=int, nargs="+", default=None)
    parser.add_argument("--temperature", type=float, default=1.0, help="Temperature T (use --calibrate-on to optimize)")
    parser.add_argument("--calibrate-on", type=str, default=None, help="Dataset path for T calibration")
    parser.add_argument("--n-samples", type=int, default=25, help="MC samples")
    parser.add_argument("--H-max", type=float, default=None, help="Uncertainty threshold for rejection")
    parser.add_argument("--dropout-rate", type=float, default=0.3)
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--model", type=str, default="CIDAS/clipseg-rd64-refined")

    args = parser.parse_args()
    main(
        image_path=args.image,
        mask_path=args.mask,
        class_names=args.classes,
        class_indices=args.indices,
        temperature=args.temperature,
        calibrate_on=args.calibrate_on,
        n_samples=args.n_samples,
        H_max=args.H_max,
        dropout_rate=args.dropout_rate,
        output_dir=args.output_dir,
        device=args.device,
        model_name=args.model,
    )
