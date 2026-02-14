"""
Main pipeline for split conformal prediction with CLIPSeg.
"""
import argparse
from pathlib import Path

from model import CLIPSegModel
from conformal import ConformalPredictor
from data_utils import load_image_and_mask, split_calibration_test
from visualization import visualize_results, print_statistics, show_example_predictions


def main(
    image_path: str,
    mask_path: str,
    class_names: list[str],
    class_indices: list[int],
    alpha: float = 0.1,
    cal_ratio: float = 0.5,
    output_dir: str = "outputs",
    seed: int = 42
):
    """Run complete conformal prediction pipeline.
    
    Args:
        image_path: Path to input image
        mask_path: Path to ground truth mask (TIFF)
        class_names: List of class name strings
        class_indices: List of channel indices in mask corresponding to classes
        alpha: Miscoverage rate (1-alpha is target coverage)
        cal_ratio: Fraction of pixels for calibration
        output_dir: Directory to save outputs
        seed: Random seed
    """
    print("="*60)
    print("SPLIT CONFORMAL PREDICTION FOR CLIPSEG")
    print("="*60)
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    # Load data
    print("\n[1/6] Loading data...")
    image, ground_truth = load_image_and_mask(
        image_path, mask_path, class_indices, target_size=(352, 352)
    )
    H, W = ground_truth.shape
    print(f"  Image size: {image.size}")
    print(f"  Ground truth shape: {ground_truth.shape}")
    print(f"  Classes: {class_names}")
    
    # Split calibration/test
    print("\n[2/6] Splitting calibration/test sets...")
    cal_mask, test_mask = split_calibration_test(H, W, cal_ratio, seed)
    print(f"  Calibration pixels: {cal_mask.sum()}")
    print(f"  Test pixels: {test_mask.sum()}")
    
    # Run model prediction
    print("\n[3/6] Running CLIPSeg predictions...")
    model = CLIPSegModel()
    probs = model.predict(image, class_names)
    probs_np = probs.numpy()
    print(f"  Prediction shape: {probs.shape}")
    
    # Calibrate conformal predictor
    print("\n[4/6] Calibrating conformal predictor...")
    cp = ConformalPredictor(alpha=alpha)
    threshold = cp.calibrate(probs_np, ground_truth, cal_mask)
    print(f"  Threshold: {threshold:.4f}")
    
    # Generate prediction sets
    print("\n[5/6] Generating prediction sets...")
    prediction_sets, set_sizes = cp.predict(probs_np)
    empirical_coverage = cp.evaluate_coverage(prediction_sets, ground_truth, test_mask)
    print(f"  Mean set size: {set_sizes.mean():.2f}")
    print(f"  Empirical coverage: {100*empirical_coverage:.2f}%")
    
    # Visualize and report
    print("\n[6/6] Creating visualizations...")
    fig = visualize_results(
        image, ground_truth, probs_np, prediction_sets, set_sizes,
        save_path=str(output_path / "results.png")
    )
    print(f"  Saved: {output_path / 'results.png'}")
    
    # Print detailed statistics
    print_statistics(set_sizes, empirical_coverage, 1-alpha, threshold)
    
    # Show example predictions
    show_example_predictions(
        probs_np, prediction_sets, ground_truth, test_mask, class_names
    )
    
    print(f"\n{'='*60}")
    print("DONE!")
    print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Split Conformal Prediction for CLIPSeg Segmentation"
    )
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--mask", type=str, required=True, help="Path to ground truth mask (TIFF)")
    parser.add_argument("--classes", type=str, nargs="+", required=True, help="Class names")
    parser.add_argument("--indices", type=int, nargs="+", required=True, help="Channel indices")
    parser.add_argument("--alpha", type=float, default=0.1, help="Miscoverage rate")
    parser.add_argument("--cal-ratio", type=float, default=0.5, help="Calibration ratio")
    parser.add_argument("--output-dir", type=str, default="outputs", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    main(
        image_path=args.image,
        mask_path=args.mask,
        class_names=args.classes,
        class_indices=args.indices,
        alpha=args.alpha,
        cal_ratio=args.cal_ratio,
        output_dir=args.output_dir,
        seed=args.seed
    )