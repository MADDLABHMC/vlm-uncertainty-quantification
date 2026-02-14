"""
Example script that replicates the notebook workflow.

This demonstrates how to use the library to replicate the exact 
workflow from the original notebook.
"""
from src.model import CLIPSegModel
from src.conformal import ConformalPredictor
from src.data_utils import load_image_and_mask, split_calibration_test
from src.visualization import visualize_results, print_statistics, show_example_predictions


def main():
    # Configuration matching the notebook
    dataset_path = "/home/ubuntu/spring2026maddlabsg/datasets/1/semantic_drone"
    
    image_path = f"{dataset_path}/images/015.jpg"
    mask_path = f"{dataset_path}/labels/tiff/015.tiff"
    
    texts = ['paved-area', 'dirt', 'grass', 'rocks', 'person', 'dog', 'car', 'bicycle', 'tree']
    indices = [1, 2, 3, 6, 15, 16, 17, 18, 19]
    
    alpha = 0.1  # 90% coverage
    target_size = (352, 352)
    
    print("="*60)
    print("CLIPSEG SPLIT CONFORMAL PREDICTION - EXAMPLE")
    print("="*60)
    
    # Step 1: Load data
    print("\n[Step 1] Loading image and mask...")
    image, ground_truth = load_image_and_mask(
        image_path, 
        mask_path, 
        indices, 
        target_size
    )
    print(f"Ground truth shape: {ground_truth.shape}")
    print(f"Ground truth classes: {set(ground_truth.flatten())}")
    
    # Step 2: Split pixels
    print("\n[Step 2] Splitting calibration/test sets...")
    H, W = target_size
    cal_mask, test_mask = split_calibration_test(H, W, cal_ratio=0.5, seed=42)
    print(f"Calibration pixels: {cal_mask.sum()}")
    print(f"Test pixels: {test_mask.sum()}")
    
    # Step 3: Run model
    print("\n[Step 3] Running CLIPSeg predictions...")
    model = CLIPSegModel()
    probs = model.predict(image, texts)
    probs_np = probs.numpy()
    print(f"Probabilities shape: {probs_np.shape}")
    
    # Step 4: Calibrate
    print("\n[Step 4] Calibrating conformal predictor...")
    cp = ConformalPredictor(alpha=alpha)
    threshold = cp.calibrate(probs_np, ground_truth, cal_mask)
    
    # Step 5: Predict
    print("\n[Step 5] Generating prediction sets...")
    prediction_sets, set_sizes = cp.predict(probs_np)
    
    # Step 6: Evaluate
    print("\n[Step 6] Evaluating coverage...")
    empirical_coverage = cp.evaluate_coverage(prediction_sets, ground_truth, test_mask)
    
    # Print statistics
    print_statistics(set_sizes, empirical_coverage, 1-alpha, threshold)
    
    # Visualize
    print("\n[Step 7] Creating visualizations...")
    fig = visualize_results(
        image, 
        ground_truth, 
        probs_np, 
        prediction_sets, 
        set_sizes,
        save_path="outputs/conformal_prediction_results.png"
    )
    print("Saved: outputs/conformal_prediction_results.png")
    
    # Show examples
    show_example_predictions(
        probs_np, 
        prediction_sets, 
        ground_truth, 
        test_mask, 
        texts,
        num_examples=5
    )
    
    print("\n" + "="*60)
    print("COMPLETE!")
    print("="*60)


if __name__ == "__main__":
    main()