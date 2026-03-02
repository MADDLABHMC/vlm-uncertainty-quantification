from src.model import CLIPSegModel
from src.conformal import ConformalPredictor
from src.data_utils import load_image_and_mask, split_calibration_test
from src.visualization import visualize_results, print_statistics, show_example_predictions
from src.get_image_classes import *
from src.conformity_scores import *

import pandas as pd


def main():
    # Configuration matching the notebook
    dataset_path = "/home/ubuntu/spring2026maddlabsg/datasets/1/semantic_drone"

    file = "004"
    image_path = f"{dataset_path}/images/{file}.jpg"
    mask_path = f"{dataset_path}/labels/tiff/{file}.tiff"
    csv_path = f"{dataset_path}/classes.csv"

    classes_df = pd.read_csv(csv_path)
    texts = list(classes_df["name"].values)
    indices = [i for i in range(len(texts))]
    # texts = ['paved-area', 'dirt', 'grass', 'rocks', 'person', 'dog', 'car', 'bicycle', 'tree']
    # indices = [1, 2, 3, 6, 15, 16, 17, 18, 19]
    
    alpha = 0.1  # 90% coverage
    target_size = (352, 352)
    
    print("="*60)
    print("CLIPSEG SPLIT CONFORMAL PREDICTION - EXAMPLE")
    print("="*60)
    
    # Step 1: Load data
    print("\n[Step 1] Loading image and mask...")
    image, ground_truth, class_pixel_counts = load_image_and_mask(
        image_path, 
        mask_path, 
        indices, 
        target_size
    )
    print(f"Ground truth shape: {ground_truth.shape}")
    print(f"Ground truth classes: {set(ground_truth.flatten())}")
    print("Class distribution:")
    for i, (class_name, count) in enumerate(zip(texts, class_pixel_counts)):
        pct = 100 * count / (352 * 352)
        print(f"  {i}. {class_name:15s}: {count:6d} pixels ({pct:5.2f}%)")
    
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
    empirical_coverage_vals = cp.evaluate_coverage(prediction_sets, ground_truth, test_mask)

    print("\nPer Class Coverage:")
    for i, cov in enumerate(empirical_coverage_vals):
        print(f"Class {texts[i]:<12} : coverage {cov*100:6.2f}%")
    
    # # Print statistics
    # print_statistics(set_sizes, empirical_coverage, 1-alpha, threshold)
    
    # # Visualize
    # print("\n[Step 7] Creating visualizations...")
    # fig = visualize_results(
    #     image, 
    #     ground_truth, 
    #     probs_np, 
    #     prediction_sets, 
    #     set_sizes,
    #     save_path=f"outputs/conformal_prediction_{file}_results.png"
    # )
    # print(f"Saved: outputs/conformal_prediction_{file}_results.png")
    
    # # Show examples
    # show_example_predictions(
    #     probs_np, 
    #     prediction_sets, 
    #     ground_truth, 
    #     test_mask, 
    #     texts,
    #     num_examples=5
    # )
    
    print("\n" + "="*60)
    print("COMPLETE!")
    print("="*60)


if __name__ == "__main__":
    main()