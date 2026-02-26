from src.model import CLIPSegModel
from src.conformal import ConformalPredictor
from src.data_utils import load_image_and_mask, split_calibration_test
from src.visualization import visualize_results, print_statistics, show_example_predictions
from src.get_image_classes import *
from src.conformity_scores import *

import pandas as pd
from pathlib import Path
from tqdm import tqdm
import time
import pickle

def get_image_mask_paths(dataset_path, image_id):
    image_path = dataset_path / "images" / f"{image_id}.jpg"
    mask_path = dataset_path / "labels" / "tiff" / f"{image_id}.tiff"
    return image_path, mask_path

def get_image_pixel_dist(image_path: str, mask_path: str, indices, target_size):
    image, ground_truth, class_pixel_counts = load_image_and_mask(
        image_path, mask_path, indices, target_size
    )
    return image, ground_truth, class_pixel_counts

def main():
    dataset_path = Path("/home/ubuntu/spring2026maddlabsg/datasets/1/semantic_drone")
    csv_path = dataset_path / "classes.csv"
    classes_df = pd.read_csv(csv_path)
    class_names = list(classes_df["name"].values)
    num_classes = len(class_names)
    indices = [i for i in range(len(class_names))]

    alpha = 0.1  # 90% coverage
    target_size = (352, 352)
    H, W = target_size
    cal_mask, test_mask = split_calibration_test(H, W, cal_ratio=0.5, seed=42)

    pixel_dist_dict = {class_name: 0 for class_name in class_names}

    image_dir = dataset_path / "images"
    image_ids = [f.name.split(".")[0] for f in image_dir.iterdir() if f.is_file()]

    model = CLIPSegModel()
    cp = ConformalPredictor(alpha=alpha)

    print("="*60)
    print("CLIPSEG SPLIT CONFORMAL PREDICTION - FULL DATASET")
    print("="*60)

    all_ids = image_ids[:]
    num_ids = len(all_ids)
    empirical_coverage_vals = np.zeros((num_ids, num_classes))

    model_times = []
    cp_times = []
    total_times = []
    
    # Process images with progress bar
    for i, image_id in enumerate(tqdm(all_ids, desc="Processing images", ncols=100)):
        image_path, mask_path = get_image_mask_paths(dataset_path, image_id)
        image, ground_truth, class_pixel_counts = get_image_pixel_dist(str(image_path), str(mask_path), indices, target_size)

        # probs = model.predict(image, class_names)
        # probs_np = probs.numpy()

        # threshold = cp.calibrate(probs_np, ground_truth, cal_mask)
        # prediction_sets, set_sizes = cp.predict(probs_np)

        # empirical_coverage_vals[i] = cp.evaluate_coverage(prediction_sets, ground_truth, test_mask)
        start_model = time.perf_counter()
        probs = model.predict(image, class_names)
        probs_np = probs.numpy()
        end_model = time.perf_counter()
        
        start_cp = time.perf_counter()
        threshold = cp.calibrate(probs_np, ground_truth, cal_mask)
        prediction_sets, set_sizes = cp.predict(probs_np)
        coverage_vals = cp.evaluate_coverage(prediction_sets, ground_truth, test_mask)
        end_cp = time.perf_counter()
        
        model_time = end_model - start_model
        cp_time = end_cp - start_cp
        total_time = model_time + cp_time

        model_times.append(model_time)
        cp_times.append(cp_time)
        total_times.append(total_time)

        empirical_coverage_vals[i] = coverage_vals


    print("="*60)
    print("RESULTS AND STATISTICS")
    print("="*60)
    
    time_dict = {
        "model_times": model_times,
        "cp_times": cp_times,
        "total_times": total_times
    }

    avg_per_class = np.nanmean(empirical_coverage_vals, axis=0)

    output_file = "avg_coverage_per_class.txt"
    
    with open(output_file, "w") as f:
        f.write("Average Per-Class Coverage Across Dataset\n")
        f.write(f"{'Class':15s} {'Coverage (%)':>15s}\n")
        f.write("-" * 35 + "\n")
    
        for i, cov in enumerate(avg_per_class):
            # Handle possible NaNs safely
            if np.isnan(cov):
                line = f"{class_names[i]:15s} {'N/A':>15s}"
            else:
                line = f"{class_names[i]:15s} {cov*100:15.4f}"
    
            print(line)          # print to console
            f.write(line + "\n") # write to file

        line = f"\nMean Model Inference Time: {np.average(model_times)} seconds"
        f.write(line + "\n")
        print(line)
        line = f"Mean Conformal Prediction Steps Time: {np.average(cp_times)} seconds"
        f.write(line + "\n")
        print(line)
        f.close()

    times_file = "times.pkl"
    with open(times_file, "wb") as f:
        pickle.dump(time_dict, f)

if __name__ == "__main__":
    main()        
        







        

