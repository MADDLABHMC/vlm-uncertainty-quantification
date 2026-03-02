from src.model import CLIPSegModel
from src.conformal import ConformalPredictor
from src.data_utils import load_image_and_mask, split_calibration_test
from src.visualization import visualize_results, print_statistics, show_example_predictions
from src.get_image_classes import *
from src.conformity_scores import *
from src.transforms import *

import pandas as pd
from pathlib import Path
from tqdm import tqdm
import time
import pickle
import cv2

def get_image_mask_paths(dataset_path, image_id):
    image_path = dataset_path / "images" / f"{image_id}.jpg"
    mask_path = dataset_path / "labels" / "tiff" / f"{image_id}.tiff"
    return image_path, mask_path

def get_image_pixel_dist(image_path: str, mask_path: str, indices, target_size):
    image, ground_truth, class_pixel_counts = load_image_and_mask(
        image_path, mask_path, indices, target_size
    )
    return image, ground_truth, class_pixel_counts

def run(
    dataset_path: str,
    indices: list[int],
    target_size=(352, 352),
    alpha=0.1,
    num_images=None,           # None = all images
    stats_dir="stats",
    image_type="normal"
):
    """
    Run CLIPSeg Conformal Prediction on the dataset.
    """

    dataset_path = Path(dataset_path)
    csv_path = dataset_path / "classes.csv"
    classes_df = pd.read_csv(csv_path)
    class_names = list(classes_df["name"].values)

    H, W = target_size
    cal_mask, test_mask = split_calibration_test(H, W, cal_ratio=0.5, seed=42)

    pixel_dist_dict = {class_name: 0 for class_name in class_names}

    image_dir = dataset_path / "images"
    image_ids = [f.stem for f in image_dir.iterdir() if f.is_file()]
    if num_images:
        all_ids = image_ids[:num_images]
    else:
        all_ids = image_ids
    num_ids = len(all_ids)

    # Picked classes
    classes = [class_names[i] for i in indices]
    num_classes = len(classes)

    # Initialize model and conformal predictor
    model = CLIPSegModel()
    cp = ConformalPredictor(alpha=alpha)

    # Initialize storage
    model_times, cp_times, total_times = [], [], []
    pixel_counts, thresholds = {}, {}
    empirical_coverage_vals = np.zeros((num_ids, num_classes))
    prediction_set_series, set_size_series = {}, {}

    print("="*60)
    print("CLIPSEG SPLIT CONFORMAL PREDICTION - FULL DATASET")
    print("="*60)

    for i, image_id in enumerate(tqdm(all_ids, desc="Processing images", ncols=100)):
        image_path, mask_path = get_image_mask_paths(dataset_path, image_id)
        _, ground_truth, class_pixel_counts = get_image_pixel_dist(
            str(image_path), str(mask_path), indices, target_size
        )
        image = load_img(image_path)
        if image_type == "monochrome":
            image = make_monochrome(image)
        elif image_type == "gaussian_blur":
            image = add_gaussian_blur(image, ksize=51, sigma=10.0)
        elif image_type == "salt_and_pepper":
            image = add_salt_pepper_noise(image, amount=0.25)
        elif image_type != "normal":
            raise ValueError(f"Unrecognized image_type: {image_type}. "
                             f"Expected one of ['normal', 'monochrome', 'gaussian_blur', 'salt_and_pepper']")

        # Model prediction
        start_model = time.perf_counter()
        probs = model.predict(image, classes)
        probs_np = probs.numpy()
        end_model = time.perf_counter()

        # Conformal prediction
        start_cp = time.perf_counter()
        threshold = cp.calibrate(probs_np, ground_truth, cal_mask)
        prediction_sets, set_sizes = cp.predict(probs_np)
        coverage_vals = cp.evaluate_coverage(prediction_sets, ground_truth, test_mask)
        end_cp = time.perf_counter()

        # Store timings
        model_times.append(end_model - start_model)
        cp_times.append(end_cp - start_cp)
        total_times.append((end_model - start_model) + (end_cp - start_cp))

        # Store results
        pixel_counts[image_id] = class_pixel_counts
        thresholds[image_id] = threshold
        prediction_set_series[image_id] = prediction_sets
        set_size_series[image_id] = set_sizes
        empirical_coverage_vals[i] = coverage_vals

    print("="*60)
    print("STORING RESULTS ...")

    Path(stats_dir + "/times").mkdir(parents=True, exist_ok=True)
    Path(stats_dir + "/pixel_counts").mkdir(parents=True, exist_ok=True)
    Path(stats_dir + "/thresholds").mkdir(parents=True, exist_ok=True)
    Path(stats_dir + "/coverage_vals").mkdir(parents=True, exist_ok=True)
    Path(stats_dir + "/prediction_sets").mkdir(parents=True, exist_ok=True)
    Path(stats_dir + "/set_sizes").mkdir(parents=True, exist_ok=True)

    # Pickle dictionaries
    def save_pickle(obj, path):
        with open(path, "wb") as f:
            pickle.dump(obj, f)

    save_pickle(
        {"model_times": model_times, "cp_times": cp_times, "total_times": total_times},
        f"{stats_dir}/times/times_{num_ids}_type={image_type}.pkl"
    )
    save_pickle(pixel_counts, f"{stats_dir}/pixel_counts/pixel_counts_{num_ids}_type={image_type}.pkl")
    save_pickle(thresholds, f"{stats_dir}/thresholds/thresholds_{num_ids}_type={image_type}.pkl")
    save_pickle(empirical_coverage_vals, f"{stats_dir}/coverage_vals/coverage_vals_{num_ids}_type={image_type}.pkl")
    save_pickle(prediction_set_series, f"{stats_dir}/prediction_sets/prediction_sets_{num_ids}_type={image_type}.pkl")
    save_pickle(set_size_series, f"{stats_dir}/set_sizes/set_sizes_{num_ids}_type={image_type}.pkl")

    print("FINISHED STORING RESULTS")
    print("="*60)

if __name__ == "__main__":
    run(
        dataset_path="/home/ubuntu/spring2026maddlabsg/datasets/1/semantic_drone",
        indices=[1, 3, 9, 17, 19, 21, 22],
        target_size=(352, 352),
        alpha=0.1,
        num_images=10,
        stats_dir="stats",
        image_type="monochrome"
    )


# def main():
#     dataset_path = Path("/home/ubuntu/spring2026maddlabsg/datasets/1/semantic_drone")
#     csv_path = dataset_path / "classes.csv"
#     classes_df = pd.read_csv(csv_path)
#     class_names = list(classes_df["name"].values)
    
#     indices = [1, 3, 9, 17, 19, 21, 22]# [i for i in range(len(class_names))] # you can define

#     # Define coverage and target size
#     alpha = 0.1  # 90% coverage
#     target_size = (352, 352)
    
#     H, W = target_size
#     cal_mask, test_mask = split_calibration_test(H, W, cal_ratio=0.5, seed=42)

#     pixel_dist_dict = {class_name: 0 for class_name in class_names}

#     image_dir = dataset_path / "images"
#     image_ids = [f.name.split(".")[0] for f in image_dir.iterdir() if f.is_file()]

#     model = CLIPSegModel()
#     cp = ConformalPredictor(alpha=alpha)

#     print("="*60)
#     print("CLIPSEG SPLIT CONFORMAL PREDICTION - FULL DATASET")
#     print("="*60)

#     all_ids = image_ids[:10]
#     num_ids = len(all_ids)

#     # define arrays for pickle files
#     classes = [class_names[i] for i in indices]
#     num_classes = len(classes)
#     model_times = []
#     cp_times = []
#     total_times = []
#     pixel_counts = {}
#     thresholds = {}
#     empirical_coverage_vals = np.zeros((num_ids, num_classes))
#     prediction_set_series = {}
#     set_size_series = {}
    
#     # Process images with progress bar
#     for i, image_id in enumerate(tqdm(all_ids, desc="Processing images", ncols=100)):
#         image_path, mask_path = get_image_mask_paths(dataset_path, image_id)
#         image, ground_truth, class_pixel_counts = get_image_pixel_dist(str(image_path), str(mask_path), indices, target_size)

#         start_model = time.perf_counter()
#         probs = model.predict(image, classes)
#         probs_np = probs.numpy()
#         end_model = time.perf_counter()
        
#         start_cp = time.perf_counter()
#         threshold = cp.calibrate(probs_np, ground_truth, cal_mask)
#         prediction_sets, set_sizes = cp.predict(probs_np)
#         coverage_vals = cp.evaluate_coverage(prediction_sets, ground_truth, test_mask)
#         end_cp = time.perf_counter()
        
#         model_time = end_model - start_model
#         cp_time = end_cp - start_cp
#         total_time = model_time + cp_time
        
#         model_times.append(model_time)
#         cp_times.append(cp_time)
#         total_times.append(total_time)

#         # print(image_id, type(image_id))
#         # print(type(class_pixel_counts))
        
#         pixel_counts[image_id] = class_pixel_counts
#         thresholds[image_id] = threshold
#         prediction_set_series[image_id] = prediction_sets
#         set_size_series[image_id] = set_sizes
        
#         empirical_coverage_vals[i] = coverage_vals

#     print("="*60)
#     print("STORING RESULTS")
#     print("...")
    
#     # pickle files
#     stats_dir = "stats"

#     # times pickle
#     time_dict = {
#         "model_times": model_times,
#         "cp_times": cp_times,
#         "total_times": total_times
#     }
#     times_path = f"{stats_dir}/times/times_{len(all_ids)}.pkl"
#     with open(times_path, "wb") as f:
#         pickle.dump(time_dict, f)

#     # pixel counts pickle
#     pixel_counts_path = f"{stats_dir}/pixel_counts/pixel_counts_{len(all_ids)}.pkl"
#     with open(pixel_counts_path, "wb") as f:
#         pickle.dump(pixel_counts, f)

#     # thresholds pickle
#     thresholds_path = f"{stats_dir}/thresholds/thresholds_{len(all_ids)}.pkl"
#     with open(thresholds_path, "wb") as f:
#         pickle.dump(thresholds, f)

#     # coverage vals pickle
#     coverage_vals_path = f"{stats_dir}/coverage_vals/coverage_vals_{len(all_ids)}.pkl"
#     with open(coverage_vals_path, "wb") as f:
#         pickle.dump(empirical_coverage_vals, f)

#     # prediction sets pickle
#     prediction_sets_path = f"{stats_dir}/prediction_sets/prediction_sets_{len(all_ids)}.pkl"
#     with open(prediction_sets_path, "wb") as f:
#         pickle.dump(prediction_set_series, f)

#     # set sizes pickle
#     set_sizes_path = f"{stats_dir}/set_sizes/set_sizes_{len(all_ids)}.pkl"
#     with open(set_sizes_path, "wb") as f:
#         pickle.dump(set_size_series, f)

#     print("FINISHED STORING RESULTS")
#     print("="*60)
    



# if __name__ == "__main__":
#     main()       









    # print("="*60)
    # print("RESULTS AND STATISTICS")
    # print("="*60)
    
    

    # avg_per_class = np.nanmean(empirical_coverage_vals, axis=0)

    

        
    # # Average coverage per class
    # output_file = "avg_coverage_per_class.txt"
    
    # with open(output_file, "w") as f:
    #     f.write("Average Per-Class Coverage Across Dataset\n")
    #     f.write(f"{'Class':15s} {'Coverage (%)':>15s}\n")
    #     f.write("-" * 35 + "\n")
    
    #     for i, cov in enumerate(avg_per_class):
    #         # Handle possible NaNs safely
    #         if np.isnan(cov):
    #             line = f"{class_names[i]:15s} {'N/A':>15s}"
    #         else:
    #             line = f"{class_names[i]:15s} {cov*100:15.4f}"
    
    #         print(line)          # print to console
    #         f.write(line + "\n") # write to file

    #     line = f"\nMean Model Inference Time: {np.average(model_times)} seconds"
    #     f.write(line + "\n")
    #     print(line)
    #     line = f"Mean Conformal Prediction Steps Time: {np.average(cp_times)} seconds"
    #     f.write(line + "\n")
    #     print(line)
    #     f.close()

    # times_file = "times.pkl"
    # with open(times_file, "wb") as f:
    #     pickle.dump(time_dict, f)
        







        

