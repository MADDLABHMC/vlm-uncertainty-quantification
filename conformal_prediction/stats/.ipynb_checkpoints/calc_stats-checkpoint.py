import pickle
import pandas as pd
import numpy as np
from pathlib import Path
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)

BOLD = "\033[1m"
RESET = "\033[0m"


def stats_coverage_vals(pickle_path, class_names, indices):
    """Print average per-class coverage as a table."""
    print()
    print(f"{BOLD}Average Per-Class Coverage:{RESET}")    

    with open(pickle_path, "rb") as f:
        empirical_coverage_vals = pickle.load(f)

    avg_per_class = np.nanmean(empirical_coverage_vals, axis=0)

    # Header
    print()
    print(f"{'Class':15s} {'Coverage (%)':>15s}")
    print("-" * 32)

    for i, cov in enumerate(avg_per_class):
        class_name = class_names[indices[i]]
        if np.isnan(cov):
            print(f"{class_name:15s} {'N/A':>15s}")
        else:
            print(f"{class_name:15s} {cov*100:15.4f}")


def stats_pixel_counts(pickle_path, class_names):
    """Print total pixel counts and percentages per class."""
    print()
    print(f"{BOLD}Pixel Count Per Class:{RESET}")

    with open(pickle_path, "rb") as f:
        pixel_counts_dict = pickle.load(f)

    total_pixel_counts = np.zeros((len(class_names),))
    for _, counts in pixel_counts_dict.items():
        total_pixel_counts += counts

    total_num_pixels = np.sum(total_pixel_counts)

    # Header
    print()
    print(f"{'Class':15s} {'Pixels':>15s} {'Percentage (%)':>18s}")
    print("-" * 50)

    for i, count in enumerate(total_pixel_counts):
        percentage = 100 * count / total_num_pixels if total_num_pixels > 0 else np.nan
        print(f"{class_names[i]:15s} {int(count):15,d} {percentage:15.4f}")


def stats_prediction_sets(prediction_pickle_path, set_size_pickle_path, class_names, out_path):
    """Print prediction set mean and standard deviation per class."""
    print()
    print(f"{BOLD}Prediction Set Statistics:{RESET}")   

    with open(prediction_pickle_path, "rb") as f:
        prediction_sets_dict = pickle.load(f)

    with open(set_size_pickle_path, "rb") as f:
        set_sizes_dict = pickle.load(f)

    pred_set_stats_dict = {class_name: {"total_sum": 0, "total_sq_sum": 0, "total_count": 0} for class_name in class_names}

    for image_id, prediction_sets in prediction_sets_dict.items():
        set_sizes = set_sizes_dict[image_id]
        for j, class_name in enumerate(class_names):
            mask = prediction_sets[:, :, j]
            selected = set_sizes[mask]

            pred_set_stats_dict[class_name]["total_sum"] += selected.sum()
            pred_set_stats_dict[class_name]["total_sq_sum"] += (selected**2).sum()
            pred_set_stats_dict[class_name]["total_count"] += selected.size

    overall_pred_results_dict = {}
    
    # Header
    print()
    print(f"{'Class':15s} {'Mean':>14s} {'Std':>14s}")
    print("-" * 45)

    for class_name, stats in pred_set_stats_dict.items():
        total_sum = stats["total_sum"]
        total_sq_sum = stats["total_sq_sum"]
        total_count = stats["total_count"]

        if total_count == 0:
            overall_mean = np.nan
            overall_std = np.nan
        else:
            overall_mean = total_sum / total_count
            overall_var = (total_sq_sum / total_count) - overall_mean ** 2
            overall_std = np.sqrt(overall_var)

        print(f"{class_name:15s} {overall_mean:14.4f} {overall_std:14.4f}")
        overall_pred_results_dict[class_name] = {
            "overall_mean": overall_mean,
            "overall_std": overall_std
        }

    with open(out_path, "wb") as f:
        pickle.dump(overall_pred_results_dict, f)

def calculate_stats(
    dataset_path: str = "/home/ubuntu/spring2026maddlabsg/datasets/1/semantic_drone",
    indices: list[int] = [1, 3, 9, 17, 19, 21, 22],
    length: int = 10,
    image_type: str = "salt_and_pepper",
    coverage_dir: str = "coverage_vals",
    pixel_counts_dir: str = "pixel_counts",
    prediction_sets_dir: str = "prediction_sets",
    set_sizes_dir: str = "set_sizes"
):
    """
    Organize and print all statistics (coverage, pixel counts, prediction set sizes)
    """
    print("="*60)
    print("RESULTS AND STATISTICS")
    print("="*60)

    dataset_path = Path(dataset_path)
    csv_path = dataset_path / "classes.csv"
    classes_df = pd.read_csv(csv_path)
    class_names = list(classes_df["name"].values)
    classes = [class_names[i] for i in indices]

    # Paths to pickled stats
    coverage_vals_path = Path(coverage_dir) / f"coverage_vals_{length}_type={image_type}.pkl"
    pixel_counts_path = Path(pixel_counts_dir) / f"pixel_counts_{length}_type={image_type}.pkl"
    prediction_sets_path = Path(prediction_sets_dir) / f"prediction_sets_{length}_type={image_type}.pkl"
    set_sizes_path = Path(set_sizes_dir) / f"set_sizes_{length}_type={image_type}.pkl"

    out_path = f"pred_sets_overall/pred_sets_overall_{length}_type={image_type}.pkl"
    
    # Compute / display statistics
    stats_coverage_vals(coverage_vals_path, class_names, indices)
    stats_pixel_counts(pixel_counts_path, classes)
    stats_prediction_sets(prediction_sets_path, set_sizes_path, classes, out_path)


if __name__ == "__main__":
    
    calculate_stats(
        dataset_path = "/home/ubuntu/spring2026maddlabsg/datasets/1/semantic_drone",
        indices = [1, 3, 9, 17, 19, 21, 22],
        length = 10,
        image_type  = "normal",
        coverage_dir = "coverage_vals",
        pixel_counts_dir = "pixel_counts",
        prediction_sets_dir = "prediction_sets",
        set_sizes_dir = "set_sizes"
    )
    calculate_stats(
        dataset_path = "/home/ubuntu/spring2026maddlabsg/datasets/1/semantic_drone",
        indices = [1, 3, 9, 17, 19, 21, 22],
        length = 10,
        image_type  = "monochrome",
        coverage_dir = "coverage_vals",
        pixel_counts_dir = "pixel_counts",
        prediction_sets_dir = "prediction_sets",
        set_sizes_dir = "set_sizes"
    )
    calculate_stats(
        dataset_path = "/home/ubuntu/spring2026maddlabsg/datasets/1/semantic_drone",
        indices = [1, 3, 9, 17, 19, 21, 22],
        length = 10,
        image_type  = "gaussian_blur",
        coverage_dir = "coverage_vals",
        pixel_counts_dir = "pixel_counts",
        prediction_sets_dir = "prediction_sets",
        set_sizes_dir = "set_sizes"
    )
    calculate_stats(
        dataset_path = "/home/ubuntu/spring2026maddlabsg/datasets/1/semantic_drone",
        indices = [1, 3, 9, 17, 19, 21, 22],
        length = 10,
        image_type  = "salt_and_pepper",
        coverage_dir = "coverage_vals",
        pixel_counts_dir = "pixel_counts",
        prediction_sets_dir = "prediction_sets",
        set_sizes_dir = "set_sizes"
    )