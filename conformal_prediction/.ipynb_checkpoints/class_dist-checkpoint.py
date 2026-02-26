from src.data_utils import load_image_and_mask
import pandas as pd
from pathlib import Path
from tqdm import tqdm

def get_image_mask_paths(dataset_path, image_id):
    image_path = dataset_path / "images" / f"{image_id}.jpg"
    mask_path = dataset_path / "labels" / "tiff" / f"{image_id}.tiff"
    return image_path, mask_path

def get_image_pixel_dist(image_path: str, mask_path: str, indices, target_size):
    _, ground_truth, class_pixel_counts = load_image_and_mask(
        image_path, mask_path, indices, target_size
    )
    return ground_truth, class_pixel_counts

def main():
    dataset_path = Path("/home/ubuntu/spring2026maddlabsg/datasets/1/semantic_drone")
    csv_path = dataset_path / "classes.csv"
    classes_df = pd.read_csv(csv_path)
    class_names = list(classes_df["name"].values)
    indices = [i for i in range(len(class_names))]
    target_size = (352, 352)

    pixel_dist_dict = {class_name: 0 for class_name in class_names}

    image_dir = dataset_path / "images"
    image_ids = [f.name.split(".")[0] for f in image_dir.iterdir() if f.is_file()]

    # Process images with progress bar
    for image_id in tqdm(image_ids, desc="Processing images", ncols=100):
        image_path, mask_path = get_image_mask_paths(dataset_path, image_id)
        _, class_pixel_counts = get_image_pixel_dist(
            str(image_path), str(mask_path), indices, target_size
        )
        for class_name, count in zip(class_names, class_pixel_counts):
            pixel_dist_dict[class_name] += count

    # Compute total pixels counted (robust)
    total_pixels_counted = sum(pixel_dist_dict.values())
    print(f"Total number of pixels: {total_pixels_counted}")

    output_file = "class_distribution.txt"
    with open(output_file, "w") as f:
        f.write("Class distribution across dataset:\n")
        f.write(f"{'Class':15s} {'Pixels':>10s} {'% of total':>12s}\n")
        f.write("-" * 40 + "\n")

        for class_name, count in pixel_dist_dict.items():
            pct = 100 * count / total_pixels_counted  # robust percentage
            line = f"{class_name:15s}: {count:6d} pixels ({pct:5.2f}%)"
            print(line)          # print to console
            f.write(line + "\n") # write to file

main()