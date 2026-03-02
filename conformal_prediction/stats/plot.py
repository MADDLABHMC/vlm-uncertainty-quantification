import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import time
import pickle
import cv2
import matplotlib.pyplot as plt

def plot(
    dataset_path: str = "/home/ubuntu/spring2026maddlabsg/datasets/1/semantic_drone",
    indices: list[int] = [1, 3, 9, 17, 19, 21, 22],
    length = 10,
    pickle_path = "pred_sets_overall",
    image_types = ["normal", "gaussian_blur", "salt_and_pepper"]
):
    dataset_path = Path(dataset_path)
    csv_path = dataset_path / "classes.csv"
    classes_df = pd.read_csv(csv_path)
    class_names = list(classes_df["name"].values)
    classes = [class_names[i] for i in indices]
    
    mean_arr = np.zeros((len(classes), len(image_types)))
    std_arr = np.zeros((len(classes), len(image_types)))

    for i, image_type in enumerate(image_types):
        
        with open(f"{pickle_path}/{pickle_path}_{length}_type={image_type}.pkl", "rb") as f:
            data = pickle.load(f)

            for j, class_name in enumerate(classes):
                mean_arr[j, i] = data[class_name]["overall_mean"]
                std_arr[j, i] = data[class_name]["overall_std"]

    # Example shapes
    # M classes, N image types
    M, N = mean_arr.shape
    
    # X locations for groups
    x = np.arange(M)
    
    # Width of each bar
    width = 0.8 / N  # total group width = 0.8
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for i in range(N):
        ax.bar(x + i*width, mean_arr[:, i], width, yerr=std_arr[:, i], label=image_types[i], capsize=4)
    
    ax.set_xticks(x + width*(N-1)/2)
    ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.set_ylabel("Mean prediction set size")
    ax.set_title("Prediction set mean & std per class and image type")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.6)
    
    plt.tight_layout()
    
    # Save figure to disk
    save_path = "prediction_set_means.png"
    plt.savefig(save_path, dpi=300)
    plt.close(fig)  # Close the figure to free memory
    print(f"Figure saved to {save_path}")

if __name__ == "__main__":
    plot(
        dataset_path = "/home/ubuntu/spring2026maddlabsg/datasets/1/semantic_drone",
        indices = [1, 3, 9, 17, 19, 21, 22],
        length = 10,
        pickle_path = "pred_sets_overall",
        image_types = ["normal", "monochrome", "gaussian_blur", "salt_and_pepper"]
    )
    
        
    