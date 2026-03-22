import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import albumentations as A
import time
import pickle
import cv2

def load_img(impath, scale=0.1):
    """
    Loads an image from a specified location and returns it in RGB format.
    Input:
    - impath: a string specifying the target image location.
    Returns an RGB image.
    """
    img = cv2.imread(impath)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    return img

# Grayscale
def apply_grayscale(image):
    transform = A.Compose([
        A.ToGray(p=1.0)  # Always convert to grayscale
    ])
    return transform(image=image)["image"]

# Gaussian Blur
def apply_gaussian_blur(img, sigma):
    blur = cv2.GaussianBlur(img, (0,0), sigma)
    return blur

# Vertical Blur (simulated via MotionBlur angle 90)
def apply_vertical_blur(image, blur_limit=(31,71)):
    transform = A.Compose([
        A.MotionBlur(blur_limit=blur_limit, p=1.0, angle_range=(90,90))
    ])
    return transform(image=image)["image"]

# Horizontal Blur (simulated via MotionBlur angle 0)
def apply_horizontal_blur(image, blur_limit=(3,7)):
    transform = A.Compose([
        A.MotionBlur(blur_limit=blur_limit, p=1.0, angle_range=(0,0))
    ])
    return transform(image=image)["image"]

# Glass Blur
def apply_glass_blur(image, sigma=0.7, max_delta=4, iterations=1):
    transform = A.Compose([
        A.GlassBlur(sigma=sigma, max_delta=max_delta, iterations=iterations, p=1.0)
    ])
    return transform(image=image)["image"]

# Atmospheric Fog
def apply_atmospheric_fog(image, fog_coef_range=(1.0,1.0), alpha_coef=0.08):
    transform = A.Compose([
        A.RandomFog(fog_coef_range=fog_coef_range, alpha_coef=alpha_coef, p=1.0)
    ])
    return transform(image=image)["image"]

# Rain
def apply_rain(image, k):
    transform = A.Compose([
        A.Spatter(
            mode="rain",
            # mean=(k, k),
            std=(1.0, 1.0),
            cutout_threshold=(k, k),
            # intensity=(0.2, 0.2),
            # color=(200, 200, 255),
            p=1.0
        )
    ])
    return transform(image=image)["image"]



if __name__ == "__main__":
    # Configuration matching the notebook
    dataset_path = "/home/ubuntu/spring2026maddlabsg/datasets/1/semantic_drone"

    file = "004"
    image_path = f"{dataset_path}/images/{file}.jpg"

    # Load original image
    img = load_img(image_path)

    # Make transformed versions
    img_gray = make_monochrome(img)
    img_gauss = add_gaussian_blur(img, ksize=51, sigma=10.0)
    img_saltpepper = add_salt_pepper_noise(img, amount=0.25)

    # Create figure with 1 row, 3 columns
    fig, axes = plt.subplots(2, 2, figsize=(15, 5))

    # Plot original
    axes[0,0].imshow(img)
    axes[0,0].set_title("Original")
    axes[0,0].axis("off")

    # Plot grayscale
    axes[0,1].imshow(img_gray, cmap="gray")
    axes[0,1].set_title("Grayscale")
    axes[0,1].axis("off")

    # Plot Gaussian noise
    axes[1,0].imshow(img_gauss)
    axes[1,0].set_title("Gaussian Blur")
    axes[1,0].axis("off")

    # Plot Salt and Pepper noise
    axes[1,1].imshow(img_saltpepper)
    axes[1,1].set_title("Salt and Pepper Noise")
    axes[1,1].axis("off")

    plt.tight_layout()

    # Save figure to disk
    save_path = "comparison.png"
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close(fig)  # Close figure to free memory

    print(f"Saved comparison figure to {save_path}")

    






    

