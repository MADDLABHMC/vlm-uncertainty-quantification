import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import time
import pickle
import cv2

def load_img(impath):
    """
    Loads an image from a specified location and returns it in RGB format.
    Input:
    - impath: a string specifying the target image location.
    Returns an RGB image.
    """
    img = cv2.imread(impath)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def make_monochrome(img):
    """
    Convert an RGB/BGR image to grayscale, but keep it 3 channels
    so models like CLIPSeg can still accept it.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # H x W
    mono_rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)  # H x W x 3
    return mono_rgb

def add_gaussian_blur(img, ksize=5, sigma=1.0):
    return cv2.GaussianBlur(img, (ksize, ksize), sigma)

def add_salt_pepper_noise(img, amount=0.01):
    """
    Adds salt & pepper noise.
    
    amount: fraction of pixels to corrupt
    """
    noisy = img.copy()
    H, W = img.shape[:2]
    
    num_pixels = int(amount * H * W)
    
    # Salt (white pixels)
    coords = (
        np.random.randint(0, H, num_pixels),
        np.random.randint(0, W, num_pixels)
    )
    noisy[coords] = 255

    # Pepper (black pixels)
    coords = (
        np.random.randint(0, H, num_pixels),
        np.random.randint(0, W, num_pixels)
    )
    noisy[coords] = 0

    return noisy

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

    






    

