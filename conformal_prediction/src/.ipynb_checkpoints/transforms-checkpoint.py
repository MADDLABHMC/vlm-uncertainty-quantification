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

def gaussian_blur(image, sigma):
    return cv2.GaussianBlur(image, (0, 0), sigma)

def vignette(image, level = 5): 
    height, width = image.shape[:2]

    x_resultant_kernel = cv2.getGaussianKernel(width, width/level)
    y_resultant_kernel = cv2.getGaussianKernel(height, height/level)

    kernel = y_resultant_kernel * x_resultant_kernel.T
    mask = kernel / kernel.max()

    image_vignette = np.copy(image)

    for i in range(3):
        image_vignette[:,:,i] = image_vignette[:,:,i] * mask

    return image_vignette

def stress_occlude(image, num_rois=2, opacity=0.8):
    img = image.copy()
    h, w = img.shape[:2]

    for _ in range(num_rois):
        # Sample patch size from a wide Gaussian, centered at 1/4 of image size
        rw = int(np.clip(np.random.normal(w//4, w//4), w//8, w//2))
        rh = int(np.clip(np.random.normal(h//4, h//4), h//8, h//2))
        x  = np.random.randint(0, w-rw)
        y  = np.random.randint(0, h-rh)
        img[y:y+rh, x:x+rw] = opacity*img[y:y+rh, x:x+rw]

    return img

def smoke(image, num_rois=2, opacity=0.8):
    img = image.copy()
    h, w = img.shape[:2]

    for _ in range(num_rois):
        # Sample patch size from a wide Gaussian, centered at 1/4 of image size
        rw = int(np.clip(np.random.normal(w//4, w//4), w//8, w//2))
        rh = int(np.clip(np.random.normal(h//4, h//4), h//8, h//2))
        x  = np.random.randint(0, w-rw)
        y  = np.random.randint(0, h-rh)
        img[y:y+rh, x:x+rw] = np.clip(
            opacity*img[y:y+rh, x:x+rw] + (1-opacity)*255,
            0, 255
        )

    return img

# def add_gaussian_blur(img, ksize=5, sigma=1.0):
#     return cv2.GaussianBlur(img, (ksize, ksize), sigma)

# def add_salt_pepper_noise(img, amount=0.01):
#     """
#     Adds salt & pepper noise.
    
#     amount: fraction of pixels to corrupt
#     """
#     noisy = img.copy()
#     H, W = img.shape[:2]
    
#     num_pixels = int(amount * H * W)
    
#     # Salt (white pixels)
#     coords = (
#         np.random.randint(0, H, num_pixels),
#         np.random.randint(0, W, num_pixels)
#     )
#     noisy[coords] = 255

#     # Pepper (black pixels)
#     coords = (
#         np.random.randint(0, H, num_pixels),
#         np.random.randint(0, W, num_pixels)
#     )
#     noisy[coords] = 0

#     return noisy

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

    






    

