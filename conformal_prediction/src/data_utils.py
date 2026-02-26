"""
Data loading and preprocessing utilities.
"""
import numpy as np
import tifffile as tiff
from PIL import Image
from transformers.image_utils import load_image
from typing import Tuple


def load_image_and_mask(
    image_path: str,
    mask_path: str,
    class_indices: list[int],
    target_size: Tuple[int, int] = (352, 352)
) -> Tuple[Image.Image, np.ndarray]:
    """Load image and corresponding multi-channel mask.
    
    Args:
        image_path: Path to input image
        mask_path: Path to multi-channel TIFF mask
        class_indices: List of channel indices to extract from mask
        target_size: Target (height, width) to resize mask to
        
    Returns:
        Tuple of (image, ground_truth_mask)
    """
    # Load image
    # print(image_path)
    image = load_image(image_path)
    
    # Load multi-channel mask
    mask = tiff.imread(mask_path)  # Shape: (H, W, num_channels)
    
    # Prepare ground truth by extracting relevant channels + class pixel counts
    ground_truth, class_pixel_counts = prepare_ground_truth(mask, class_indices, target_size)
    
    return image, ground_truth, class_pixel_counts


# def prepare_ground_truth(
#     mask: np.ndarray,
#     class_indices: list[int],
#     target_size: Tuple[int, int]
# ) -> np.ndarray:
#     """Extract relevant channels from mask and resize.
    
#     Args:
#         mask: Multi-channel mask array of shape (H, W, num_channels)
#         class_indices: List of channel indices to extract
#         target_size: Target (height, width) to resize to
        
#     Returns:
#         Ground truth array of shape (target_H, target_W) with class labels
#     """
#     H, W = target_size
#     ground_truth = np.zeros((H, W), dtype=np.int64)
    
#     for i, idx in enumerate(class_indices):
#         channel = mask[:, :, idx]
        
#         # Resize channel to target size
#         channel_resized = np.array(
#             Image.fromarray(channel.astype(np.uint8)).resize((W, H), Image.NEAREST)
#         )
        
#         # Assign class i where channel has max value (255)
#         ground_truth[channel_resized == 255] = i
    
#     return ground_truth
def prepare_ground_truth(
    mask: np.ndarray,
    class_indices: list[int],
    target_size: Tuple[int, int]
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract relevant channels from mask and resize.
    
    Args:
        mask: Multi-channel mask array of shape (H, W, num_channels)
        class_indices: List of channel indices to extract
        target_size: Target (height, width) to resize to
        
    Returns:
        ground_truth: Array of shape (target_H, target_W) with class labels
        class_pixel_counts: Array of shape (num_classes,) with pixel count per class
    """
    H, W = target_size
    num_classes = len(class_indices)
    ground_truth = np.zeros((H, W), dtype=np.int64)
    
    for i, idx in enumerate(class_indices):
        channel = mask[:, :, idx]
        
        # Resize channel to target size
        channel_resized = np.array(
            Image.fromarray(channel.astype(np.uint8)).resize((W, H), Image.NEAREST)
        )
        
        # Assign class i where channel has max value (255)
        ground_truth[channel_resized == 255] = i
    
    # Count pixels per class
    class_pixel_counts = np.bincount(ground_truth.flatten(), minlength=num_classes)
    
    return ground_truth, class_pixel_counts


def split_calibration_test(
    height: int,
    width: int,
    cal_ratio: float = 0.5,
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """Split pixels into calibration and test sets.
    
    Args:
        height: Image height
        width: Image width
        cal_ratio: Fraction of pixels to use for calibration
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (cal_mask, test_mask) boolean arrays of shape (height, width)
    """
    np.random.seed(seed)
    
    total_pixels = height * width
    pixel_indices = np.arange(total_pixels)
    np.random.shuffle(pixel_indices)
    
    # Split pixels
    n_cal = int(total_pixels * cal_ratio)
    cal_indices = pixel_indices[:n_cal]
    test_indices = pixel_indices[n_cal:]
    
    # Create masks
    cal_mask = np.zeros((height, width), dtype=bool)
    test_mask = np.zeros((height, width), dtype=bool)
    
    cal_mask.flat[cal_indices] = True
    test_mask.flat[test_indices] = True
    
    return cal_mask, test_mask