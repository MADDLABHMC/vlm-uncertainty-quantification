"""
Split Conformal Prediction for CLIPSeg Segmentation.
"""
from .model import CLIPSegModel
from .conformal import ConformalPredictor
from .data_utils import load_image_and_mask, split_calibration_test
from .visualization import visualize_results, print_statistics, show_example_predictions

__all__ = [
    "CLIPSegModel",
    "ConformalPredictor", 
    "load_image_and_mask",
    "split_calibration_test",
    "visualize_results",
    "print_statistics",
    "show_example_predictions",
]