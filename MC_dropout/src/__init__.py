"""
MC Dropout for Uncertainty Quantification in CLIPSeg.

This package implements Monte Carlo Dropout for semantic segmentation
uncertainty estimation using CLIPSeg with dropout in the vision encoder.
"""

from .model import CLIPSegWithEncoderDropout, load_model
from .inference import mc_dropout_predict
from .data_utils import load_image_and_mask, prepare_ground_truth

__all__ = [
    "CLIPSegWithEncoderDropout",
    "load_model",
    "mc_dropout_predict",
    "load_image_and_mask",
    "prepare_ground_truth",
]
