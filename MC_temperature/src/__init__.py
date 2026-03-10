"""
Temperature-scaled MC Dropout for CLIPSeg uncertainty quantification.
"""

from .model import CLIPSegWithDecoderDropout, load_model
from .inference import mc_temperature_predict
from .calibration import calibrate_temperature

__all__ = [
    "CLIPSegWithDecoderDropout",
    "load_model",
    "mc_temperature_predict",
    "calibrate_temperature",
]
