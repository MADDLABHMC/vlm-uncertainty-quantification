"""
Split Conformal Prediction for semantic segmentation.
"""
import numpy as np
from typing import Tuple


class ConformalPredictor:
    """Split conformal prediction for pixel-wise segmentation."""
    
    def __init__(self, alpha: float = 0.1):
        """Initialize conformal predictor.
        
        Args:
            alpha: Miscoverage rate (1-alpha is target coverage)
        """
        self.alpha = alpha
        self.threshold = None
    
    def calibrate(
        self, 
        probs: np.ndarray, 
        ground_truth: np.ndarray,
        cal_mask: np.ndarray
    ) -> float:
        """Calibrate the conformal predictor.
        
        Args:
            probs: Probability array of shape (H, W, num_classes)
            ground_truth: Ground truth labels of shape (H, W)
            cal_mask: Boolean mask of shape (H, W) indicating calibration pixels
            
        Returns:
            Computed threshold value
        """
        # Flatten arrays
        probs_flat = probs.reshape(-1, probs.shape[-1])
        ground_truth_flat = ground_truth.flatten()
        cal_mask_flat = cal_mask.flatten()
        
        # Get calibration pixel indices
        cal_indices = np.where(cal_mask_flat)[0]
        
        # Get probability of true class for calibration pixels
        true_class_probs = probs_flat[cal_indices, ground_truth_flat[cal_indices]]
        
        # Conformity scores: 1 - p(true_class)
        conformity_scores = 1 - true_class_probs
        
        # Compute quantile threshold
        n = len(conformity_scores)
        q_level = np.ceil((n + 1) * (1 - self.alpha)) / n
        self.threshold = np.quantile(conformity_scores, q_level)
        
        return self.threshold
    
    def predict(self, probs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Generate prediction sets using calibrated threshold.
        
        Args:
            probs: Probability array of shape (H, W, num_classes)
            
        Returns:
            Tuple of:
                - prediction_sets: Boolean array of shape (H, W, num_classes)
                - set_sizes: Integer array of shape (H, W)
        """
        if self.threshold is None:
            raise ValueError("Must call calibrate() before predict()")
        
        # Include classes where p(class) >= 1 - threshold
        prediction_sets = probs >= (1 - self.threshold)
        set_sizes = prediction_sets.sum(axis=-1)
        
        return prediction_sets, set_sizes
    
    def evaluate_coverage(
        self,
        prediction_sets: np.ndarray,
        ground_truth: np.ndarray,
        test_mask: np.ndarray
    ) -> float:
        """Evaluate empirical coverage on test set.
        
        Args:
            prediction_sets: Boolean array of shape (H, W, num_classes)
            ground_truth: Ground truth labels of shape (H, W)
            test_mask: Boolean mask of shape (H, W) indicating test pixels
            
        Returns:
            Empirical coverage rate
        """
        # Flatten arrays
        prediction_sets_flat = prediction_sets.reshape(-1, prediction_sets.shape[-1])
        ground_truth_flat = ground_truth.flatten()
        test_mask_flat = test_mask.flatten()
        
        # Get test pixel indices
        test_indices = np.where(test_mask_flat)[0]
        
        # Check if true class is in prediction set for each test pixel
        coverage = prediction_sets_flat[test_indices, ground_truth_flat[test_indices]]
        
        return coverage.mean()