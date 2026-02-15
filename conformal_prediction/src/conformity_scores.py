import numpy as np
import torch
from typing import Tuple, Dict, List

# ==========================================
# CONFORMITY SCORE FUNCTIONS
# ==========================================
def absolute_conformity_score(probs: np.ndarray, ground_truth: np.ndarray) -> np.ndarray:
    """
    Absolute conformity score: 1 - p(true_class)
    
    This is the simplest score - just the "error" in probability space.
    Symmetric score: produces constant-width prediction sets across all pixels.
    
    Args:
        probs: (H, W, num_classes) array of probabilities
        ground_truth: (H, W) array of true class indices
    
    Returns:
        conformity_scores: (H*W,) array of scores
    """
    H, W, num_classes = probs.shape

    # Get probability of true class for each pixel
    true_class_probs = probs[
        np.arange(H)[:, None],
        np.arange(W)[None, :],
        ground_truth
    ] # Shape: (H, W)

    # Score: 1 - p(true_class)
    # Lower score = better (model was confident in correct answer)
    conformity_Scores = 1 - true_class_probs

    return conformity_score.flatten()

def gamma_conformity_score(probs: np.ndarray, ground_truth: np.ndarray, epsilon: float = 1e-8) -> np.ndarray:
    """
    Gamma conformity score: (1 - p(true_class)) / p(true_class)
    
    Adaptive score: normalizes by the prediction confidence.
    Asymmetric: regions with low confidence get larger prediction sets.
    
    Inversely proportional to confidence: when model is very confident (high p),
    score is low; when uncertain (low p), score is high.
    
    Args:
        probs: (H, W, num_classes) array of probabilities
        ground_truth: (H, W) array of true class indices
        epsilon: small constant to avoid division by zero
    
    Returns:
        conformity_scores: (H*W,) array of scores
    """
    H, W, num_classes = probs.shape

    # Get probability of true class for each pixel
    true_class_probs = probs[
        np.arange(H)[:, None],
        np.arange(W)[None, :],
        ground_truth
    ] # Shape: (H, W)

    # Score: (1 - p) / (p + epsilon)
    # Larger when p is small (uncertain predictions)
    conformity_Scores = (1 - true_class_probs) / (true_class_probs + epsilon)
    
    return conformity_scores.flatten()


def residual_normalized_conformity_score(
    probs: np.ndarray,
    ground_truth: np.ndarray,
    learned_residuals: np.ndarray,
    epsilon: float = 1e-8
) -> np.ndarray:
    """
    Residual normalized score: (1 - p(true_class)) / sigma_hat(x)
    
    Most sophisticated adaptive score: normalizes by learned difficulty estimate.
    Requires training an additional model to predict |residuals| from features.
    
    Symmetric: but adaptive based on learned uncertainty patterns.
    
    Args:
        probs: (H, W, num_classes) array of probabilities
        ground_truth: (H, W) array of true class indices
        learned_residuals: (H, W) array of predicted residual magnitudes
        epsilon: small constant to avoid division by zero
    
    Returns:
        conformity_scores: (H*W,) array of scores
    """

    H, W, num_classes = probs.shape

    # Get probability of true class for each pixel
    true_class_probs = probs[
        np.arange(H)[:, None],
        np.arange(W)[None, :],
        ground_truth
    ] # Shape: (H, W)

    # Score: (1 - p) / sigma_hat
    # Normalized by learned difficulty
    conformity_scores = (1 - true_class_probs) / (learned_residuals + epsilon)

    return conformity_scores.flatten()

# ==========================================
# CALIBRATION FUNCTIONS
# ==========================================

def calibrate_conformal_predictor(
    conformity_scores: np.ndarray,
    alpha: float
) -> float:
    """
    Compute the conformal prediction threshold for given coverage level.
    
    Args:
        conformity_scores: Array of conformity scores from calibration set
        alpha: Miscoverage rate (e.g., 0.1 for 90% coverage)
    
    Returns:
        threshold: The conformity score threshold (q_hat)
    """
    n = len(conformity_scores)
    q_level = np.ceil((n + 1) * (1 - alpha)) / n
    threshold = np.quantile(conformity_scores, q_level)

    return threshold

# ==========================================
# PREDICTION FUNCTIONS (for each score type)
# ==========================================

def predict_with_absolute_score(
    probs: np.ndarray,
    threshold: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create prediction sets using absolute conformity score.
    
    Include class c if: 1 - p(c) <= threshold
    Equivalently: p(c) >= 1 - threshold
    
    Args:
        probs: (H, W, num_classes) array of probabilities
        threshold: Calibrated threshold
    
    Returns:
        prediction_sets: (H, W, num_classes) boolean array
        set_sizes: (H, W) array of set sizes
    """
    prediction_sets = probs >= (1 - threshold)
    set_sizes = prediction_sets.sum(axis=-1)

    return prediction_sets, set_sizes


def predict_with_gamma_score(
    probs: np.ndarray,
    threshold: float,
    epsilon: float = 1e-8
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create prediction sets using gamma conformity score.
    
    Include class c if: (1 - p(c)) / p(c) <= threshold
    Equivalently: p(c) >= 1 / (1 + threshold)
    
    Args:
        probs: (H, W, num_classes) array of probabilities
        threshold: Calibrated threshold
        epsilon: small constant for numerical stability
    
    Returns:
        prediction_sets: (H, W, num_classes) boolean array
        set_sizes: (H, W) array of set sizes
    """
    # From (1 - p) / p <= threshold
    # => 1 - p <= threshold * p
    # => 1 <= p * (1 + threshold)
    # => p >= 1 / (1 + threshold)
    
    prob_threshold = 1 / (1 + threshold)
    prediction_sets = probs >= prob_threshold
    set_sizes = prediction_sets.sum(axis=-1)
    
    return prediction_sets, set_sizes    


def predict_with_residual_normalized_score(
    probs: np.ndarray,
    threshold: float,
    learned_residuals: np.ndarray,
    epsilon: float = 1e-8
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create prediction sets using residual normalized conformity score.
    
    Include class c if: (1 - p(c)) / sigma_hat <= threshold
    Equivalently: p(c) >= 1 - threshold * sigma_hat
    
    Args:
        probs: (H, W, num_classes) array of probabilities
        threshold: Calibrated threshold
        learned_residuals: (H, W) array of predicted residual magnitudes
        epsilon: small constant
    
    Returns:
        prediction_sets: (H, W, num_classes) boolean array
        set_sizes: (H, W) array of set sizes
    """
    H, W, num_classes = probs.shape
    
    # Compute adaptive threshold for each pixel
    # p(c) >= 1 - threshold * sigma_hat(x)
    adaptive_threshold = 1 - threshold * learned_residuals
    
    # Broadcast to compare with all classes
    adaptive_threshold_expanded = adaptive_threshold[:, :, np.newaxis]  # (H, W, 1)
    
    prediction_sets = probs >= adaptive_threshold_expanded
    set_sizes = prediction_sets.sum(axis=-1)
    
    return prediction_sets, set_sizes

# ==========================================
# EVALUATION FUNCTION
# ==========================================

def evaluate_coverage(
    prediction_sets: np.ndarray,
    ground_truth: np.ndarray, 
    mask: np.ndarray = None
) -> Dict[str, float]:
    """
    Evaluate the empirical coverage of prediction sets.
    
    Args:
        prediction_sets: (H, W, num_classes) boolean array
        ground_truth: (H, W) array of true class indices
        mask: Optional (H, W) boolean mask for which pixels to evaluate
    
    Returns:
        metrics: Dictionary with coverage and other statistics
    """
    H, W, num_classes = prediction_sets.shape

    if mask is None:
        mask = np.ones((H,W), dtype=bool)

    # Check if true class is in prediction set
    coverage_map = prediction_Sets[
        np.arange(H)[: None],
        np.arange(W)[None, :],
        ground_truth
    ] # (H, W) boolean

    # Apply mask
    coverage_values = coverage_map[mask]

    # compute metrics
    empirical_coverage = coveage_values.mean()
    set_sizes = prediction_sets.sum(axis=-1)[mask]

    return {
        'coverage': empirical_coverage,
        'mean_set_size': set_sizes.mean(),
        'median_set_size': np.median(set_sizes),
        'min_set_size': set_sizes.min(),
        'max_set_size': set_sizes.max(),
        'std_set_size': set_sizes.std()
    }

# ==========================================
# COMPLETE WORKFLOW FUNCTION
# ==========================================

def run_conformal_experiment(
    probs_cal: np.ndarray,
    ground_truth_cal: np.ndarray,
    probs_test: np.ndarray,
    ground_truth_test: np.ndarray,
    score_type: str = 'absolute',
    alpha_values: List[float] = [0.05, 0.1, 0.2],
    learned_residuals_cal: np.ndarray = None,
    learned_residuals_test: np.ndarray = None
) -> Dict:
    """
    Run complete conformal prediction experiment with multiple alpha values.
    
    Args:
        probs_cal: Calibration probabilities (H, W, num_classes)
        ground_truth_cal: Calibration ground truth (H, W)
        probs_test: Test probabilities (H, W, num_classes)
        ground_truth_test: Test ground truth (H, W)
        score_type: 'absolute', 'gamma', or 'residual_normalized'
        alpha_values: List of miscoverage rates to test
        learned_residuals_cal: For residual_normalized score (H, W)
        learned_residuals_test: For residual_normalized score (H, W)
    
    Returns:
        results: Dictionary with results for each alpha
    """
    results = {}
    
    # Compute conformity scores on calibration set
    if score_type == 'absolute':
        cal_scores = absolute_conformity_score(probs_cal, ground_truth_cal)
    elif score_type == 'gamma':
        cal_scores = gamma_conformity_score(probs_cal, ground_truth_cal)
    elif score_type == 'residual_normalized':
        if learned_residuals_cal is None:
            raise ValueError("learned_residuals_cal required for residual_normalized score")
        cal_scores = residual_normalized_conformity_score(
            probs_cal, ground_truth_cal, learned_residuals_cal
        )
    else:
        raise ValueError(f"Unknown score_type: {score_type}")
    
    # Test each alpha value
    for alpha in alpha_values:
        # Calibrate threshold
        threshold = calibrate_conformal_predictor(cal_scores, alpha)
        
        # Make predictions on test set
        if score_type == 'absolute':
            pred_sets, set_sizes = predict_with_absolute_score(probs_test, threshold)
        elif score_type == 'gamma':
            pred_sets, set_sizes = predict_with_gamma_score(probs_test, threshold)
        elif score_type == 'residual_normalized':
            pred_sets, set_sizes = predict_with_residual_normalized_score(
                probs_test, threshold, learned_residuals_test
            )
        
        # Evaluate
        metrics = evaluate_coverage(pred_sets, ground_truth_test)
        
        # Store results
        results[alpha] = {
            'threshold': threshold,
            'target_coverage': 1 - alpha,
            'prediction_sets': pred_sets,
            'set_sizes': set_sizes,
            **metrics
        }
        
        # Print summary
        print(f"\n{'='*60}")
        print(f"Score: {score_type.upper()} | Alpha: {alpha} (Target: {100*(1-alpha):.0f}% coverage)")
        print(f"{'='*60}")
        print(f"Threshold: {threshold:.4f}")
        print(f"Empirical Coverage: {100*metrics['coverage']:.2f}%")
        print(f"Mean Set Size: {metrics['mean_set_size']:.2f}")
        print(f"Median Set Size: {metrics['median_set_size']:.1f}")
        print(f"Set Size Range: [{metrics['min_set_size']}, {metrics['max_set_size']}]")
    
    return results



    