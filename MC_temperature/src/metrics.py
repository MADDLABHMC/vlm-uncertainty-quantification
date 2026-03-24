"""
Evaluation metrics for Temperature-Scaled MC Dropout.
"""
import numpy as np


def compute_predictions_and_entropy(mean_probs: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute point predictions and entropy from mean probabilities.

    Returns:
        predictions: (H, W) argmax class indices
        entropy: (H, W) raw entropy
        normalized_entropy: (H, W) entropy / max_entropy [0, 1]
    """
    predictions = mean_probs.argmax(axis=-1)
    entropy = -np.sum(mean_probs * np.log(mean_probs + 1e-8), axis=-1)
    num_classes = mean_probs.shape[-1]
    max_entropy = np.log(num_classes)
    normalized_entropy = entropy / max_entropy
    return predictions, entropy, normalized_entropy


def compute_accuracy(
    predictions: np.ndarray,
    ground_truth: np.ndarray,
    class_names: list[str],
    verbose: bool = True,
) -> tuple[float, dict]:
    """Compute overall pixel accuracy and per-class accuracy."""
    valid_mask = ground_truth >= 0
    if valid_mask.sum() == 0:
        valid_mask = np.ones_like(ground_truth, dtype=bool)

    pred_valid = predictions[valid_mask]
    gt_valid = ground_truth[valid_mask]
    pixel_accuracy = float((pred_valid == gt_valid).mean()) if gt_valid.size > 0 else 0.0

    per_class_accuracy = {}
    for i, name in enumerate(class_names):
        class_mask = gt_valid == i
        count = int(class_mask.sum())
        if count > 0:
            acc = float((pred_valid[class_mask] == i).mean())
            per_class_accuracy[name] = (acc, count)
        else:
            per_class_accuracy[name] = (None, 0)

    if verbose:
        print(f"\n{'='*50}")
        print(f"PIXEL ACCURACY: {pixel_accuracy:.4f} ({pixel_accuracy*100:.2f}%)")
        print(f"{'='*50}")
        for name, (acc, count) in per_class_accuracy.items():
            if acc is not None:
                print(f"  {name:15s}: {acc:.4f} - {count} pixels")

    return pixel_accuracy, per_class_accuracy


def compute_uncertainty_stats(
    normalized_entropy: np.ndarray,
    predictions: np.ndarray,
    ground_truth: np.ndarray,
) -> dict:
    """Compute uncertainty vs correctness statistics."""
    errors = predictions != ground_truth
    correct = predictions == ground_truth

    return {
        "mean_uncertainty": float(normalized_entropy.mean()),
        "uncertainty_when_correct": float(normalized_entropy[correct].mean())
        if correct.sum() > 0
        else None,
        "uncertainty_when_wrong": float(normalized_entropy[errors].mean())
        if errors.sum() > 0
        else None,
    }
