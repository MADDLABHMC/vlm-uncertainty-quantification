"""
Evaluation metrics for MC Dropout predictions.
"""
import numpy as np
from typing import Optional


def compute_predictions_and_entropy(mean_probs: np.ndarray):
    """
    Compute point predictions and predictive entropy from mean probabilities.

    Args:
        mean_probs: (H, W, num_classes) array

    Returns:
        predictions: (H, W) argmax class indices
        entropy: (H, W) raw entropy
        normalized_entropy: (H, W) entropy / max_entropy [0, 1]
    """
    predictions = mean_probs.argmax(axis=-1)
    entropy = -np.sum(
        mean_probs * np.log(mean_probs + 1e-8), axis=-1
    )
    num_classes = mean_probs.shape[-1]
    max_entropy = np.log(num_classes)
    normalized_entropy = entropy / max_entropy
    return predictions, entropy, normalized_entropy


def compute_iou(
    predictions: np.ndarray,
    ground_truth: np.ndarray,
    num_classes: int,
    ignore_index: int = -1,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute per-class IoU (Intersection over Union).

    IoU_c = TP_c / (TP_c + FP_c + FN_c) = intersection / union for class c.

    Args:
        predictions: (H, W) predicted class indices
        ground_truth: (H, W) ground truth class indices
        num_classes: Number of classes
        ignore_index: Ignore pixels with this GT value (e.g. unlabeled)

    Returns:
        iou_per_class: (num_classes,) IoU per class, NaN where class has no pixels
        present_mask: (num_classes,) True for classes present in ground truth
    """
    valid_mask = ground_truth != ignore_index
    pred_valid = predictions[valid_mask]
    gt_valid = ground_truth[valid_mask]

    iou_per_class = np.full(num_classes, np.nan)
    for c in range(num_classes):
        pred_c = pred_valid == c
        gt_c = gt_valid == c
        intersection = (pred_c & gt_c).sum()
        union = (pred_c | gt_c).sum()
        if union > 0:
            iou_per_class[c] = intersection / union

    present_mask = np.array(
        [(ground_truth == c).sum() > 0 for c in range(num_classes)]
    )
    return iou_per_class, present_mask


def compute_accuracy(
    predictions: np.ndarray,
    ground_truth: np.ndarray,
    class_names: list[str],
    verbose: bool = True,
) -> tuple[float, dict]:
    """
    Compute overall pixel accuracy and per-class pixel accuracy.

    Returns:
        pixel_accuracy: float - Overall accuracy over valid pixels
        per_class_accuracy: dict mapping class name -> (accuracy, pixel_count) or (None, 0)
    """
    valid_mask = ground_truth >= 0  # Ignore unlabeled if encoded as -1
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
        print("PIXEL ACCURACY: {:.4f} ({:.2f}%)".format(pixel_accuracy, pixel_accuracy * 100))
        print(f"{'='*50}")
        print("\nPER-CLASS PIXEL ACCURACY")
        print(f"{'='*50}")
        for name, (acc, count) in per_class_accuracy.items():
            if acc is not None:
                print(f"  {name:15s}: {acc:.4f} ({acc*100:.2f}%) - {count} pixels")
            else:
                print(f"  {name:15s}: No pixels in ground truth")

    return pixel_accuracy, per_class_accuracy


def compute_uncertainty_stats(
    normalized_entropy: np.ndarray,
    predictions: np.ndarray,
    ground_truth: np.ndarray,
    class_names: list[str],
    verbose: bool = True,
) -> dict:
    """
    Compute uncertainty statistics and calibration (uncertainty vs correctness).
    """
    errors = predictions != ground_truth
    correct = predictions == ground_truth

    uncertainty_when_wrong = normalized_entropy[errors]
    uncertainty_when_correct = normalized_entropy[correct]

    stats = {
        "mean_uncertainty": float(normalized_entropy.mean()),
        "uncertainty_when_correct": float(uncertainty_when_correct.mean())
        if correct.sum() > 0
        else None,
        "uncertainty_when_wrong": float(uncertainty_when_wrong.mean())
        if errors.sum() > 0
        else None,
        "uncertainty_separation": None,
    }
    if stats["uncertainty_when_correct"] is not None and stats["uncertainty_when_wrong"] is not None:
        stats["uncertainty_separation"] = (
            stats["uncertainty_when_wrong"] - stats["uncertainty_when_correct"]
        )

    if verbose:
        print(f"\n{'='*50}")
        print("UNCERTAINTY STATISTICS (Encoder Dropout)")
        print(f"{'='*50}")
        print(f"Mean uncertainty: {stats['mean_uncertainty']:.6f}")
        print(f"\nUNCERTAINTY VS PREDICTION CORRECTNESS")
        print(f"{'='*50}")
        uc = stats["uncertainty_when_correct"]
        uw = stats["uncertainty_when_wrong"]
        if uc is not None:
            print(f"Mean uncertainty when CORRECT: {uc:.6f}")
        if uw is not None:
            print(f"Mean uncertainty when WRONG:   {uw:.6f}")
        if stats["uncertainty_separation"] is not None:
            print(f"Difference: {stats['uncertainty_separation']:.6f}")

        print(f"\nPER-CLASS UNCERTAINTY (Mean ± Std)")
        print(f"{'='*50}")
        for i, name in enumerate(class_names):
            class_mask = ground_truth == i
            if class_mask.sum() > 0:
                ce = normalized_entropy[class_mask]
                print(f"  {name:15s}: {ce.mean():.6f} ± {ce.std():.6f}")
            else:
                print(f"  {name:15s}: No pixels in ground truth")

    return stats
