"""
Visualization utilities for conformal prediction results.
"""
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from typing import Optional


def visualize_results(
    image: Image.Image,
    ground_truth: np.ndarray,
    probs: np.ndarray,
    prediction_sets: np.ndarray,
    set_sizes: np.ndarray,
    save_path: Optional[str] = None
) -> plt.Figure:
    """Create comprehensive visualization of conformal prediction results.
    
    Args:
        image: Original input image
        ground_truth: Ground truth labels (H, W)
        probs: Model probabilities (H, W, num_classes)
        prediction_sets: Prediction sets (H, W, num_classes)
        set_sizes: Set sizes per pixel (H, W)
        save_path: Optional path to save figure
        
    Returns:
        Matplotlib figure object
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Original image
    axes[0, 0].imshow(image)
    axes[0, 0].set_title("Original Image", fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')
    
    # Ground truth segmentation
    axes[0, 1].imshow(ground_truth, cmap='tab10')
    axes[0, 1].set_title("Ground Truth Labels", fontsize=12, fontweight='bold')
    axes[0, 1].axis('off')
    
    # Most probable class (argmax)
    most_probable = probs.argmax(axis=-1)
    axes[0, 2].imshow(most_probable, cmap='tab10')
    axes[0, 2].set_title("Most Probable Class (No CP)", fontsize=12, fontweight='bold')
    axes[0, 2].axis('off')
    
    # Prediction set sizes (uncertainty map)
    im1 = axes[1, 0].imshow(set_sizes, cmap='hot', vmin=1, vmax=probs.shape[-1])
    axes[1, 0].set_title("Prediction Set Size\n(Uncertainty Map)", fontsize=12, fontweight='bold')
    axes[1, 0].axis('off')
    plt.colorbar(im1, ax=axes[1, 0], fraction=0.046, pad=0.04)
    
    # Confident predictions (set size <= 4)
    confident = (set_sizes <= 4)
    axes[1, 1].imshow(confident, cmap='RdYlGn')
    axes[1, 1].set_title(
        f"Confident Predictions\n(Set Size <= 4: {100*confident.mean():.1f}%)",
        fontsize=12, fontweight='bold'
    )
    axes[1, 1].axis('off')
    
    # # Uncertain predictions (set size >= 5)
    # uncertain = (set_sizes >= 5)
    # axes[1, 2].imshow(uncertain, cmap='RdYlGn_r')
    # axes[1, 2].set_title(
    #     f"Uncertain Predictions\n(Set Size >= 5: {100*uncertain.mean():.1f}%)",
    #     fontsize=12, fontweight='bold'
    # )
    # axes[1, 2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def print_statistics(
    set_sizes: np.ndarray,
    empirical_coverage: float,
    target_coverage: float,
    threshold: float
):
    """Print comprehensive statistics about conformal prediction.
    
    Args:
        set_sizes: Array of prediction set sizes (H, W)
        empirical_coverage: Achieved coverage on test set
        target_coverage: Target coverage level
        threshold: Conformity score threshold
    """
    H, W = set_sizes.shape
    total_pixels = H * W
    
    print(f"\n{'='*50}")
    print(f"CALIBRATION RESULTS")
    print(f"{'='*50}")
    print(f"Target coverage: {100*target_coverage:.1f}%")
    print(f"Conformity threshold: {threshold:.4f}")
    print(f"This means: include class if p(class) >= {1-threshold:.4f}")
    
    print(f"\n{'='*50}")
    print(f"PREDICTION SET STATISTICS")
    print(f"{'='*50}")
    print(f"Set size distribution:")
    print(f"  Min: {set_sizes.min()}")
    print(f"  Max: {set_sizes.max()}")
    print(f"  Mean: {set_sizes.mean():.2f}")
    print(f"  Median: {np.median(set_sizes):.1f}")
    
    print(f"\nPer-size counts:")
    max_classes = set_sizes.max()
    for size in range(1, int(max_classes) + 1):
        count = (set_sizes == size).sum()
        pct = 100 * count / total_pixels
        print(f"  Size {size}: {count:6d} pixels ({pct:5.2f}%)")
    
    print(f"\n{'='*50}")
    print(f"COVERAGE ON TEST PIXELS")
    print(f"{'='*50}")
    print(f"Target coverage: {100*target_coverage:.1f}%")
    print(f"Empirical coverage: {100*empirical_coverage:.2f}%")


def show_example_predictions(
    probs: np.ndarray,
    prediction_sets: np.ndarray,
    ground_truth: np.ndarray,
    test_mask: np.ndarray,
    class_names: list[str],
    num_examples: int = 5,
    seed: int = 42
):
    """Show example prediction sets for random test pixels.
    
    Args:
        probs: Probability array (H, W, num_classes)
        prediction_sets: Prediction sets (H, W, num_classes)
        ground_truth: Ground truth labels (H, W)
        test_mask: Boolean test mask (H, W)
        class_names: List of class name strings
        num_examples: Number of examples to show
        seed: Random seed
    """
    np.random.seed(seed)
    
    H, W = ground_truth.shape
    probs_flat = probs.reshape(-1, probs.shape[-1])
    prediction_sets_flat = prediction_sets.reshape(-1, prediction_sets.shape[-1])
    ground_truth_flat = ground_truth.flatten()
    test_indices = np.where(test_mask.flatten())[0]
    
    sample_pixels = np.random.choice(test_indices, size=num_examples, replace=False)
    
    print(f"\n{'='*50}")
    print(f"EXAMPLE PREDICTION SETS")
    print(f"{'='*50}")
    
    for pix_idx in sample_pixels:
        row = pix_idx // W
        col = pix_idx % W
        true_class = ground_truth_flat[pix_idx]
        pred_set = np.where(prediction_sets_flat[pix_idx])[0]
        
        print(f"\nPixel ({row}, {col}):")
        print(f"  True class: {class_names[true_class]}")
        print(f"  Prediction set: {[class_names[c] for c in pred_set]}")
        print(f"  Set size: {len(pred_set)}")
        print(f"  Coverage: {'[YES]' if true_class in pred_set else '[NO]'}")
        print(f"  Probabilities: {dict(zip(class_names, probs_flat[pix_idx].round(3)))}")