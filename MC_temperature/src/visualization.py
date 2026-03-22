"""
Visualization utilities for Temperature-Scaled MC Dropout results.
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional


def visualize_results(
    image,
    ground_truth: np.ndarray,
    predictions: np.ndarray,
    normalized_entropy: np.ndarray,
    pixel_accuracy: float,
    class_names: list[str],
    save_path: Optional[str | Path] = None,
) -> plt.Figure:
    """Create 2x3 visualization for MC Temperature results."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    axes[0, 0].imshow(image)
    axes[0, 0].set_title("Original Image", fontsize=14)
    axes[0, 0].axis("off")

    im1 = axes[0, 1].imshow(
        ground_truth, cmap="tab10", vmin=0, vmax=max(len(class_names) - 1, 1)
    )
    axes[0, 1].set_title("Ground Truth", fontsize=14)
    axes[0, 1].axis("off")
    plt.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)

    im2 = axes[0, 2].imshow(
        predictions, cmap="tab10", vmin=0, vmax=max(len(class_names) - 1, 1)
    )
    axes[0, 2].set_title("MC Temperature Predictions\n(T-scaled, N=25)", fontsize=14)
    axes[0, 2].axis("off")
    plt.colorbar(im2, ax=axes[0, 2], fraction=0.046, pad=0.04)

    error_map = (predictions != ground_truth).astype(float)
    im3 = axes[1, 0].imshow(error_map, cmap="RdYlGn_r", vmin=0, vmax=1)
    axes[1, 0].set_title(
        f"Errors (Red=Wrong, Green=Correct)\nPixel Accuracy: {pixel_accuracy:.4f}",
        fontsize=14,
    )
    axes[1, 0].axis("off")
    plt.colorbar(im3, ax=axes[1, 0], fraction=0.046, pad=0.04)

    im4 = axes[1, 1].imshow(normalized_entropy, cmap="hot")
    axes[1, 1].set_title(
        f"Epistemic Uncertainty (Norm. Entropy)\nMean: {normalized_entropy.mean():.6f}",
        fontsize=14,
    )
    axes[1, 1].axis("off")
    plt.colorbar(im4, ax=axes[1, 1], fraction=0.046, pad=0.04)

    errors_mask = predictions != ground_truth
    correct_mask = predictions == ground_truth
    axes[1, 2].hist(
        normalized_entropy[correct_mask],
        bins=50, alpha=0.3, label="Correct", color="green",
    )
    axes[1, 2].hist(
        normalized_entropy[errors_mask],
        bins=50, alpha=0.3, label="Wrong", color="red",
    )
    axes[1, 2].set_xlabel("Uncertainty (Norm. Entropy)")
    axes[1, 2].set_ylabel("Frequency")
    axes[1, 2].set_title("Uncertainty Distribution")
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
