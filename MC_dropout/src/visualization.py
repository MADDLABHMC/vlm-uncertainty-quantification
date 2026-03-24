"""
Visualization utilities for MC Dropout results.
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
    """
    Create 2x3 visualization: image, ground truth, predictions, errors, uncertainty, histogram.
    """
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
    axes[0, 2].set_title("MC Dropout Predictions\n(Encoder Dropout)", fontsize=14)
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
        bins=50,
        alpha=0.5,
        label="Correct",
        color="green",
    )
    axes[1, 2].hist(
        normalized_entropy[errors_mask],
        bins=50,
        alpha=0.5,
        label="Wrong",
        color="red",
    )
    axes[1, 2].set_xlabel("Uncertainty (Norm. Entropy)", fontsize=12)
    axes[1, 2].set_ylabel("Frequency", fontsize=12)
    axes[1, 2].set_title("Uncertainty Distribution", fontsize=14)
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Visualization saved to {save_path}")

    return fig


def print_example_predictions(
    mean_probs: np.ndarray,
    std_probs: np.ndarray,
    predictions: np.ndarray,
    ground_truth: np.ndarray,
    normalized_entropy: np.ndarray,
    class_names: list[str],
    n_examples: int = 5,
    seed: int = 42,
):
    """Print example pixel predictions with uncertainty."""
    H, W = predictions.shape
    np.random.seed(seed)
    sample_rows = np.random.randint(0, H, size=n_examples)
    sample_cols = np.random.randint(0, W, size=n_examples)

    print(f"\n{'='*70}")
    print("EXAMPLE PREDICTIONS WITH UNCERTAINTY")
    print(f"{'='*70}")

    for i, (row, col) in enumerate(zip(sample_rows, sample_cols)):
        true_class = ground_truth[row, col]
        pred_class = predictions[row, col]
        is_correct = true_class == pred_class

        pixel_probs = mean_probs[row, col]
        pixel_std = std_probs[row, col]
        pixel_entropy = normalized_entropy[row, col]

        print(f"\nPixel ({row}, {col}):")
        print(f"  True class:      {class_names[true_class]}")
        print(f"  Predicted class: {class_names[pred_class]}")
        print(f"  Correct: {'✓' if is_correct else '✗'}")
        print(f"  Uncertainty: {pixel_entropy:.6f}")
        print(f"  Top-3 predictions (prob ± std):")

        top3_indices = np.argsort(pixel_probs)[::-1][:3]
        for idx in top3_indices:
            prob = pixel_probs[idx]
            std = pixel_std[idx]
            marker = "->" if idx == pred_class else "  "
            print(f"    {marker} {class_names[idx]:15s}: {prob:.4f} ± {std:.6f}")


def visualize_convergence(df_convergence, save_path: Optional[str | Path] = None):
    """Visualize convergence study results (accuracy, uncertainty, time vs n_samples)."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # 1. Pixel Accuracy vs Samples
    col = "pixel_accuracy" if "pixel_accuracy" in df_convergence.columns else "accuracy"
    axes[0, 0].plot(
        df_convergence["n_samples"],
        df_convergence[col],
        "o-",
        linewidth=2,
        markersize=8,
    )
    axes[0, 0].set_xlabel("Number of MC Samples", fontsize=12)
    axes[0, 0].set_ylabel("Pixel Accuracy", fontsize=12)
    axes[0, 0].set_title("Pixel Accuracy vs MC Samples", fontsize=14, fontweight="bold")
    axes[0, 0].grid(True, alpha=0.3)

    # 2. Mean Uncertainty vs Samples
    axes[0, 1].plot(
        df_convergence["n_samples"],
        df_convergence["mean_uncertainty"],
        "o-",
        linewidth=2,
        markersize=8,
        color="orange",
    )
    axes[0, 1].set_xlabel("Number of MC Samples", fontsize=12)
    axes[0, 1].set_ylabel("Mean Uncertainty", fontsize=12)
    axes[0, 1].set_title("Mean Uncertainty vs MC Samples", fontsize=14, fontweight="bold")
    axes[0, 1].grid(True, alpha=0.3)

    # 3. Uncertainty Separation
    if "uncertainty_separation" in df_convergence.columns:
        axes[0, 2].plot(
            df_convergence["n_samples"],
            df_convergence["uncertainty_separation"],
            "o-",
            linewidth=2,
            markersize=8,
            color="green",
        )
        axes[0, 2].set_xlabel("Number of MC Samples", fontsize=12)
        axes[0, 2].set_ylabel("Uncertainty Separation", fontsize=12)
        axes[0, 2].set_title(
            "Uncertainty Separation vs MC Samples", fontsize=14, fontweight="bold"
        )
        axes[0, 2].grid(True, alpha=0.3)

    # 4. Time vs Samples
    axes[1, 0].plot(
        df_convergence["n_samples"],
        df_convergence["time_seconds"],
        "o-",
        linewidth=2,
        markersize=8,
        color="red",
    )
    axes[1, 0].set_xlabel("Number of MC Samples", fontsize=12)
    axes[1, 0].set_ylabel("Time (seconds)", fontsize=12)
    axes[1, 0].set_title("Computational Cost vs MC Samples", fontsize=14, fontweight="bold")
    axes[1, 0].grid(True, alpha=0.3)

    # 5. Relative change in uncertainty
    relative_changes = [0.0]
    for i in range(1, len(df_convergence)):
        prev = df_convergence["mean_uncertainty"].iloc[i - 1]
        curr = df_convergence["mean_uncertainty"].iloc[i]
        if prev > 0:
            relative_changes.append(abs(curr - prev) / prev * 100)
        else:
            relative_changes.append(0.0)
    axes[1, 1].plot(
        df_convergence["n_samples"], relative_changes, "o-",
        linewidth=2, markersize=8, color="purple",
    )
    axes[1, 1].set_xlabel("Number of MC Samples", fontsize=12)
    axes[1, 1].set_ylabel("% Change from Previous", fontsize=12)
    axes[1, 1].set_title("Uncertainty Stability", fontsize=14, fontweight="bold")
    axes[1, 1].axhline(y=5, color="orange", linestyle="--", alpha=0.5)
    axes[1, 1].grid(True, alpha=0.3)

    # 6. Correct vs Wrong uncertainty
    uc = df_convergence.get("uncertainty_when_correct")
    uw = df_convergence.get("uncertainty_when_wrong")
    if uc is not None and uw is not None and uc.notna().any() and uw.notna().any():
        uc = uc.fillna(0)
        uw = uw.fillna(0)
        x_pos = np.arange(len(df_convergence))
        width = 0.35
        axes[1, 2].bar(x_pos - width / 2, uc, width, label="Correct", alpha=0.8, color="green")
        axes[1, 2].bar(x_pos + width / 2, uw, width, label="Wrong", alpha=0.8, color="red")
        axes[1, 2].set_xlabel("Number of MC Samples", fontsize=12)
        axes[1, 2].set_ylabel("Mean Uncertainty", fontsize=12)
        axes[1, 2].set_title("Uncertainty: Correct vs Wrong", fontsize=14, fontweight="bold")
        axes[1, 2].set_xticks(x_pos)
        axes[1, 2].set_xticklabels(df_convergence["n_samples"])
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3, axis="y")
    else:
        axes[1, 2].axis("off")

    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Convergence plot saved to {save_path}")

    return fig
