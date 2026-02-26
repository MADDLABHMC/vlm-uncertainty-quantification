"""
Minimal example for MC Dropout with CLIPSeg.

Run from MC_dropout directory:
    python example.py

Edit paths below to point to your image and mask files.
"""
import torch

from src import load_model, mc_dropout_predict, load_image_and_mask
from src.metrics import compute_predictions_and_entropy, compute_accuracy

# Configure paths
IMAGE_PATH = "MC_dropout/input/021.jpg"
MASK_PATH = "MC_dropout/input/021.tiff"  # set to None for inference-only


def _get_device():
    """Auto-detect device: CUDA > MPS (Apple Silicon) > CPU."""
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main():
    print("Loading model...")
    model, processor = load_model(dropout_rate=0.1)

    print("Loading image...")
    image, ground_truth, class_names, _ = load_image_and_mask(
        IMAGE_PATH, MASK_PATH
    )

    device = _get_device()
    print(f"Running MC Dropout inference (30 samples) on {device}...")
    mean_probs, std_probs, _ = mc_dropout_predict(
        model, processor, image, class_names,
        n_samples=30, device=device, verbose=True
    )

    predictions, _, normalized_entropy = compute_predictions_and_entropy(mean_probs)
    print(f"\nPredictions shape: {predictions.shape}")
    print(f"Mean uncertainty: {normalized_entropy.mean():.6f}")

    if ground_truth is not None:
        mean_iou, _ = compute_accuracy(
            predictions, ground_truth, class_names, verbose=False
        )
        print(f"Mean IoU: {mean_iou:.4f}")

if __name__ == "__main__":
    main()
