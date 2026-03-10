"""
Temperature-scaled MC Dropout inference for CLIPSeg.

Implements: p̂(x) = (1/N) Σ σ_SM(T^{-1} f_{w_i}(x))
With optional uncertainty threshold H̃_max for rejection in safety-critical applications.
"""
import numpy as np
import torch
from tqdm import tqdm

from .model import CLIPSegWithDecoderDropout


def mc_temperature_predict(
    model: CLIPSegWithDecoderDropout,
    processor,
    image,
    texts: list[str],
    temperature: float = 1.0,
    n_samples: int = 25,
    device: str = "cpu",
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Perform temperature-scaled MC Dropout inference.

    p̂(x) = (1/N) Σ softmax(logits_i / T)

    Args:
        model: CLIPSegWithDecoderDropout
        processor: CLIPSeg processor
        image: PIL Image
        texts: List of class names
        temperature: Calibrated temperature T > 0 (default 1.0 = no scaling)
        n_samples: Number of MC forward passes (default 25 per paper)
        device: 'cpu' or 'cuda'
        verbose: Whether to show progress bar

    Returns:
        mean_probs: (H, W, num_classes) - Mean calibrated probabilities
        std_probs: (H, W, num_classes) - Std across MC samples
        normalized_entropy: (H, W) - Predictive entropy / max_entropy [0,1]
    """
    model.to(device)
    model.enable_dropout()

    inputs = processor(
        text=texts,
        images=[image] * len(texts),
        padding=True,
        return_tensors="pt",
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    all_probs = []
    iterator = range(n_samples)
    if verbose:
        iterator = tqdm(iterator, desc="MC Temperature samples")

    with torch.no_grad():
        for _ in iterator:
            outputs = model(**inputs)
            logits = outputs.logits  # (num_classes, H, W)
            # Apply temperature before softmax: σ_SM(T^{-1} f(x))
            scaled = logits / temperature
            probs = torch.softmax(scaled, dim=0)
            all_probs.append(probs.cpu().numpy())

    all_probs = np.stack(all_probs, axis=0)  # (n_samples, C, H, W)
    mean_probs = all_probs.mean(axis=0)
    std_probs = all_probs.std(axis=0)

    # (C, H, W) -> (H, W, C)
    mean_probs = mean_probs.transpose(1, 2, 0)
    std_probs = std_probs.transpose(1, 2, 0)

    # Predictive entropy: H = -Σ p log p, normalized to [0, 1]
    entropy = -np.sum(
        mean_probs * np.log(mean_probs + 1e-8), axis=-1
    )
    num_classes = mean_probs.shape[-1]
    max_entropy = np.log(num_classes)
    normalized_entropy = np.clip(entropy / max_entropy, 0, 1)

    return mean_probs, std_probs, normalized_entropy


def predict_with_rejection(
    mean_probs: np.ndarray,
    normalized_entropy: np.ndarray,
    H_max: float | None = None,
    reject_value: int = -1,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply optional uncertainty threshold for rejection.

    Per the paper: set H̃_max and reject predictions exceeding it for
    safety-critical applications.

    Args:
        mean_probs: (H, W, C) probabilities
        normalized_entropy: (H, W) uncertainty in [0, 1]
        H_max: If set, pixels with entropy > H_max are rejected (reject_value)
        reject_value: Class index for rejected pixels (default -1)

    Returns:
        predictions: (H, W) argmax class, or reject_value where rejected
        rejected_mask: (H, W) boolean, True where rejected
    """
    predictions = mean_probs.argmax(axis=-1).astype(np.int64)

    if H_max is None:
        return predictions, np.zeros_like(predictions, dtype=bool)

    rejected = normalized_entropy > H_max
    predictions = predictions.copy()
    predictions[rejected] = reject_value
    return predictions, rejected
