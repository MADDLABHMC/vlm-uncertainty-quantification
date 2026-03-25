"""
MC Dropout inference for CLIPSeg.
"""
import numpy as np
import torch
from tqdm import tqdm

from .model import CLIPSegWithEncoderDropout


def mc_dropout_predict(
    model: CLIPSegWithEncoderDropout,
    processor,
    image,
    texts: list[str],
    n_samples: int = 500,
    device: str = "cpu",
    verbose: bool = True,
):
    """
    Perform MC Dropout inference with encoder dropout.

    Args:
        model: CLIPSegWithEncoderDropout model
        processor: CLIPSeg processor
        image: PIL Image
        texts: List of class names
        n_samples: Number of forward passes (default: 30)
        device: 'cpu' or 'cuda'
        verbose: Whether to show progress bar

    Returns:
        mean_probs: (H, W, num_classes) - Mean softmax probabilities across samples
        std_probs: (H, W, num_classes) - Standard deviation (uncertainty)
        all_probs: (n_samples, num_classes, H, W) - All predictions
    """
    model.to(device)

    inputs = processor(
        text=texts,
        images=[image] * len(texts),
        padding=True,
        return_tensors="pt",
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    model.enable_dropout()

    all_probs = []
    iterator = range(n_samples)
    if verbose:
        iterator = tqdm(iterator, desc="MC Dropout samples")

    with torch.no_grad():
        for _ in iterator:
            outputs = model(**inputs)
            logits = outputs.logits  # (num_classes, H, W)
            probs = torch.softmax(logits, dim=0)
            all_probs.append(probs.cpu().numpy())

            del outputs, logits, probs
            if device == "cuda":
                torch.cuda.empty_cache()

    all_probs = np.stack(all_probs, axis=0)
    mean_probs = all_probs.mean(axis=0)
    std_probs = all_probs.std(axis=0)

    mean_probs = mean_probs.transpose(1, 2, 0)
    std_probs = std_probs.transpose(1, 2, 0)

    return mean_probs, std_probs, all_probs
