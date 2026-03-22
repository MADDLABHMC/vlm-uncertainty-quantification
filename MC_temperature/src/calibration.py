"""
Temperature calibration for MC Dropout.

Optimize temperature T on a held-out validation set by minimizing NLL.
Per the paper: with model weights frozen, find T > 0 that minimizes
negative log-likelihood using MC dropout.
"""
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm

from .model import CLIPSegWithDecoderDropout
from .data_utils import (
    iter_dataset_pairs,
    load_image_and_mask,
    load_dataset_classes,
    get_indices_for_classes,
    create_train_val_split,
)


def _compute_nll(
    mean_probs: np.ndarray,
    ground_truth: np.ndarray,
    ignore_index: int = -1,
) -> float:
    """
    Compute mean negative log-likelihood over valid pixels.

    mean_probs: (H, W, num_classes) - softmax probabilities
    ground_truth: (H, W) - class indices, use ignore_index for unlabeled
    """
    valid = ground_truth >= 0
    if valid.sum() == 0:
        return float("inf")

    H, W, C = mean_probs.shape
    gt_flat = ground_truth[valid].astype(np.int64)
    probs_flat = mean_probs[valid]  # (N, C)

    # p[c] for true class c
    p_true = np.take_along_axis(
        probs_flat, gt_flat[:, np.newaxis], axis=1
    ).squeeze(1)

    # Clamp to avoid log(0)
    p_true = np.clip(p_true, 1e-8, 1.0)
    nll = -np.log(p_true).mean()
    return float(nll)


def _collect_mc_logits(
    model: CLIPSegWithDecoderDropout,
    processor,
    image,
    texts: list[str],
    n_samples: int,
    device: str,
) -> np.ndarray:
    """
    Run MC dropout, return raw logits (no temperature).
    Returns: (n_samples, num_classes, H, W)
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

    all_logits = []
    with torch.no_grad():
        for _ in range(n_samples):
            outputs = model(**inputs)
            logits = outputs.logits.cpu().numpy()  # (num_classes, H, W)
            all_logits.append(logits)

    return np.stack(all_logits, axis=0)


def _mean_probs_from_logits(logits: np.ndarray, temperature: float) -> np.ndarray:
    """
    Compute p̂ = (1/N) Σ softmax(logits_i / T).
    logits: (n_samples, C, H, W) - softmax over axis=1 (class dim)
    """
    scaled = logits / temperature
    max_scaled = scaled.max(axis=1, keepdims=True)
    probs = np.exp(scaled - max_scaled)
    probs = probs / probs.sum(axis=1, keepdims=True)
    mean_probs = probs.mean(axis=0)  # (C, H, W)
    return mean_probs.transpose(1, 2, 0)  # (H, W, C)


def _objective(
    log_T: float,
    val_logits_gt: list[tuple[np.ndarray, np.ndarray]],
) -> float:
    """
    NLL for a given temperature T = exp(log_T).
    val_logits_gt: list of (logits, ground_truth); logits shape (n_samples, C, H, W)
    """
    T = np.exp(log_T)
    if T < 0.01 or T > 100:
        return 1e6

    total_nll = 0.0
    n_pixels = 0

    for logits, ground_truth in val_logits_gt:
        mean_probs = _mean_probs_from_logits(logits, T)
        valid = ground_truth >= 0
        if valid.sum() == 0:
            continue
        nll = _compute_nll(mean_probs, ground_truth)
        n_pixels += valid.sum()
        total_nll += nll * valid.sum()

    if n_pixels == 0:
        return 1e6
    return total_nll / n_pixels


def calibrate_temperature(
    model: CLIPSegWithDecoderDropout,
    processor,
    dataset_path: str | Path,
    n_mc_samples: int = 25,
    val_images_min: int = 50,
    max_val_images: int = 100,
    device: str | None = None,
    class_names: list[str] | None = None,
    class_indices: list[int] | None = None,
    seed: int = 42,
    verbose: bool = True,
) -> float:
    """
    Optimize temperature T on validation set to minimize NLL.

    Args:
        model: CLIPSegWithDecoderDropout (weights frozen)
        processor: CLIPSeg processor
        dataset_path: Path to dataset (Semantic Drone layout)
        n_mc_samples: Number of MC dropout samples (default 25 per paper)
        val_images_min: Minimum validation images
        max_val_images: Cap validation images for speed
        device: cuda/cpu
        class_names: Optional subset of classes
        class_indices: Optional mask indices
        seed: Random seed for split
        verbose: Print progress

    Returns:
        Optimal temperature T > 0
    """
    try:
        from scipy.optimize import minimize_scalar
    except ImportError:
        raise ImportError("scipy required for temperature calibration. pip install scipy")

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    dataset_path = Path(dataset_path)

    # Get class config
    full_class_names, full_indices = load_dataset_classes(dataset_path)
    if class_names is not None:
        if class_indices is None:
            class_indices = get_indices_for_classes(full_class_names, class_names)
    else:
        class_names = full_class_names
        class_indices = full_indices

    # Create validation split
    train_pairs, val_pairs = create_train_val_split(
        dataset_path,
        val_size=5000,
        val_images_min=val_images_min,
        seed=seed,
    )
    val_pairs = val_pairs[:max_val_images]

    if verbose:
        print(f"Precomputing MC logits for {len(val_pairs)} validation images (n_mc={n_mc_samples})...")

    # Precompute MC logits for all validation images (expensive, done once)
    val_logits_gt = []
    for img_path, mask_path in tqdm(val_pairs, disable=not verbose):
        image, ground_truth, _, _ = load_image_and_mask(
            str(img_path),
            str(mask_path),
            class_names=class_names,
            class_indices=class_indices,
        )
        if ground_truth is None:
            continue
        logits = _collect_mc_logits(
            model, processor, image, class_names,
            n_samples=n_mc_samples, device=device,
        )
        val_logits_gt.append((logits, ground_truth))

    if not val_logits_gt:
        raise ValueError("No valid validation samples with ground truth")

    if verbose:
        print(f"Optimizing temperature T (minimize NLL)...")

    # Optimize log(T) so T = exp(log_T) is always > 0
    def obj(log_T):
        return _objective(float(log_T), val_logits_gt)

    result = minimize_scalar(
        obj,
        bounds=(-2, 5),  # T in [exp(-2), exp(5)] ≈ [0.14, 148]
        method="bounded",
        options={"xatol": 0.01},
    )

    T_opt = np.exp(result.x)
    if verbose:
        print(f"Optimal T = {T_opt:.4f}, NLL = {result.fun:.4f}")

    return float(T_opt)
