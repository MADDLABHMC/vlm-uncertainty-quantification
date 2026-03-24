"""
Data loading and validation split for Temperature-Scaled MC Dropout calibration.
Mirrors MC_dropout/data_utils for consistency; adds train/val split for calibration.
"""
import numpy as np
import pandas as pd
import random
from PIL import Image
from pathlib import Path
import tifffile as tiff
from transformers.image_utils import load_image


DEFAULT_CLASSES = {
    "texts": [
        "paved-area",
        "dirt",
        "grass",
        "rocks",
        "vegetation",
        "person",
        "dog",
        "bicycle",
        "tree",
    ],
    "indices": [1, 2, 3, 6, 8, 15, 16, 18, 19],
}


def load_image_and_mask(
    image_path: str,
    mask_path: str | None = None,
    class_names: list[str] | None = None,
    class_indices: list[int] | None = None,
    target_size: tuple[int, int] = (352, 352),
):
    """Load image and optionally ground truth mask."""
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    image = load_image(str(path))

    if class_names is None:
        class_names = DEFAULT_CLASSES["texts"]
    if class_indices is None:
        class_indices = DEFAULT_CLASSES["indices"]

    ground_truth = None
    if mask_path:
        mask_path = Path(mask_path)
        if mask_path.exists():
            ground_truth = prepare_ground_truth(
                mask_path, class_indices, target_size
            )

    return image, ground_truth, class_names, class_indices


def prepare_ground_truth(
    mask_path: str | Path,
    class_indices: list[int],
    target_size: tuple[int, int] = (352, 352),
) -> np.ndarray:
    """Load and prepare ground truth mask for evaluation."""
    H, W = target_size
    mask = tiff.imread(str(mask_path))

    ground_truth = np.zeros((H, W), dtype=np.int64)
    for i, idx in enumerate(class_indices):
        channel = mask[:, :, idx]
        channel_resized = np.array(
            Image.fromarray(channel.astype(np.uint8)).resize(
                (W, H), Image.NEAREST
            )
        )
        ground_truth[channel_resized == 255] = i

    return ground_truth


def get_indices_for_classes(
    full_class_names: list[str], selected_names: list[str]
) -> list[int]:
    """Map selected class names to their indices in the full class list."""
    name_to_idx = {name: i for i, name in enumerate(full_class_names)}
    indices = []
    for name in selected_names:
        if name not in name_to_idx:
            raise ValueError(
                f"Unknown class '{name}'. Available: {full_class_names}"
            )
        indices.append(name_to_idx[name])
    return indices


def load_dataset_classes(dataset_path: str | Path) -> tuple[list[str], list[int]]:
    """Load class names and indices from dataset classes.csv."""
    dataset_path = Path(dataset_path)
    csv_path = dataset_path / "classes.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        class_names = df["name"].tolist()
        indices = list(range(len(class_names)))
        return class_names, indices
    return DEFAULT_CLASSES["texts"], DEFAULT_CLASSES["indices"]


def iter_dataset_pairs(
    dataset_path: str | Path,
    images_subdir: str = "images",
    labels_subdir: str = "labels/tiff",
    image_ext: str = ".jpg",
    label_ext: str = ".tiff",
) -> list[tuple[Path, Path]]:
    """Discover image/mask pairs in a Semantic Drone–style dataset."""
    dataset_path = Path(dataset_path)

    for images_dir, labels_dir in [
        (dataset_path / images_subdir, dataset_path / labels_subdir),
        (
            dataset_path / "aerial_semantic_drone" / "images",
            dataset_path / "aerial_semantic_drone" / "labels" / "tiff",
        ),
        (dataset_path / "images", dataset_path / "labels" / "tiff"),
    ]:
        if not images_dir.exists() or not labels_dir.exists():
            continue

        pairs = []
        for img_path in sorted(images_dir.glob(f"*{image_ext}")):
            stem = img_path.stem
            mask_path = labels_dir / f"{stem}{label_ext}"
            if not mask_path.exists():
                mask_path = labels_dir / f"{stem}.tif"
            if mask_path.exists():
                pairs.append((img_path, mask_path))

        if pairs:
            return pairs

    raise FileNotFoundError(
        f"No image/mask pairs found in {dataset_path}. "
        f"Expected layout: images/ and labels/tiff/ (or aerial_semantic_drone/...)"
    )


def create_train_val_split(
    dataset_path: str | Path,
    val_size: int = 5000,
    val_images_min: int = 50,
    seed: int = 42,
) -> tuple[list[tuple[Path, Path]], list[tuple[Path, Path]]]:
    """
    Split dataset into train and calibration (validation) sets.

    The paper reserves ~5,000 samples for calibration. For segmentation,
    we reserve enough images to get roughly val_size labeled pixels.
    Uses at least val_images_min validation images.

    Returns:
        train_pairs: List of (image_path, mask_path) for training
        val_pairs: List of (image_path, mask_path) for calibration
    """
    dataset_path = Path(dataset_path)
    pairs = list(iter_dataset_pairs(dataset_path))

    if len(pairs) < 2 * val_images_min:
        n_val = max(1, len(pairs) // 5)
    else:
        pixels_per_image = 352 * 352
        n_val_images = max(
            val_images_min,
            min(len(pairs) // 5, val_size // (pixels_per_image // 10)),
        )
        n_val = min(n_val_images, len(pairs) // 2)

    random.seed(seed)
    shuffled = pairs.copy()
    random.shuffle(shuffled)

    val_pairs = shuffled[:n_val]
    train_pairs = shuffled[n_val:]

    return train_pairs, val_pairs
