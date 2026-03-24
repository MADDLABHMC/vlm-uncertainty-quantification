"""
Image transforms for robustness evaluation (from Image_Transform.ipynb).
Accepts PIL Images, returns PIL Images.
"""
import numpy as np
from PIL import Image

try:
    import cv2
except ImportError:
    raise ImportError("opencv-python required for transforms. pip install opencv-python")


def _pil_to_np(pil_image: Image.Image) -> np.ndarray:
    """PIL RGB -> numpy RGB (H, W, 3)."""
    return np.array(pil_image)


def _np_to_pil(arr: np.ndarray) -> Image.Image:
    """Numpy RGB (H, W, 3) -> PIL Image."""
    return Image.fromarray(arr.astype(np.uint8))


def gaussian_blur(image: Image.Image, sigma: float = 10) -> Image.Image:
    """Apply Gaussian blur. sigma controls blur strength."""
    arr = _pil_to_np(image)
    blurred = cv2.GaussianBlur(arr, (0, 0), sigma)
    return _np_to_pil(blurred)


def vignette(image: Image.Image, level: float = 5) -> Image.Image:
    """Apply vignette effect. Lower level = stronger vignette."""
    arr = _pil_to_np(image)
    height, width = arr.shape[:2]

    x_kernel = cv2.getGaussianKernel(width, width / level)
    y_kernel = cv2.getGaussianKernel(height, height / level)
    kernel = y_kernel * x_kernel.T
    mask = kernel / kernel.max()

    out = arr.copy()
    for i in range(3):
        out[:, :, i] = (out[:, :, i] * mask).astype(np.uint8)
    return _np_to_pil(out)


def stress_occlude(
    image: Image.Image,
    num_rois: int = 2,
    opacity: float = 1.0,
    seed: int | None = None,
) -> Image.Image:
    """Occlude random patches (darken). num_rois=number of patches, opacity=darkening (1=black)."""
    arr = _pil_to_np(image)
    h, w = arr.shape[:2]
    rng = np.random.default_rng(seed)

    for _ in range(num_rois):
        rw = int(np.clip(rng.normal(w // 4, w // 4), w // 8, w // 2))
        rh = int(np.clip(rng.normal(h // 4, h // 4), h // 8, h // 2))
        rw, rh = max(1, rw), max(1, rh)
        x = rng.integers(0, max(1, w - rw + 1))
        y = rng.integers(0, max(1, h - rh + 1))
        arr[y : y + rh, x : x + rw] = (opacity * arr[y : y + rh, x : x + rw]).astype(np.uint8)

    return _np_to_pil(arr)


def smoke(
    image: Image.Image,
    num_rois: int = 2,
    opacity: float = 1.0,
    seed: int | None = None,
) -> Image.Image:
    """Add smoke-like patches (whiten). num_rois=number of patches, opacity=blend (0=full white)."""
    arr = _pil_to_np(image).astype(np.float32)
    h, w = arr.shape[:2]
    rng = np.random.default_rng(seed)

    for _ in range(num_rois):
        rw = int(np.clip(rng.normal(w // 4, w // 4), w // 8, w // 2))
        rh = int(np.clip(rng.normal(h // 4, h // 4), h // 8, h // 2))
        rw, rh = max(1, rw), max(1, rh)
        x = rng.integers(0, max(1, w - rw + 1))
        y = rng.integers(0, max(1, h - rh + 1))
        arr[y : y + rh, x : x + rw] = (
            opacity * arr[y : y + rh, x : x + rw] + (1 - opacity) * 255
        )

    return _np_to_pil(np.clip(arr, 0, 255).astype(np.uint8))


TRANSFORMS = {
    "gaussian_blur": gaussian_blur,
    "vignette": vignette,
    "occlusions": stress_occlude,
    "smoke": smoke,
}


def apply_transform(
    image: Image.Image,
    transform_name: str,
    **kwargs,
) -> Image.Image:
    """
    Apply a named transform to a PIL image.

    Args:
        image: PIL Image (RGB)
        transform_name: One of gaussian_blur, vignette, occlusions, smoke
        **kwargs: Transform-specific params (e.g. sigma, level, num_rois, opacity, seed)

    Returns:
        Transformed PIL Image
    """
    if transform_name not in TRANSFORMS:
        raise ValueError(
            f"Unknown transform '{transform_name}'. "
            f"Choose from: {list(TRANSFORMS.keys())}"
        )
    return TRANSFORMS[transform_name](image, **kwargs)
