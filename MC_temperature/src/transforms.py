"""
Image transforms for robustness evaluation (MC_temperature).
Includes monochrome (grayscale as RGB) plus transforms from Image_Transform.ipynb.
"""
import numpy as np
from PIL import Image

try:
    import cv2
except ImportError:
    raise ImportError("opencv-python required for transforms. pip install opencv-python")


def _pil_to_np(pil_image: Image.Image) -> np.ndarray:
    return np.array(pil_image)


def _np_to_pil(arr: np.ndarray) -> Image.Image:
    return Image.fromarray(arr.astype(np.uint8))


def monochrome(image: Image.Image) -> Image.Image:
    """Convert to grayscale and stack to RGB (CLIPSeg expects 3 channels)."""
    gray = image.convert("L")
    return gray.convert("RGB")


def gaussian_blur(image: Image.Image, sigma: float = 10) -> Image.Image:
    arr = _pil_to_np(image)
    blurred = cv2.GaussianBlur(arr, (0, 0), sigma)
    return _np_to_pil(blurred)


def vignette(image: Image.Image, level: float = 5) -> Image.Image:
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
    "monochrome": monochrome,
    "gaussian_blur": gaussian_blur,
    "vignette": vignette,
    "occlusions": stress_occlude,
    "smoke": smoke,
}


def apply_transform(image: Image.Image, transform_name: str, **kwargs) -> Image.Image:
    if transform_name not in TRANSFORMS:
        raise ValueError(
            f"Unknown transform '{transform_name}'. Choose from: {list(TRANSFORMS.keys())}"
        )
    return TRANSFORMS[transform_name](image, **kwargs)
