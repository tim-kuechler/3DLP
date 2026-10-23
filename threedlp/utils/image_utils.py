import cv2
import numpy as np
from numpy.typing import NDArray


def resize_image(
    image: np.ndarray | None,
    target_size: tuple[int, int] | None,
    *,
    allow_upscale: bool = True,
) -> np.ndarray | None:
    """Resize an image to ``(height, width)`` using area interpolation."""
    if image is None or target_size is None:
        return image

    target_h, target_w = target_size
    if target_h <= 0 or target_w <= 0:
        raise ValueError("Target dimensions must be positive.")
    h, w = image.shape[:2]

    if not allow_upscale and (target_h > h or target_w > w):
        raise ValueError(f"Requested size {(target_h, target_w)} is larger than the original image size {(h, w)}.")

    if (h, w) == (target_h, target_w):
        return image

    return cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_AREA)


def linear_to_srgb(img_lin: NDArray, ev_shift: float = 0.0) -> NDArray:
    """
    Applies exposure compensation and the sRGB transfer function to a linear image.

    :param img_lin: Input linear HDR image (numpy array).
    :param ev_shift: Exposure compensation in stops. Defaults to 0.0.
    :return: sRGB image clipped to [0.0, 1.0].
    """
    # Apply exposure scale in linear space
    scale = 2.0 ** ev_shift
    img_exposed = img_lin * scale

    # Apply sRGB transfer function
    img_srgb = np.where(
        img_exposed <= 0.0031308,
        12.92 * img_exposed,
        1.055 * np.power(np.maximum(img_exposed, 0.0031308), 1 / 2.4) - 0.055
    )

    return np.clip(img_srgb, 0.0, 1.0)


def srgb_to_linear(img_srgb: NDArray, ev_shift: float = 0.0) -> NDArray:
    """
    Inverts the sRGB transfer function and removes exposure compensation.
    Automatically normalizes 8-bit inputs to [0, 1] if values exceed 1.0.
    """
    is_uint8 = img_srgb.dtype == np.uint8
    img_srgb = img_srgb.astype(np.float32)

    # Auto-normalize if unnormalized LDR image
    if is_uint8 or img_srgb.max() > 1.0:
        img_srgb /= 255.0

    # Invert sRGB curve
    img_lin_exposed = np.where(
        img_srgb <= 0.04045,
        img_srgb / 12.92,
        np.power(np.maximum((img_srgb + 0.055) / 1.055, 0.0), 2.4)
    )

    # Undo exposure scale
    scale = 2.0 ** ev_shift

    return img_lin_exposed / scale
