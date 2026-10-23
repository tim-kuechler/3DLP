from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter


def compute_evaluation_masks(
    img_gt_8bit: np.ndarray,
    img_ai_8bit: np.ndarray,
    light_gt: np.ndarray,
    clipping_threshold: int = 255,
    min_light_percent: float = 0.01,
    max_light_percent: float | None = None,
    shadow_blur_percent: float = 0.02,
    force_all_valid: bool = True,
    ap_gaussian_filter: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute valid evaluation masks from SDR clipping and GT light influence."""
    ai_unclipped = (img_ai_8bit * 255) < clipping_threshold
    gt_unclipped = (img_gt_8bit * 255) < clipping_threshold

    intensity_gt = np.mean(light_gt, axis=-1)
    blur_sigma = max(intensity_gt.shape) * shadow_blur_percent
    if ap_gaussian_filter:
        light_envelope = gaussian_filter(intensity_gt, sigma=blur_sigma)
    else:
        light_envelope = intensity_gt
    peak_envelope = np.percentile(light_envelope, 99)

    gt_lamp_mask_2d = light_envelope > (min_light_percent * peak_envelope)
    if max_light_percent is not None:
        gt_lamp_mask_2d = gt_lamp_mask_2d & (light_envelope <= (max_light_percent * peak_envelope))

    valid_mask_3d = ai_unclipped & gt_unclipped & gt_lamp_mask_2d[..., np.newaxis]
    valid_mask_2d = valid_mask_3d.all(axis=-1)

    if force_all_valid:
        valid_mask_3d = valid_mask_2d[..., np.newaxis].repeat(axis=-1, repeats=3)

    return valid_mask_2d, valid_mask_3d
