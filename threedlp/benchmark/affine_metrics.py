from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import sobel

from threedlp.benchmark.constants import BASE_METRICS, LOW_FREQ_FALLOFF_KEEP_PERCENTILE


def robust_standardize(arr: np.ndarray, epsilon: float = 1e-6) -> np.ndarray:
    med = np.median(arr)
    mad = np.median(np.abs(arr - med))
    return (arr - med) / (mad + epsilon)


@dataclass
class MetricAccumulator:
    error_sum: float = 0.0
    error_count: int = 0

    def add(self, values: np.ndarray, *, drop_nonfinite: bool = True) -> None:
        values = np.asarray(values)
        if drop_nonfinite:
            values = values[np.isfinite(values)]
        self.error_sum += np.sum(values)
        self.error_count += int(values.size)

    @property
    def mean(self) -> float:
        return self.error_sum / self.error_count if self.error_count > 0 else np.nan

    def as_tuple(self) -> tuple[float, float, int]:
        return self.mean, self.error_sum, self.error_count


def get_grad_mag(img_2d: np.ndarray) -> np.ndarray:
    grad_x = sobel(img_2d, axis=0)
    grad_y = sobel(img_2d, axis=1)
    return np.sqrt(grad_x**2 + grad_y**2)


def metric_keys() -> list[str]:
    return [f"affine_{metric}" for metric in BASE_METRICS]


def compute_ratio_maps_for_task(
    I_AI: np.ndarray,
    I_off: np.ndarray,
    I_ON: np.ndarray,
    is_turn_off: bool,
    epsilon: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return target light, AI ratio, and target ratio for the selected task.

    Turn-on follows ``I_AI_on = C * L_R + K * I_R_off`` and compares
    ``I_AI_on / I_R_off`` to ``I_R_on / I_R_off``. Turn-off uses the analogous
    signed hypothesis ``I_AI_off = K * I_R_on - C * L_R`` and compares
    ``I_AI_off / I_R_on`` to ``I_R_off / I_R_on``.
    """
    I_off_safe = np.clip(I_off, 0, None)
    I_ON_safe = np.clip(I_ON, 0, None)
    I_AI_safe = np.asarray(I_AI, dtype=np.float32)

    L_R = np.clip(I_ON_safe - I_off_safe, 0, None)
    if is_turn_off:
        denominator = I_ON_safe + epsilon
        return L_R, I_AI_safe / denominator, I_off_safe / denominator

    denominator = I_off_safe + epsilon
    return L_R, I_AI_safe / denominator, I_ON_safe / denominator


def valid_channel_masks(
    L_R: np.ndarray,
    alignment_mask_2d: np.ndarray,
    averaging_mask_2d: np.ndarray,
    channel: int,
    noise_floor: float,
    min_pixels: int = 20,
) -> tuple[np.ndarray, np.ndarray] | None:
    signal_mask = L_R[..., channel] > noise_floor
    alignment_mask = alignment_mask_2d & signal_mask
    averaging_mask = averaging_mask_2d & alignment_mask
    if np.sum(alignment_mask) < min_pixels or np.sum(averaging_mask) < min_pixels:
        return None
    return alignment_mask, averaging_mask


def compute_intensity_error(
    R_AI: np.ndarray,
    R_L: np.ndarray,
    L_R: np.ndarray,
    alignment_mask_2d: np.ndarray,
    averaging_mask_2d: np.ndarray,
    noise_floor: float = 0.005,
) -> float:
    accumulator = MetricAccumulator()
    for c in range(3):
        masks = valid_channel_masks(L_R, alignment_mask_2d, averaging_mask_2d, c, noise_floor)
        if masks is None:
            continue
        alignment_mask, averaging_mask = masks

        r_ai_flat = R_AI[..., c][alignment_mask]
        r_l_flat = R_L[..., c][alignment_mask]
        err_full = np.abs(robust_standardize(r_ai_flat) - robust_standardize(r_l_flat))

        keep = averaging_mask[alignment_mask]
        accumulator.add(err_full[keep], drop_nonfinite=False)

    return accumulator.mean


def compute_percentile_falloff(
    R_AI: np.ndarray,
    R_L: np.ndarray,
    L_R: np.ndarray,
    alignment_mask_2d: np.ndarray,
    averaging_mask_2d: np.ndarray,
    keep_percentile: float = 80.0,
    noise_floor: float = 0.005,
) -> float:
    if keep_percentile <= 0 or keep_percentile > 100:
        raise ValueError("keep_percentile must be in the interval (0, 100].")

    accumulator = MetricAccumulator()
    for c in range(3):
        masks = valid_channel_masks(L_R, alignment_mask_2d, averaging_mask_2d, c, noise_floor)
        if masks is None:
            continue
        alignment_mask, averaging_mask = masks

        g_ai_full = get_grad_mag(R_AI[..., c])
        g_l_full = get_grad_mag(R_L[..., c])
        g_ai_flat = g_ai_full[alignment_mask]
        g_l_flat = g_l_full[alignment_mask]

        ai_edge_cutoff = np.percentile(g_ai_flat, keep_percentile)
        gt_edge_cutoff = np.percentile(g_l_flat, keep_percentile)
        edge_keep = (g_ai_flat <= ai_edge_cutoff) & (g_l_flat <= gt_edge_cutoff)
        keep = edge_keep & averaging_mask[alignment_mask]
        if np.sum(keep) < 20:
            continue

        err_full = np.abs(robust_standardize(g_ai_flat[edge_keep]) - robust_standardize(g_l_flat[edge_keep]))
        accumulator.add(err_full[keep[edge_keep]])

    return accumulator.mean


def compute_metric_scores(
    I_AI: np.ndarray,
    I_off: np.ndarray,
    I_ON: np.ndarray,
    alignment_mask_2d: np.ndarray | None,
    averaging_mask_2d: np.ndarray | None,
    is_turn_off: bool,
) -> dict[str, float]:
    keys = metric_keys()
    if alignment_mask_2d is None or averaging_mask_2d is None or np.sum(averaging_mask_2d) < 20:
        return {metric_key: np.nan for metric_key in keys}

    L_R, R_AI, R_L = compute_ratio_maps_for_task(
        I_AI=I_AI,
        I_off=I_off,
        I_ON=I_ON,
        is_turn_off=is_turn_off,
    )

    scores: dict[str, float] = {}
    scores["affine_intensity_error"] = compute_intensity_error(
        R_AI=R_AI,
        R_L=R_L,
        L_R=L_R,
        alignment_mask_2d=alignment_mask_2d,
        averaging_mask_2d=averaging_mask_2d,
    )
    scores["affine_low_freq_falloff"] = compute_percentile_falloff(
        R_AI=R_AI,
        R_L=R_L,
        L_R=L_R,
        alignment_mask_2d=alignment_mask_2d,
        averaging_mask_2d=averaging_mask_2d,
        keep_percentile=LOW_FREQ_FALLOFF_KEEP_PERCENTILE,
    )

    return {metric_key: scores.get(metric_key, np.nan) for metric_key in keys}
