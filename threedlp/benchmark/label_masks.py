from __future__ import annotations

import io
import os
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from threedlp.benchmark.constants import EXCLUDE_LABELS


def _infer_label_paths_from_image_paths(sample: dict[str, Any], label_name: str) -> list[str]:
    """Infer 3DLP per-image label masks when metadata omits explicit fields."""
    candidates: list[str] = []
    seen: set[str] = set()

    for image_path in (sample.get("on_path"), sample.get("off_path")):
        if not image_path:
            continue

        image_path_obj = Path(str(image_path)).expanduser()
        stem = image_path_obj.stem
        for suffix in ("_on", "_off"):
            if stem.endswith(suffix):
                stem = stem[: -len(suffix)]
                break

        label_stems = [stem]
        if stem.startswith("images_"):
            label_stems.append(stem.replace("images_", "image_", 1))

        if image_path_obj.parent.name == "images_real":
            label_dir = image_path_obj.parent.parent / "images_labels" / label_name
            for label_stem in label_stems:
                for extension in (".png", ".jpg", ".jpeg"):
                    candidate = str(label_dir / f"{label_stem}_{label_name}{extension}")
                    if candidate not in seen:
                        candidates.append(candidate)
                        seen.add(candidate)

    return candidates


def _candidate_label_values(sample: dict[str, Any], label_name: str) -> list[Any]:
    values = []
    for key in (label_name, f"{label_name}_path", f"{label_name}_file_name"):
        value = sample.get(key)
        if value is None:
            continue
        if isinstance(value, str) and value == "":
            continue
        values.append(value)
    values.extend(_infer_label_paths_from_image_paths(sample, label_name))
    return values


def _candidate_label_paths(sample: dict[str, Any], value: Any) -> list[str]:
    raw_paths: list[str] = []
    if isinstance(value, str):
        raw_paths.append(value)
    elif isinstance(value, dict) and isinstance(value.get("path"), str):
        raw_paths.append(value["path"])

    roots = [sample.get("_asset_root"), sample.get("asset_root")]
    for image_path in (sample.get("on_path"), sample.get("off_path")):
        if not image_path:
            continue
        normalized = str(image_path).replace("\\", "/")
        marker = "/images/"
        if marker in normalized:
            roots.append(normalized.split(marker, maxsplit=1)[0])

    candidates: list[str] = []
    seen: set[str] = set()
    for raw_path in raw_paths:
        path_candidates = [os.path.expanduser(raw_path)]
        if not os.path.isabs(raw_path):
            for root in roots:
                if root:
                    path_candidates.append(os.path.join(os.path.expanduser(str(root)), raw_path))

        for candidate in path_candidates:
            if candidate not in seen:
                candidates.append(candidate)
                seen.add(candidate)
    return candidates


def _read_label_array(sample: dict[str, Any], label_name: str) -> np.ndarray | None:
    try:
        from PIL import Image
    except ImportError as exc:
        raise ImportError("Label-mask metrics need Pillow to read annotation images.") from exc

    for value in _candidate_label_values(sample, label_name):
        if isinstance(value, np.ndarray):
            return value

        if isinstance(value, dict) and value.get("bytes") is not None:
            with Image.open(io.BytesIO(value["bytes"])) as image:
                return np.asarray(image)

        for candidate in _candidate_label_paths(sample, value):
            if os.path.exists(candidate):
                with Image.open(candidate) as image:
                    return np.asarray(image)
    return None


def _label_array_to_mask(label_array: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    mask = np.asarray(label_array)
    if mask.ndim == 3:
        if mask.shape[2] == 4 and np.max(mask[..., 3]) > 0:
            mask = mask[..., 3]
        else:
            mask = np.max(mask[..., :3], axis=2)

    if mask.shape[:2] != target_shape:
        mask = cv2.resize(mask, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_NEAREST)

    mask = mask.astype(np.float32)
    if mask.size == 0:
        return np.zeros(target_shape, dtype=bool)
    if np.nanmax(mask) > 1.0:
        mask = mask / 255.0
    return mask > 0.5


def load_label_masks(
    sample: dict[str, Any],
    target_shape: tuple[int, int],
    label_names: list[str] | None = None,
) -> dict[str, np.ndarray | None]:
    label_names = list(EXCLUDE_LABELS) if label_names is None else label_names
    masks: dict[str, np.ndarray | None] = {}
    for label_name in label_names:
        label_array = _read_label_array(sample, label_name)
        masks[label_name] = None if label_array is None else _label_array_to_mask(label_array, target_shape)
    return masks


def build_default_exclude_mask(
    base_valid_mask: np.ndarray,
    label_masks: dict[str, np.ndarray | None],
    exclude_labels: list[str] | None = None,
) -> np.ndarray:
    exclude_labels = EXCLUDE_LABELS if exclude_labels is None else exclude_labels
    excluded_mask = np.zeros_like(base_valid_mask, dtype=bool)
    for label in exclude_labels:
        label_mask = label_masks.get(label)
        if label_mask is not None:
            excluded_mask |= label_mask
    return excluded_mask


def apply_default_exclude_mask(
    base_valid_mask: np.ndarray,
    label_masks: dict[str, np.ndarray | None],
    exclude_labels: list[str] | None = None,
) -> np.ndarray:
    excluded_mask = build_default_exclude_mask(base_valid_mask, label_masks, exclude_labels)
    return base_valid_mask & ~excluded_mask


def build_benchmark_mask(
    base_valid_mask: np.ndarray,
    label_masks: dict[str, np.ndarray | None],
    exclude_labels: list[str] | None = None,
) -> np.ndarray:
    return apply_default_exclude_mask(base_valid_mask, label_masks, exclude_labels)
