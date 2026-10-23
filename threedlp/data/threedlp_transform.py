"""Sample transform helpers for :class:`threedlp.data.threedlp.ThreeDLPDataset`."""

from __future__ import annotations

import os
from typing import Any

import numpy as np
from datasets import Image as HFImage

from threedlp.data.threedlp_generation import (
    SUPPORTED_GENERATED_EXTENSIONS,
    candidate_generated_image_paths,
)
from threedlp.data.threedlp_io import (
    decode_image,
    extract_exif_dict,
    parse_exposure,
    path_from_value,
    read_tiff,
    resize_image,
)
from threedlp.data.threedlp_metadata import add_metadata_fields
from threedlp.utils.image_utils import linear_to_srgb, srgb_to_linear


def model_name_candidates(model_name: str) -> list[str]:
    """Return exact and common delimiter-variant aliases for a model name."""
    candidates = [model_name]
    for candidate in (model_name.replace("-", "_"), model_name.replace("_", "-")):
        if candidate not in candidates:
            candidates.append(candidate)
    return candidates


def resolve_generated_image_value(
    row: dict[str, Any],
    asset_root: str | None,
    model_name: str,
    state: str,
    generated_root_dir: str | os.PathLike[str] | None = None,
) -> dict[str, Any] | str | None:
    """Find a generated image from metadata, an external root, or the legacy path."""
    metadata_images = row.get(f"images_gen_{state}") or {}
    for candidate_model_name in model_name_candidates(model_name):
        image_value = metadata_images.get(candidate_model_name)
        if image_value is not None:
            return image_value

    reference_path = path_from_value(row[f"image_{state}"], asset_root)
    if reference_path is None:
        return None

    for candidate_model_name in model_name_candidates(model_name):
        try:
            candidates = candidate_generated_image_paths(
                reference_path,
                candidate_model_name,
                state,
                generated_root_dir=generated_root_dir,
                asset_root=asset_root,
                extensions=SUPPORTED_GENERATED_EXTENSIONS,
            )
        except ValueError:
            candidates = []
        for candidate in candidates:
            if candidate.exists():
                return str(candidate)

    if generated_root_dir is not None:
        for candidate_model_name in model_name_candidates(model_name):
            try:
                candidates = candidate_generated_image_paths(
                    reference_path,
                    candidate_model_name,
                    state,
                    generated_root_dir=None,
                    extensions=SUPPORTED_GENERATED_EXTENSIONS,
                )
            except ValueError:
                candidates = []
            for candidate in candidates:
                if candidate.exists():
                    return str(candidate)

    return None


def build_image_pair(
    row: dict[str, Any],
    *,
    asset_root: str | None,
    ai_models: list[str],
    generated_root_dir: str | os.PathLike[str] | None,
) -> dict[str, Any]:
    """Build the lightweight path summary stored on ``dataset.image_pairs``."""
    pair: dict[str, Any] = {
        "gt_on": path_from_value(row["image_on"], asset_root),
        "gt_off": path_from_value(row["image_off"], asset_root),
        "ai_paths": {},
    }

    for model_name in ai_models:
        resolved_on_value = resolve_generated_image_value(row, asset_root, model_name, "on", generated_root_dir)
        resolved_off_value = resolve_generated_image_value(row, asset_root, model_name, "off", generated_root_dir)
        pair["ai_paths"][model_name] = {
            "on": path_from_value(resolved_on_value, asset_root),
            "off": path_from_value(resolved_off_value, asset_root),
        }

    return pair


def should_include_pair(
    pair: dict[str, Any],
    *,
    ai_models: list[str],
    load_only_missing: str | None,
    missing_condition: str,
    load_only_existing: bool,
    loading_condition: str | None,
) -> bool:
    """Apply availability filters without loading image pixels."""
    include_pair = True

    if load_only_missing:
        model_paths = pair["ai_paths"][load_only_missing]
        is_on_missing = model_paths["on"] is None
        is_off_missing = model_paths["off"] is None

        if missing_condition == "either":
            include_pair = is_on_missing or is_off_missing
        elif missing_condition == "both":
            include_pair = is_on_missing and is_off_missing
        elif missing_condition == "on":
            include_pair = is_on_missing
        elif missing_condition == "off":
            include_pair = is_off_missing
        else:
            raise ValueError("missing_condition must be 'either', 'both', 'on', or 'off'.")

    if include_pair and load_only_existing:
        for model_name in ai_models:
            model_paths = pair["ai_paths"][model_name]
            has_on = model_paths["on"] is not None
            has_off = model_paths["off"] is not None

            if loading_condition == "both":
                condition_met = has_on and has_off
            elif loading_condition == "either":
                condition_met = has_on or has_off
            elif loading_condition == "on":
                condition_met = has_on
            elif loading_condition == "off":
                condition_met = has_off
            else:
                raise ValueError("loading_condition must be 'either', 'both', 'on', or 'off'.")

            if not condition_met:
                return False

    return include_pair


def rows_from_transform_batch(batch: dict[str, Any]) -> tuple[list[dict[str, Any]], bool]:
    """Normalize a transform input into a list of row dicts."""
    if not batch:
        return [], False

    first_value = next(iter(batch.values()))
    if isinstance(first_value, list):
        rows = []
        for index in range(len(first_value)):
            rows.append({key: value[index] for key, value in batch.items()})
        return rows, True

    return [batch], False


def rows_to_transform_batch(rows: list[dict[str, Any]], batched: bool) -> dict[str, Any]:
    """Convert transformed rows back into the shape expected by ``with_transform``."""
    if not rows:
        return {}
    if not batched:
        return rows[0]

    keys = dict.fromkeys(key for row in rows for key in row)
    return {key: [row.get(key) for row in rows] for key in keys}


def transform_row(
    row: dict[str, Any],
    *,
    asset_root: str | None,
    image_feature: HFImage,
    ev_shift: float,
    correct_off_img: bool,
    ai_models: list[str],
    size: tuple[int, int] | None,
    generated_root_dir: str | os.PathLike[str] | None,
) -> dict[str, Any]:
    """Decode TIFFs, apply exposure normalization, and attach generated images."""
    on_tiff = read_tiff(row["image_on"], asset_root)
    off_tiff = read_tiff(row["image_off"], asset_root)
    on_path = path_from_value(row["image_on"], asset_root)
    off_path = path_from_value(row["image_off"], asset_root)

    exif_on = extract_exif_dict(row["image_on"], asset_root)
    exif_off = extract_exif_dict(row["image_off"], asset_root)

    base_exposure = parse_exposure(exif_on["EXIF ExposureTime"])
    brightness_on = parse_exposure(exif_on["EXIF BrightnessValue"])
    brightness_off = parse_exposure(exif_off["EXIF BrightnessValue"])
    auto_ev_shift = brightness_on - brightness_off

    on_tiff = on_tiff * base_exposure
    off_tiff = off_tiff * base_exposure

    if correct_off_img:
        off_tiff = off_tiff * (2.0 ** auto_ev_shift)

    on_tiff = resize_image(on_tiff, size)
    off_tiff = resize_image(off_tiff, size)

    sample = {
        "on_tiff": on_tiff,
        "off_tiff": off_tiff,
        "on_ldr": linear_to_srgb(on_tiff, ev_shift).astype(np.float32),
        "off_ldr": linear_to_srgb(off_tiff, ev_shift).astype(np.float32),
        "on_path": on_path,
        "off_path": off_path,
        "auto_ev_shift_off": auto_ev_shift,
    }
    add_metadata_fields(sample, row, asset_root)

    for model_name in ai_models:
        image_on_value = resolve_generated_image_value(row, asset_root, model_name, "on", generated_root_dir)
        image_off_value = resolve_generated_image_value(row, asset_root, model_name, "off", generated_root_dir)
        model_on_srgb, model_on_path = decode_image(image_on_value, image_feature, asset_root)
        model_off_srgb, model_off_path = decode_image(image_off_value, image_feature, asset_root)

        sample[f"{model_name}_on"] = model_on_srgb
        sample[f"{model_name}_off"] = model_off_srgb
        sample[f"{model_name}_on_linear"] = (
            srgb_to_linear(model_on_srgb, ev_shift=ev_shift).astype(np.float32)
            if model_on_srgb is not None
            else None
        )
        sample[f"{model_name}_off_linear"] = (
            srgb_to_linear(model_off_srgb, ev_shift=ev_shift).astype(np.float32)
            if model_off_srgb is not None
            else None
        )
        sample[f"{model_name}_on_path"] = model_on_path
        sample[f"{model_name}_off_path"] = model_off_path

    return sample


def transform_batch(
    batch: dict[str, Any],
    *,
    asset_root: str | None,
    image_feature: HFImage,
    ev_shift: float,
    correct_off_img: bool,
    ai_models: list[str],
    size: tuple[int, int] | None,
    generated_root_dir: str | os.PathLike[str] | None,
) -> dict[str, Any]:
    """Apply the 3DLP sample transform to a single row or a batched transform payload."""
    rows, batched = rows_from_transform_batch(batch)
    transformed_rows = [
        transform_row(
            row,
            asset_root=asset_root,
            image_feature=image_feature,
            ev_shift=ev_shift,
            correct_off_img=correct_off_img,
            ai_models=ai_models,
            size=size,
            generated_root_dir=generated_root_dir,
        )
        for row in rows
    ]
    return rows_to_transform_batch(transformed_rows, batched)
