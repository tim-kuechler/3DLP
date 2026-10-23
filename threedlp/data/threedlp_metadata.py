"""Metadata normalization helpers for 3DLP samples."""

from __future__ import annotations

from typing import Any

from threedlp.data.threedlp_io import path_from_value

RAW_IMAGE_COLUMNS = {
    "image_on",
    "image_off",
    "images_gen_on",
    "images_gen_off",
    "image_gen_on",
    "image_gen_off",
}

AMBIENT_LIGHT_TYPE_ALIASES = {
    "mixed": {"mixedlight", "mixed", "mixed_light", "mixed light"},
    "natural": {"daylight", "natural", "natural_light", "natural light"},
    "synthetic": {
        "artificial_light",
        "artificial light",
        "artificiallight",
        "synthetic",
        "synthetic_light",
        "synthetic light",
    },
}


def normalize_ambient_light_type(value: Any) -> str | None:
    """Normalize 3DLP ambient-light metadata to mixed, natural, or synthetic."""
    if value is None:
        return None

    label = str(value).strip()
    if not label or label.lower() in {"none", "null", "undefined", "unknown"}:
        return None

    normalized = label.lower().replace("-", "_")
    for category, aliases in AMBIENT_LIGHT_TYPE_ALIASES.items():
        if normalized in aliases:
            return category
    return None


def ambient_light_value(row: dict[str, Any]) -> Any:
    """Return the first non-empty ambient-light value from supported metadata fields."""
    for field in ("Ambient_Light", "ambient_light"):
        value = row.get(field)
        if value not in (None, "", {}, []):
            return value
    return None


def metadata_path_key(field_name: str) -> str | None:
    """Return the resolved path key for metadata fields that point at assets."""
    if field_name in {"image_on_file_name", "image_off_file_name"}:
        return None
    if field_name.endswith("_file_name"):
        return f"{field_name.removesuffix('_file_name')}_path"
    if field_name.endswith(("_mask", "_map")):
        return f"{field_name}_path"
    return None


def add_metadata_fields(sample: dict[str, Any], row: dict[str, Any], asset_root: str | None) -> None:
    """Attach non-image metadata and resolved annotation/mask paths to a sample."""
    if asset_root is not None:
        sample["_asset_root"] = asset_root

    sample["ambient_light_type"] = normalize_ambient_light_type(ambient_light_value(row))

    for key, value in row.items():
        if key in RAW_IMAGE_COLUMNS:
            continue

        sample.setdefault(key, value)

        path_key = metadata_path_key(key)
        if path_key is None:
            continue

        resolved_path = path_from_value(value, asset_root)
        if resolved_path is not None:
            sample[path_key] = resolved_path

