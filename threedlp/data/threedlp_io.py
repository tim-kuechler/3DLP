"""IO helpers for the 3DLP dataset loader."""

from __future__ import annotations

import io
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import tifffile
from datasets import Image as HFImage

from threedlp.utils.image_utils import resize_image as _resize_image


def resize_image(image: np.ndarray | None, target_size: tuple[int, int] | None) -> np.ndarray | None:
    """Resize an image to ``(height, width)`` without upscaling."""
    return _resize_image(image, target_size, allow_upscale=False)


def parse_exposure(value: str | None) -> float:
    """Parse EXIF-style numeric values such as ``"1/60"`` into a float."""
    if not value:
        return 1.0
    if "/" in value:
        numerator, denominator = value.split("/", maxsplit=1)
        return float(numerator) / float(denominator)
    return float(value)


def repair_missing_view_path(path: str | os.PathLike[str]) -> str | None:
    """Repair missing ``view_*`` asset paths by searching sibling view folders."""
    path_obj = Path(path).expanduser()
    parts = path_obj.parts
    view_index = next((index for index, part in enumerate(parts) if part.startswith("view_")), None)
    if view_index is None:
        return None

    parent_dir = Path(*parts[:view_index])
    suffix_parts = parts[view_index + 1 :]
    if not parent_dir.is_dir() or not suffix_parts:
        return None

    matches: list[Path] = []
    for sibling in sorted(parent_dir.glob("view_*")):
        candidate = sibling.joinpath(*suffix_parts)
        if candidate.exists():
            matches.append(candidate)

    if len(matches) == 1:
        return str(matches[0])
    return None


def path_from_value(image_value: dict[str, Any] | str | None, asset_root: str | None) -> str | None:
    """Resolve a dataset image reference into a filesystem path when possible."""
    if isinstance(image_value, str):
        path = image_value
    elif isinstance(image_value, dict):
        path = image_value.get("path")
    else:
        return None

    if path is None:
        return None

    if os.path.exists(path):
        return path

    repaired_path = repair_missing_view_path(path)
    if repaired_path is not None:
        return repaired_path

    if asset_root is not None:
        candidate = os.path.join(asset_root, path)
        if os.path.exists(candidate):
            return candidate
        repaired_candidate = repair_missing_view_path(candidate)
        if repaired_candidate is not None:
            return repaired_candidate

    return path


def asset_root_from_image_value(image_value: dict[str, Any] | str) -> str | None:
    """Infer the dataset asset root from an absolute image path."""
    path = path_from_value(image_value, asset_root=None)
    if path is None or not os.path.isabs(path):
        return None

    normalized = path.replace("\\", "/")
    marker = "/images/"
    if marker in normalized:
        return normalized.split(marker, maxsplit=1)[0]
    return os.path.dirname(path)


def open_tiff_source(image_value: dict[str, Any] | str, asset_root: str | None) -> str | io.BytesIO:
    """Return a TIFF source accepted by ``tifffile``."""
    if isinstance(image_value, dict) and image_value.get("bytes") is not None:
        return io.BytesIO(image_value["bytes"])

    path = path_from_value(image_value, asset_root)
    if path is not None:
        return path

    raise ValueError("Image feature has neither bytes nor path.")


def extract_exif_dict(image_value: dict[str, Any] | str, asset_root: str | None) -> dict[str, Any]:
    """Read the embedded EXIF payload stored in the TIFF ImageDescription tag."""
    with tifffile.TiffFile(open_tiff_source(image_value, asset_root)) as tif:
        exif_json = tif.pages[0].tags["ImageDescription"].value
    exif_dict = json.loads(exif_json)
    return json.loads(exif_dict["ImageDescription"])


def read_tiff(image_value: dict[str, Any] | str, asset_root: str | None) -> np.ndarray:
    """Load a TIFF into float32 linear-light space."""
    return tifffile.imread(open_tiff_source(image_value, asset_root)).astype(np.float32)


def decode_image(
    image_value: dict[str, Any] | str | None,
    image_feature: HFImage,
    asset_root: str | None,
) -> tuple[np.ndarray | None, str | None]:
    """Decode an optional HF image reference into an RGB float array and source path."""
    if image_value is None:
        return None, None

    path = path_from_value(image_value, asset_root)
    image_bytes = image_value.get("bytes") if isinstance(image_value, dict) else None
    decoded = image_feature.decode_example({"path": path, "bytes": image_bytes})
    return np.asarray(decoded.convert("RGB"), dtype=np.float32) / 255.0, path
