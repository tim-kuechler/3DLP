"""File layout helpers for external 3DLP generation assets.

The released 3DLP dataset contains only real HDR TIFF assets. Generated model
outputs are expected to live outside that dataset, using a mirrored directory
layout so they can be mounted, regenerated, or deleted independently.
"""

from __future__ import annotations

import os
from pathlib import Path

GENERATED_IMAGE_DIR_NAME = "images_gen"
GENERATION_INPUT_DIR_NAME = "images_input"
SUPPORTED_GENERATED_EXTENSIONS = (".png", ".jpg", ".jpeg")


def _validate_state(state: str) -> None:
    if state not in {"on", "off"}:
        raise ValueError(f"state must be 'on' or 'off', got: {state}")


def reverse_state(state: str) -> str:
    """Return the opposite 3DLP lighting state."""
    _validate_state(state)
    return "off" if state == "on" else "on"


def base_stem_from_state_path(path: str | os.PathLike[str]) -> str:
    """Return a sample stem without the trailing ON/OFF state suffix."""
    stem_parts = Path(path).stem.rsplit("_", maxsplit=1)
    return stem_parts[0] if len(stem_parts) == 2 and stem_parts[1] in {"on", "off"} else Path(path).stem


def _relative_scene_dir(
    reference_path: str | os.PathLike[str],
    asset_root: str | os.PathLike[str] | None = None,
) -> Path:
    """Return the path below the split root that contains ``images_real``."""
    path = Path(reference_path).expanduser()

    if asset_root is not None:
        try:
            path = path.relative_to(Path(asset_root).expanduser())
        except ValueError:
            pass

    parts = path.parts
    if "images" in parts:
        images_index = parts.index("images")
        path = Path(*parts[images_index:])

    if path.parent.name == "images_real":
        return path.parent.parent
    return path.parent


def build_generated_image_path(
    reference_path: str | os.PathLike[str],
    model_name: str,
    state: str,
    extension: str,
    generated_root_dir: str | os.PathLike[str] | None = None,
    asset_root: str | os.PathLike[str] | None = None,
) -> Path:
    """Map a GT image path to the conventional generated-image location.

    Without ``generated_root_dir`` this preserves the historical in-dataset
    layout: ``.../images_real/<sample>_off.tiff`` maps to
    ``.../images_gen/<model>/<sample>_on.png``.

    With ``generated_root_dir`` the path mirrors the dataset layout below that
    external root. This keeps generated model outputs separate from the
    publication dataset while preserving stable sample-relative paths.
    """
    _validate_state(state)
    extension = extension if extension.startswith(".") else f".{extension}"
    path = Path(reference_path).expanduser()
    filename = f"{base_stem_from_state_path(path)}_{state}{extension}"

    if generated_root_dir is None:
        if path.parent.name != "images_real":
            raise ValueError(f"Expected an images_real path, got: {path}")
        return path.parent.parent / GENERATED_IMAGE_DIR_NAME / model_name / filename

    return (
        Path(generated_root_dir).expanduser()
        / _relative_scene_dir(path, asset_root)
        / GENERATED_IMAGE_DIR_NAME
        / model_name
        / filename
    )


def build_generation_input_path(
    reference_path: str | os.PathLike[str],
    state: str,
    extension: str,
    output_root_dir: str | os.PathLike[str],
    asset_root: str | os.PathLike[str] | None = None,
) -> Path:
    """Map a real TIFF path to the mirrored tonemapped generation-input PNG."""
    _validate_state(state)
    extension = extension if extension.startswith(".") else f".{extension}"
    path = Path(reference_path).expanduser()
    filename = f"{base_stem_from_state_path(path)}_{state}{extension}"
    return (
        Path(output_root_dir).expanduser()
        / _relative_scene_dir(path, asset_root)
        / GENERATION_INPUT_DIR_NAME
        / filename
    )


def candidate_generated_image_paths(
    reference_path: str | os.PathLike[str],
    model_name: str,
    state: str,
    generated_root_dir: str | os.PathLike[str] | None,
    asset_root: str | os.PathLike[str] | None = None,
    extensions: tuple[str, ...] = SUPPORTED_GENERATED_EXTENSIONS,
) -> list[Path]:
    """Return possible generated output paths in priority order.

    The first location is the canonical mirrored ``images_gen`` path. The two
    fallback patterns support outputs saved next to exported PNG inputs, which
    is convenient for simple model runners that operate in-place.
    """
    if generated_root_dir is None:
        return [
            build_generated_image_path(reference_path, model_name, state, extension)
            for extension in extensions
        ]

    paths = [
        build_generated_image_path(
            reference_path,
            model_name,
            state,
            extension,
            generated_root_dir=generated_root_dir,
            asset_root=asset_root,
        )
        for extension in extensions
    ]

    # Also accept outputs saved beside the exported input PNGs. This makes it
    # easy to hand the generated root to a model runner without metadata edits.
    source_state = reverse_state(state)
    input_path = build_generation_input_path(
        reference_path,
        source_state,
        ".png",
        output_root_dir=generated_root_dir,
        asset_root=asset_root,
    )
    input_dir = input_path.parent
    filename_base = f"{base_stem_from_state_path(reference_path)}_{state}"
    paths.extend(input_dir / model_name / f"{filename_base}{extension}" for extension in extensions)
    paths.extend(input_dir / f"{filename_base}_{model_name}{extension}" for extension in extensions)
    return paths
