"""Utilities for exporting tonemapped 3DLP generation inputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from threedlp.data.threedlp import ThreeDLPDataset
from threedlp.data.threedlp_generation import (
    build_generated_image_path,
    build_generation_input_path,
    reverse_state,
)


def parse_resize_size(height: int | None, width: int | None) -> tuple[int, int] | None:
    """Return a dataset resize tuple or ``None``."""
    if height is None and width is None:
        return None
    if height is None or width is None:
        raise ValueError("Provide both --resize-height and --resize-width, or neither.")
    if height <= 0 or width <= 0:
        raise ValueError("Resize dimensions must be positive.")
    return (height, width)


def save_float_png(image: np.ndarray, output_path: Path) -> None:
    """Save an RGB float image in [0, 1] as an 8-bit PNG."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image_uint8 = np.clip(image * 255.0, 0.0, 255.0).astype(np.uint8)
    if image_uint8.ndim == 2:
        pil_image = Image.fromarray(image_uint8, mode="L").convert("RGB")
    else:
        pil_image = Image.fromarray(image_uint8, mode="RGB")
    pil_image.save(output_path)


def load_export_dataset(args: argparse.Namespace):
    """Load 3DLP without generated outputs for input export."""
    return ThreeDLPDataset(
        dataset_name=args.dataset_name,
        ev_shift=args.ev_shift,
        correct_off_img=args.correct_off_img,
        ai_models=[],
        size=parse_resize_size(args.resize_height, args.resize_width),
    )


def generation_manifest_row(
    *,
    sample_index: int,
    target_state: str,
    source_state: str,
    sample: dict[str, Any],
    input_path: Path,
    output_root: Path,
    extension: str,
    asset_root: str | None,
) -> dict[str, Any]:
    """Create one manifest row for a generation input PNG."""
    source_path = sample[f"{source_state}_path"]
    target_path = sample[f"{target_state}_path"]
    generated_template = build_generated_image_path(
        target_path,
        "{model_name}",
        target_state,
        extension,
        generated_root_dir=output_root,
        asset_root=asset_root,
    )
    return {
        "sample_index": sample_index,
        "target_state": target_state,
        "source_state": source_state,
        "input_path": input_path.as_posix(),
        "source_tiff_path": source_path,
        "target_tiff_path": target_path,
        "generated_path_template": generated_template.as_posix(),
    }


def export_generation_inputs(args: argparse.Namespace) -> Path:
    """Write generation input PNGs and return the manifest path."""
    if args.limit is not None and args.limit < 0:
        raise ValueError("--limit must be non-negative.")
    dataset = load_export_dataset(args)
    output_root = Path(args.output_root).expanduser()
    manifest_path = Path(args.manifest_path).expanduser() if args.manifest_path else output_root / "manifest.jsonl"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    count = len(dataset) if args.limit is None else min(args.limit, len(dataset))
    asset_root = getattr(dataset, "asset_root", None)
    rows_written = 0

    with manifest_path.open("w", encoding="utf-8") as manifest_file:
        for sample_index in range(count):
            sample = dataset[sample_index]
            for target_state in args.states:
                source_state = reverse_state(target_state)
                source_path = sample[f"{source_state}_path"]
                input_path = build_generation_input_path(
                    source_path,
                    source_state,
                    ".png",
                    output_root_dir=output_root,
                    asset_root=asset_root,
                )
                save_float_png(sample[f"{source_state}_ldr"], input_path)
                row = generation_manifest_row(
                    sample_index=sample_index,
                    target_state=target_state,
                    source_state=source_state,
                    sample=sample,
                    input_path=input_path,
                    output_root=output_root,
                    extension=args.generated_extension,
                    asset_root=asset_root,
                )
                manifest_file.write(json.dumps(row, sort_keys=True))
                manifest_file.write("\n")
                rows_written += 1

    print(f"Exported {rows_written} input PNG(s) to {output_root}")
    print(f"Wrote manifest: {manifest_path}")
    return manifest_path


def build_export_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for ``export_3dlp_generation_inputs``."""
    parser = argparse.ArgumentParser(
        description="Export tonemapped 3DLP PNG inputs for external ON/OFF image generation."
    )
    parser.add_argument("--dataset-name", default="timkuechler/3DLP", help="Hugging Face Hub dataset name.")
    parser.add_argument("--output-root", required=True, help="Root for exported inputs and future generated outputs.")
    parser.add_argument("--manifest-path", default=None, help="Optional manifest path. Defaults to output-root/manifest.jsonl.")
    parser.add_argument("--states", nargs="+", choices=["on", "off"], default=["on", "off"])
    parser.add_argument("--ev-shift", type=float, default=0.0, help="Exposure shift used for tonemapping.")
    parser.add_argument(
        "--correct-off-img",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply the same EXIF brightness correction to OFF TIFFs as ThreeDLPDataset.",
    )
    parser.add_argument("--resize-height", type=int, default=None, help="Optional resize height.")
    parser.add_argument("--resize-width", type=int, default=None, help="Optional resize width.")
    parser.add_argument("--limit", type=int, default=None, help="Optional number of dataset samples to export.")
    parser.add_argument(
        "--generated-extension",
        default=".png",
        choices=[".png", ".jpg", ".jpeg"],
        help="Extension used in generated_path_template.",
    )
    return parser
