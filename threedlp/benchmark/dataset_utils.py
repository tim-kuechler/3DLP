from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import numpy as np
import tifffile
from PIL import Image

from threedlp.utils.image_utils import resize_image as _resize_image


def get_model_image(sample: dict[str, Any], model: str, state_key: str) -> tuple[np.ndarray | None, str | None]:
    return sample.get(f"{model}_{state_key}"), sample.get(f"{model}_{state_key}_path")


def resize_image(img: np.ndarray | None, target_shape: tuple[int, int]) -> np.ndarray | None:
    """Resize benchmark images to a common evaluation shape."""
    return _resize_image(img, target_shape, allow_upscale=True)


def active_models_for_task(
    dataset: Any,
    ai_models: list[str],
    task: str,
    missing_message: str = "no '{state_key}' results found.",
) -> list[str]:
    state_key = "on" if task == "turn_on" else "off"
    active_models = []

    for model in ai_models:
        has_data = any(
            pair.get("ai_paths", {}).get(model, {}).get(state_key) is not None
            for pair in dataset.image_pairs
        )
        if has_data:
            active_models.append(model)
        else:
            print(f"  -> Skipping '{model}': {missing_message.format(state_key=state_key)}")

    return active_models


def get_global_target_resolution(
    dataset: Any,
    ai_models: Iterable[str],
    explicit_res: tuple[int, int] | None,
    task: str,
) -> tuple[int, int]:
    if explicit_res is not None:
        if len(explicit_res) != 2 or any(dimension <= 0 for dimension in explicit_res):
            raise ValueError("Target resolution must be two positive integers: HEIGHT WIDTH.")
        print(f"Using explicit target resolution: {explicit_res}")
        return explicit_res

    print(f"Scanning dataset to determine global minimum resolution for {task}...")
    min_h, min_w = float("inf"), float("inf")
    models = list(ai_models)
    state_key = "on" if task == "turn_on" else "off"

    # Read headers from every pair so later, smaller outputs are included without
    # decoding the entire HDR dataset a second time.
    seen_paths: set[str] = set()
    for pair in dataset.image_pairs:
        paths = [pair.get("gt_on"), pair.get("gt_off")]
        paths.extend(pair.get("ai_paths", {}).get(model, {}).get(state_key) for model in models)
        for path in paths:
            if path is None or path in seen_paths:
                continue
            seen_paths.add(path)
            if Path(path).suffix.lower() in {".tif", ".tiff"}:
                with tifffile.TiffFile(path) as image:
                    h, w = image.pages[0].shape[:2]
            else:
                with Image.open(path) as image:
                    w, h = image.size
            min_h, min_w = min(min_h, h), min(min_w, w)

    if min_h == float("inf"):
        raise ValueError("Could not find any valid images to determine the target resolution.")

    global_res = (int(min_h), int(min_w))
    print(f"Global target resolution set to: {global_res}")
    return global_res
