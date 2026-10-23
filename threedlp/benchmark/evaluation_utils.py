from __future__ import annotations

import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Iterator

import numpy as np
from tqdm import tqdm

from threedlp.benchmark.dataset_utils import get_model_image, resize_image
from threedlp.utils.image_utils import srgb_to_linear


@dataclass(frozen=True)
class TaskState:
    state_key: str
    ev_shift: float
    is_turn_off: bool


@dataclass(frozen=True)
class ReferenceImages:
    off_8bit: np.ndarray
    on_8bit: np.ndarray
    off_linear: np.ndarray
    on_linear: np.ndarray
    light_linear: np.ndarray


@dataclass(frozen=True)
class ModelImages:
    image_8bit: np.ndarray
    image_linear: np.ndarray
    path: str | None


_WORKER_CONTEXT: dict[str, Any] = {}
_WORKER_PROCESSOR: Callable[..., dict[str, Any]] | None = None


def state_key_for_task(task: str) -> str:
    if task == "turn_on":
        return "on"
    if task == "turn_off":
        return "off"
    raise ValueError(f"Unknown task: {task}")


def task_state(sample: dict[str, Any], task: str) -> TaskState:
    state_key = state_key_for_task(task)
    return TaskState(
        state_key=state_key,
        ev_shift=float(sample["auto_ev_shift_off"]) if state_key == "on" else 0.0,
        is_turn_off=state_key == "off",
    )


def prepare_reference_images(sample: dict[str, Any], target_shape: tuple[int, int]) -> ReferenceImages:
    off_8bit = resize_image(sample["off_ldr"], target_shape)
    on_8bit = resize_image(sample["on_ldr"], target_shape)
    off_linear = resize_image(sample["off_tiff"], target_shape)
    on_linear = resize_image(sample["on_tiff"], target_shape)
    return ReferenceImages(
        off_8bit=off_8bit,
        on_8bit=on_8bit,
        off_linear=off_linear,
        on_linear=on_linear,
        light_linear=np.clip(on_linear - off_linear, 0, None),
    )


def prepare_model_images(
    sample: dict[str, Any],
    model: str,
    state: TaskState,
    target_shape: tuple[int, int],
) -> ModelImages | None:
    image_raw, image_path = get_model_image(sample, model, state.state_key)
    if image_raw is None:
        return None

    image_8bit = resize_image(image_raw, target_shape)
    return ModelImages(
        image_8bit=image_8bit,
        image_linear=srgb_to_linear(image_8bit, ev_shift=state.ev_shift).astype(np.float32),
        path=image_path,
    )


def _set_worker_context(
    processor: Callable[..., dict[str, Any]] | None = None,
    **kwargs: Any,
) -> None:
    global _WORKER_PROCESSOR
    _WORKER_PROCESSOR = processor
    _WORKER_CONTEXT.clear()
    _WORKER_CONTEXT.update(kwargs)


def _process_from_worker_context(idx: int) -> dict[str, Any]:
    if _WORKER_PROCESSOR is None:
        return {"idx": idx, "error": "Worker processor context is not initialized."}
    if "dataset" not in _WORKER_CONTEXT:
        return {"idx": idx, "error": "Worker dataset context is not initialized."}
    return _WORKER_PROCESSOR(idx=idx, **_WORKER_CONTEXT)


def iter_indexed_jobs(
    *,
    indices: Iterable[int],
    processor: Callable[..., dict[str, Any]],
    processor_kwargs: dict[str, Any],
    max_workers: int,
    desc: str,
) -> Iterator[dict[str, Any]]:
    sample_indices = list(indices)
    if max_workers <= 1:
        for idx in tqdm(sample_indices, desc=desc):
            yield processor(idx=idx, **processor_kwargs)
        return

    _set_worker_context(processor, **processor_kwargs)
    try:
        try:
            multiprocessing_context = mp.get_context("fork")
        except ValueError:
            multiprocessing_context = None

        if multiprocessing_context is None:
            for idx in tqdm(sample_indices, desc=desc):
                yield _process_from_worker_context(idx)
        else:
            with ProcessPoolExecutor(max_workers=max_workers, mp_context=multiprocessing_context) as executor:
                futures = {
                    executor.submit(_process_from_worker_context, idx): idx
                    for idx in sample_indices
                }
                for future in tqdm(as_completed(futures), total=len(sample_indices), desc=desc):
                    yield future.result()
    finally:
        _set_worker_context()
