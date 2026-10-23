from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from threedlp.benchmark.affine_metrics import (
    compute_metric_scores,
    metric_keys,
)
from threedlp.benchmark.constants import EXCLUDE_LABELS
from threedlp.benchmark.dataset_utils import (
    active_models_for_task,
    get_global_target_resolution,
)
from threedlp.benchmark.evaluation_masks import compute_evaluation_masks
from threedlp.benchmark.evaluation_utils import (
    iter_indexed_jobs,
    prepare_model_images,
    prepare_reference_images,
    task_state,
)
from threedlp.benchmark.label_masks import apply_default_exclude_mask, load_label_masks
from threedlp.data.threedlp import ThreeDLPDataset

BEST_IMAGE_KEEP_PERCENTILE = 80.0


def format_band_label(min_intensity: float, max_intensity: float) -> str:
    lower = int(min_intensity * 100)
    upper = int(max_intensity * 100)
    if lower == upper and lower > 0:
        lower -= 1
    return f"{lower}-{upper}%"


def generate_equidistant_light_bands(
    num_bands: int,
    min_intensity: float = 0.01,
    max_intensity: float = 1.0,
) -> list[tuple[float, float]]:
    if num_bands <= 0:
        raise ValueError("num_bands must be positive.")
    if not 0 < min_intensity < max_intensity <= 1:
        raise ValueError("Intensities must satisfy 0 < min_intensity < max_intensity <= 1.")
    r_max = 1.0 / np.sqrt(min_intensity)
    r_min = 1.0 / np.sqrt(max_intensity)
    r_steps = np.linspace(r_max, r_min, num_bands + 1)
    i_steps = 1.0 / (r_steps**2)
    return [(round(i_steps[i], 4), round(i_steps[i + 1], 4)) for i in range(num_bands)]


def process_intensity_band_sample(
    idx: int,
    dataset: Any,
    ai_models: list[str],
    global_target_shape: tuple[int, int],
    task: str,
    bands: list[tuple[float, float]],
    apply_band_mask_after_metric: bool,
) -> dict[str, Any]:
    try:
        sample = dataset[idx]
        result_dict = {"idx": idx, "models": {}}

        references = prepare_reference_images(sample, global_target_shape)
        state = task_state(sample, task)
        label_masks = load_label_masks(sample, global_target_shape, label_names=EXCLUDE_LABELS)

        for model in ai_models:
            model_images = prepare_model_images(sample, model, state, global_target_shape)
            if model_images is None:
                continue

            result_dict["models"][model] = {}

            full_valid_mask_2d = None
            if apply_band_mask_after_metric:
                full_valid_mask_2d, _ = compute_evaluation_masks(
                    img_gt_8bit=references.on_8bit,
                    img_ai_8bit=model_images.image_8bit,
                    light_gt=references.light_linear,
                    clipping_threshold=255,
                    min_light_percent=0.01,
                    max_light_percent=None,
                    ap_gaussian_filter=True,
                )
                full_valid_mask_2d = apply_default_exclude_mask(full_valid_mask_2d, label_masks)

            for min_pct, max_pct in bands:
                band_key = format_band_label(min_pct, max_pct)
                band_mask_2d, _ = compute_evaluation_masks(
                    img_gt_8bit=references.on_8bit,
                    img_ai_8bit=model_images.image_8bit,
                    light_gt=references.light_linear,
                    min_light_percent=min_pct,
                    max_light_percent=max_pct,
                    force_all_valid=True,
                    ap_gaussian_filter=False,
                )
                band_mask_2d = apply_default_exclude_mask(band_mask_2d, label_masks)

                if apply_band_mask_after_metric:
                    alignment_mask_2d = full_valid_mask_2d
                    averaging_mask_2d = band_mask_2d
                else:
                    alignment_mask_2d = band_mask_2d
                    averaging_mask_2d = band_mask_2d

                result_dict["models"][model][band_key] = compute_metric_scores(
                    I_AI=model_images.image_linear,
                    I_off=references.off_linear,
                    I_ON=references.on_linear,
                    alignment_mask_2d=alignment_mask_2d,
                    averaging_mask_2d=averaging_mask_2d,
                    is_turn_off=state.is_turn_off,
                )

        return result_dict
    except Exception as exc:
        return {"idx": idx, "error": str(exc), "models": {}}


def keep_lowest_error_images(values: np.ndarray, keep_percentile: float) -> tuple[np.ndarray, int]:
    if keep_percentile <= 0 or keep_percentile > 100:
        raise ValueError("keep_percentile must be in the interval (0, 100].")

    valid_values = np.asarray(values, dtype=float)
    valid_values = valid_values[np.isfinite(valid_values)]
    original_count = int(valid_values.size)
    if original_count == 0 or keep_percentile >= 100:
        return valid_values, original_count

    keep_count = max(1, int(np.ceil(original_count * keep_percentile / 100.0)))
    return np.sort(valid_values)[:keep_count], original_count


def run_intensity_bands(
    dataset: Any,
    ai_models: list[str],
    output_csv: str | Path,
    target_resolution: tuple[int, int] | None,
    max_cores: int,
    task: str,
    bands: list[tuple[float, float]],
    apply_band_mask_after_metric: bool,
) -> None:
    bands = sorted(bands, key=lambda band: band[1], reverse=True)
    band_labels = [format_band_label(mn, mx) for mn, mx in bands]
    active_models = active_models_for_task(dataset, ai_models, task)
    if not active_models:
        raise ValueError(f"No generated model images found for {task}.")
    global_target_shape = get_global_target_resolution(dataset, active_models, target_resolution, task=task)

    metrics_list = metric_keys()
    agg_data = {
        model: {band_label: {metric: [] for metric in metrics_list} for band_label in band_labels}
        for model in active_models
    }

    for result in iter_indexed_jobs(
        indices=range(len(dataset)),
        processor=process_intensity_band_sample,
        processor_kwargs={
            "dataset": dataset,
            "ai_models": active_models,
            "global_target_shape": global_target_shape,
            "task": task,
            "bands": bands,
            "apply_band_mask_after_metric": apply_band_mask_after_metric,
        },
        max_workers=max_cores,
        desc=f"Intensity bands ({task})",
    ):
        if "error" in result:
            print(f"Skipping sample {result.get('idx')}: {result['error']}")
            continue

        for model, band_data in result["models"].items():
            for band_key, metrics in band_data.items():
                for metric_name in metrics_list:
                    agg_data[model][band_key][metric_name].append(metrics[metric_name])

    summary_rows = []
    for metric in metrics_list:
        for model in active_models:
            for band_label in band_labels:
                arr = np.asarray(agg_data[model][band_label][metric], dtype=float)
                kept_arr, original_valid_count = keep_lowest_error_images(arr, BEST_IMAGE_KEEP_PERCENTILE)
                summary_rows.append(
                    {
                        "Task": task,
                        "Model": model,
                        "Metric": metric,
                        "Band": band_label,
                        "Mean_Error": np.nanmean(kept_arr) if kept_arr.size > 0 else np.nan,
                        "Std_Error": np.nanstd(kept_arr) if kept_arr.size > 0 else np.nan,
                        "Valid_Images": int(kept_arr.size),
                        "Candidate_Valid_Images": original_valid_count,
                        "Best_Image_Keep_Percentile": BEST_IMAGE_KEEP_PERCENTILE,
                    }
                )

    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summary_rows).to_csv(output_csv, index=False)
    print(f"Saved data to {output_csv}")


def parse_target_resolution(value: str | None) -> tuple[int, int] | None:
    if value is None:
        return None
    height, width = value.lower().split("x", maxsplit=1)
    resolution = (int(height), int(width))
    if any(dimension <= 0 for dimension in resolution):
        raise ValueError("Resolution dimensions must be positive.")
    return resolution


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute final affine SIE/LFE metrics over light intensity bands.")
    parser.add_argument("--generated-root-dir", default=None, help="External root containing generated 3DLP images.")
    parser.add_argument("--dataset-name", default="timkuechler/3DLP", help="Hugging Face Hub dataset name.")
    parser.add_argument("--task", choices=["turn_on", "turn_off"], default="turn_on")
    parser.add_argument("--models", nargs="+", required=True, help="Generated model name(s) to evaluate.")
    parser.add_argument("--num-bands", type=int, default=6)
    parser.add_argument("--min-intensity", type=float, default=0.01)
    parser.add_argument("--max-intensity", type=float, default=1.0)
    parser.add_argument("--target-resolution", type=parse_target_resolution, default=None, help="Format: HEIGHTxWIDTH")
    parser.add_argument("--max-cores", type=int, default=32)
    parser.add_argument("--output-csv", default="threedlp/experiments/intensity_bands/intensity_bands_on_sie_lfe_p80.csv")
    parser.add_argument("--apply-band-mask-after-metric", action="store_true")
    args = parser.parse_args()
    evaluation_models = args.models

    bands = generate_equidistant_light_bands(
        num_bands=args.num_bands,
        min_intensity=args.min_intensity,
        max_intensity=args.max_intensity,
    )
    print(f"Using bands: {bands}")

    dataset = ThreeDLPDataset(
        dataset_name=args.dataset_name,
        ev_shift=0.0,
        ai_models=evaluation_models,
        correct_off_img=False,
        generated_root_dir=args.generated_root_dir,
    )

    run_intensity_bands(
        dataset=dataset,
        ai_models=evaluation_models,
        output_csv=args.output_csv,
        target_resolution=args.target_resolution,
        max_cores=args.max_cores,
        task=args.task,
        bands=bands,
        apply_band_mask_after_metric=args.apply_band_mask_after_metric,
    )


if __name__ == "__main__":
    main()
