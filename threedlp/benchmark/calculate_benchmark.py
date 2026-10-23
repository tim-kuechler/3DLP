from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from threedlp.benchmark.affine_metrics import (
    compute_metric_scores,
    metric_keys,
)
from threedlp.benchmark.constants import (
    LOW_FREQ_FALLOFF_KEEP_PERCENTILE,
)
from threedlp.benchmark.dataset_utils import (
    active_models_for_task as _active_models_for_task,
)
from threedlp.benchmark.dataset_utils import (
    get_global_target_resolution,
)
from threedlp.benchmark.evaluation_masks import compute_evaluation_masks
from threedlp.benchmark.evaluation_utils import (
    iter_indexed_jobs,
    prepare_model_images,
    prepare_reference_images,
    task_state,
)
from threedlp.benchmark.label_masks import (
    build_benchmark_mask,
    load_label_masks,
)
from threedlp.data.threedlp import ThreeDLPDataset

cv2.setNumThreads(0)


def process_single_sample(
    idx: int,
    dataset: Any,
    ai_models: list[str],
    global_target_shape: tuple[int, int],
    task: str,
) -> dict[str, Any]:
    try:
        sample = dataset[idx]
        original_indices = getattr(dataset, "original_indices", None)
        original_idx = original_indices[idx] if original_indices is not None else idx
        result_dict = {"idx": idx, "original_idx": original_idx, "models": {}}

        references = prepare_reference_images(sample, global_target_shape)
        state = task_state(sample, task)
        label_masks = load_label_masks(sample, global_target_shape)

        for model in ai_models:
            model_images = prepare_model_images(sample, model, state, global_target_shape)
            if model_images is None:
                continue

            valid_mask_2d, _ = compute_evaluation_masks(
                img_gt_8bit=references.on_8bit,
                img_ai_8bit=model_images.image_8bit,
                light_gt=references.light_linear,
                clipping_threshold=255,
                min_light_percent=0.01,
                max_light_percent=None,
                ap_gaussian_filter=True,
            )

            benchmark_mask = build_benchmark_mask(valid_mask_2d, label_masks)
            result_dict["models"][model] = compute_metric_scores(
                I_AI=model_images.image_linear,
                I_off=references.off_linear,
                I_ON=references.on_linear,
                alignment_mask_2d=benchmark_mask,
                averaging_mask_2d=benchmark_mask,
                is_turn_off=state.is_turn_off,
            )

        return result_dict
    except Exception as exc:
        return {"idx": idx, "error": str(exc), "models": {}}


def active_models_for_task(dataset: Any, ai_models: list[str], task: str) -> list[str]:
    return _active_models_for_task(
        dataset=dataset,
        ai_models=ai_models,
        task=task,
        missing_message="No '{state_key}' results found.",
    )


def initialize_results(
    task: str,
    active_models: list[str],
    global_target_shape: tuple[int, int],
) -> dict[str, Any]:
    keys = metric_keys()
    return {
        "experiment": "affine_ratio_benchmark",
        "task": task,
        "ratio_definition": {
            "turn_on": "I_AI_on / I_R_off vs I_R_on / I_R_off",
            "turn_off": "I_AI_off / I_R_on vs I_R_off / I_R_on",
        },
        "metric_family": "affine robust standardization for both tasks; invariant to per-channel affine scale/shift",
        "low_freq_falloff": f"edge-filtered unblurred gradient score with keep_percentile={LOW_FREQ_FALLOFF_KEEP_PERCENTILE:g}",
        "target_resolution": [int(global_target_shape[0]), int(global_target_shape[1])],
        "evaluation_mask": "full benchmark-valid image with windows excluded",
        "metrics": keys,
        "models": {
            model: {
                "data_coverage": 0,
                "metric_valid_counts": {metric_key: 0 for metric_key in keys},
                **{metric_key: [] for metric_key in keys},
            }
            for model in active_models
        },
    }


def _is_valid_metric_value(value: Any) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def append_sample_result(
    model_results: dict[str, Any],
    sample_result: dict[str, Any],
    keys: list[str],
) -> None:
    for model, metrics in sample_result.get("models", {}).items():
        if model not in model_results["models"]:
            continue
        destination = model_results["models"][model]
        valid_values = []
        for metric_key in keys:
            value = metrics.get(metric_key, np.nan)
            destination[metric_key].append(value)
            if _is_valid_metric_value(value):
                destination["metric_valid_counts"][metric_key] += 1
                valid_values.append(value)
        if valid_values:
            destination["data_coverage"] += 1


def finalize_results(model_results: dict[str, Any]) -> None:
    for model_payload in model_results["models"].values():
        for metric_key in model_results["metrics"]:
            metric_array = np.asarray(model_payload[metric_key], dtype=float)
            valid_metrics = metric_array[np.isfinite(metric_array)]
            if valid_metrics.size == 0:
                model_payload[f"{metric_key}_dataset_avg"] = np.nan
                model_payload[f"{metric_key}_dataset_std"] = np.nan
            else:
                model_payload[f"{metric_key}_dataset_avg"] = float(np.mean(valid_metrics))
                model_payload[f"{metric_key}_dataset_std"] = (
                    float(np.std(valid_metrics, ddof=1)) if valid_metrics.size > 1 else np.nan
                )


def results_to_rows(model_results: dict[str, Any]) -> list[dict[str, Any]]:
    keys = model_results["metrics"]
    rows = []
    for model, payload in model_results["models"].items():
        row = {
            "task": model_results["task"],
            "Model": model,
            "Data_Coverage": payload["data_coverage"],
        }
        for key in keys:
            row[f"{key}_dataset_avg"] = payload.get(f"{key}_dataset_avg", np.nan)
            row[f"{key}_dataset_std"] = payload.get(f"{key}_dataset_std", np.nan)
            row[f"{key}_valid_count"] = payload["metric_valid_counts"][key]
        rows.append(row)
    return rows


def write_results_csv(rows: list[dict[str, Any]], output_path: Path, metric_keys_: list[str]) -> None:
    fieldnames = ["task", "Model", "Data_Coverage"]
    for key in metric_keys_:
        fieldnames.extend([f"{key}_dataset_avg", f"{key}_dataset_std", f"{key}_valid_count"])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def evaluate_task(
    dataset: Any,
    ai_models: list[str],
    task: str,
    target_resolution: tuple[int, int] | None,
    max_cores: int,
) -> list[dict[str, Any]]:
    print(f"\n{'=' * 50}")
    print(f"Starting Affine-Ratio Evaluation for Task: {task.upper()}")
    print(f"{'=' * 50}")

    active_models = active_models_for_task(dataset, ai_models, task)
    if not active_models:
        print(f"No valid AI model data found for {task}. Skipping.")
        return []

    keys = metric_keys()
    global_target_shape = get_global_target_resolution(dataset, active_models, target_resolution, task)
    sample_indices = list(range(len(dataset)))

    model_results = initialize_results(
        task=task,
        active_models=active_models,
        global_target_shape=global_target_shape,
    )

    if max_cores <= 1:
        print("Running synchronously because max_cores <= 1...")
    else:
        print(f"Spinning up {max_cores} parallel workers...")

    for result in iter_indexed_jobs(
        indices=sample_indices,
        processor=process_single_sample,
        processor_kwargs={
            "dataset": dataset,
            "ai_models": active_models,
            "global_target_shape": global_target_shape,
            "task": task,
        },
        max_workers=max_cores,
        desc=f"Evaluating {task}",
    ):
        if "error" in result:
            print(f"\nError processing sample {result['idx']}: {result['error']}")
            continue
        append_sample_result(model_results, result, keys)

    finalize_results(model_results)
    return results_to_rows(model_results)


def parse_resolution(values: list[int] | None) -> tuple[int, int] | None:
    if values is None:
        return None
    if len(values) != 2:
        raise ValueError("Resolution must be two integers: HEIGHT WIDTH.")
    if any(value <= 0 for value in values):
        raise ValueError("Resolution dimensions must be positive.")
    return values[0], values[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate affine ratio light transport metrics.")
    parser.add_argument("--generated-root-dir", default=None, help="External root containing generated 3DLP images.")
    parser.add_argument("--dataset-name", default="timkuechler/3DLP", help="Hugging Face Hub dataset name.")
    parser.add_argument("--models", nargs="+", required=True, help="Generated model name(s) to evaluate.")
    parser.add_argument("--tasks", nargs="+", choices=["turn_on", "turn_off"], default=["turn_on", "turn_off"])
    parser.add_argument("--target-resolution", nargs=2, type=int, metavar=("HEIGHT", "WIDTH"))
    parser.add_argument(
        "--output-csv",
        default=str(Path(__file__).resolve().parent / "AffineRatio_MAE_Final_Ratios.csv"),
    )
    parser.add_argument("--max-cores", type=int, default=32)
    parser.add_argument("--correct-off-img", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    target_resolution = parse_resolution(args.target_resolution)
    max_workers = min(args.max_cores, max(1, (os.cpu_count() or 1) - 1))
    evaluation_models = args.models

    print(f"Initializing Dataset from Hub dataset: {args.dataset_name}")
    dataset = ThreeDLPDataset(
        dataset_name=args.dataset_name,
        ev_shift=0.0,
        ai_models=evaluation_models,
        correct_off_img=args.correct_off_img,
        load_only_existing=False,
        generated_root_dir=args.generated_root_dir,
    )
    print(f"Dataset fully loaded. Total potential pairs: {len(dataset)}")

    output_rows: list[dict[str, Any]] = []
    for task in args.tasks:
        output_rows.extend(
            evaluate_task(
                dataset=dataset,
                ai_models=evaluation_models,
                task=task,
                target_resolution=target_resolution,
                max_cores=max_workers,
            )
        )

    output_csv = Path(args.output_csv)
    write_results_csv(output_rows, output_csv, metric_keys())
    print(f"\nSaved benchmark results to {output_csv}")
    if not any(row["Data_Coverage"] > 0 for row in output_rows):
        raise RuntimeError("No valid benchmark scores were produced. Check generated paths and sample errors.")
    print("\nAll affine-ratio tasks complete!")


if __name__ == "__main__":
    main()
