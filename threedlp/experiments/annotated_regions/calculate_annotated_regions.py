"""SIE/LFE80 over annotated 3DLP regions.

The robust affine standardization is fitted on the full benchmark-valid image
mask, after clipping/light-influence checks and default label exclusions. Scores
are then computed inside each annotation mask and summarized with pixel-weighted
averages across images.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np

from threedlp.benchmark.affine_metrics import (
    MetricAccumulator,
    compute_ratio_maps_for_task,
    get_grad_mag,
    robust_standardize,
    valid_channel_masks,
)
from threedlp.benchmark.constants import (
    EXCLUDE_LABELS,
    LOW_FREQ_FALLOFF_KEEP_PERCENTILE,
)
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

cv2.setNumThreads(0)


REGION_LABELS = [
    "metallic_surfaces",
    "mirror_surfaces",
    "transparent_surfaces",
    "other_lamps_highlights",
    "other_lamps_shadows",
    "isolated_cast_shadows",
    "isolated_specular_highlights",
]
FULL_IMAGE_REGION = "full_image"
OUTPUT_METRICS = ("SIE", "LFE80")


def compute_sie_on_region(
    R_AI: np.ndarray,
    R_L: np.ndarray,
    L_R: np.ndarray,
    standardization_mask_2d: np.ndarray,
    scoring_mask_2d: np.ndarray,
    noise_floor: float = 0.005,
    min_pixels: int = 20,
) -> tuple[float, float, int]:
    accumulator = MetricAccumulator()
    for channel in range(3):
        masks = valid_channel_masks(
            L_R=L_R,
            alignment_mask_2d=standardization_mask_2d,
            averaging_mask_2d=scoring_mask_2d,
            channel=channel,
            noise_floor=noise_floor,
            min_pixels=min_pixels,
        )
        if masks is None:
            continue
        standardization_mask, scoring_mask = masks

        r_ai_flat = R_AI[..., channel][standardization_mask]
        r_l_flat = R_L[..., channel][standardization_mask]
        err_full = np.abs(robust_standardize(r_ai_flat) - robust_standardize(r_l_flat))
        err = err_full[scoring_mask[standardization_mask]]
        accumulator.add(err)

    return accumulator.as_tuple()


def compute_lfe_on_region(
    R_AI: np.ndarray,
    R_L: np.ndarray,
    L_R: np.ndarray,
    standardization_mask_2d: np.ndarray,
    scoring_mask_2d: np.ndarray,
    keep_percentile: float = LOW_FREQ_FALLOFF_KEEP_PERCENTILE,
    noise_floor: float = 0.005,
    min_pixels: int = 20,
) -> tuple[float, float, int]:
    if keep_percentile <= 0 or keep_percentile > 100:
        raise ValueError("keep_percentile must be in the interval (0, 100].")

    accumulator = MetricAccumulator()
    for channel in range(3):
        masks = valid_channel_masks(
            L_R=L_R,
            alignment_mask_2d=standardization_mask_2d,
            averaging_mask_2d=scoring_mask_2d,
            channel=channel,
            noise_floor=noise_floor,
            min_pixels=min_pixels,
        )
        if masks is None:
            continue
        standardization_mask, scoring_mask = masks

        g_ai_flat = get_grad_mag(R_AI[..., channel])[standardization_mask]
        g_l_flat = get_grad_mag(R_L[..., channel])[standardization_mask]
        ai_edge_cutoff = np.percentile(g_ai_flat, keep_percentile)
        gt_edge_cutoff = np.percentile(g_l_flat, keep_percentile)
        low_frequency_keep = (g_ai_flat <= ai_edge_cutoff) & (g_l_flat <= gt_edge_cutoff)
        scoring_keep = low_frequency_keep & scoring_mask[standardization_mask]
        if np.sum(low_frequency_keep) < min_pixels or np.sum(scoring_keep) < min_pixels:
            continue

        # Keep LFE's low-frequency definition global: choose and standardize the
        # low-gradient subset from the full valid image, then average by region.
        ai_z = robust_standardize(g_ai_flat[low_frequency_keep])
        gt_z = robust_standardize(g_l_flat[low_frequency_keep])
        err_full = np.abs(ai_z - gt_z)
        err = err_full[scoring_keep[low_frequency_keep]]
        accumulator.add(err)

    return accumulator.as_tuple()


def compute_region_scores(
    I_AI: np.ndarray,
    I_off: np.ndarray,
    I_ON: np.ndarray,
    standardization_mask_2d: np.ndarray,
    scoring_mask_2d: np.ndarray,
    is_turn_off: bool,
) -> dict[str, float | int]:
    if np.sum(standardization_mask_2d) < 20 or np.sum(scoring_mask_2d) < 20:
        return {
            "SIE": np.nan,
            "SIE_error_sum": 0.0,
            "SIE_error_count": 0,
            "LFE80": np.nan,
            "LFE80_error_sum": 0.0,
            "LFE80_error_count": 0,
        }

    L_R, R_AI, R_L = compute_ratio_maps_for_task(
        I_AI=I_AI,
        I_off=I_off,
        I_ON=I_ON,
        is_turn_off=is_turn_off,
    )
    sie, sie_error_sum, sie_error_count = compute_sie_on_region(
        R_AI=R_AI,
        R_L=R_L,
        L_R=L_R,
        standardization_mask_2d=standardization_mask_2d,
        scoring_mask_2d=scoring_mask_2d,
    )
    lfe80, lfe80_error_sum, lfe80_error_count = compute_lfe_on_region(
        R_AI=R_AI,
        R_L=R_L,
        L_R=L_R,
        standardization_mask_2d=standardization_mask_2d,
        scoring_mask_2d=scoring_mask_2d,
    )
    return {
        "SIE": sie,
        "SIE_error_sum": sie_error_sum,
        "SIE_error_count": sie_error_count,
        "LFE80": lfe80,
        "LFE80_error_sum": lfe80_error_sum,
        "LFE80_error_count": lfe80_error_count,
    }


def process_single_sample(
    idx: int,
    dataset: Any,
    ai_models: list[str],
    global_target_shape: tuple[int, int],
    task: str,
    region_labels: list[str],
    exclude_labels: list[str],
) -> dict[str, Any]:
    try:
        sample = dataset[idx]
        rows = []

        references = prepare_reference_images(sample, global_target_shape)
        state = task_state(sample, task)

        label_names = sorted({label for label in region_labels + exclude_labels if label != FULL_IMAGE_REGION})
        label_masks = load_label_masks(sample, global_target_shape, label_names=label_names)

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
            standardization_mask_2d = apply_default_exclude_mask(
                valid_mask_2d,
                label_masks,
                exclude_labels=exclude_labels,
            )

            scores = compute_region_scores(
                I_AI=model_images.image_linear,
                I_off=references.off_linear,
                I_ON=references.on_linear,
                standardization_mask_2d=standardization_mask_2d,
                scoring_mask_2d=standardization_mask_2d,
                is_turn_off=state.is_turn_off,
            )
            rows.append(
                {
                    "task": task,
                    "model": model,
                    "region": FULL_IMAGE_REGION,
                    "region_mask_available": True,
                    "SIE": scores["SIE"],
                    "SIE_error_sum": scores["SIE_error_sum"],
                    "SIE_error_count": scores["SIE_error_count"],
                    "LFE80": scores["LFE80"],
                    "LFE80_error_sum": scores["LFE80_error_sum"],
                    "LFE80_error_count": scores["LFE80_error_count"],
                }
            )

            for region_label in region_labels:
                if region_label == FULL_IMAGE_REGION:
                    continue

                region_mask = label_masks.get(region_label)
                region_available = region_mask is not None
                if region_mask is None:
                    scoring_mask_2d = np.zeros_like(standardization_mask_2d, dtype=bool)
                else:
                    scoring_mask_2d = standardization_mask_2d & region_mask

                scores = compute_region_scores(
                    I_AI=model_images.image_linear,
                    I_off=references.off_linear,
                    I_ON=references.on_linear,
                    standardization_mask_2d=standardization_mask_2d,
                    scoring_mask_2d=scoring_mask_2d,
                    is_turn_off=state.is_turn_off,
                )
                rows.append(
                    {
                        "task": task,
                        "model": model,
                        "region": region_label,
                        "region_mask_available": region_available,
                        "SIE": scores["SIE"],
                        "SIE_error_sum": scores["SIE_error_sum"],
                        "SIE_error_count": scores["SIE_error_count"],
                        "LFE80": scores["LFE80"],
                        "LFE80_error_sum": scores["LFE80_error_sum"],
                        "LFE80_error_count": scores["LFE80_error_count"],
                    }
                )

        return {"idx": idx, "rows": rows}
    except Exception as exc:
        return {"idx": idx, "error": str(exc), "rows": []}


def _is_finite(value: Any) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def _metric_weight(row: dict[str, Any], metric: str) -> int:
    weight_key = f"{metric}_error_count"
    if _is_finite(row.get(weight_key)):
        weight = int(float(row[weight_key]))
        if weight > 0:
            return weight

    return 0


def _metric_error_sum(row: dict[str, Any], metric: str) -> float:
    sum_key = f"{metric}_error_sum"
    if _is_finite(row.get(sum_key)):
        return float(row[sum_key])

    value = row.get(metric)
    if not _is_finite(value):
        return 0.0
    return float(value) * _metric_weight(row, metric)


def summarize_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in rows:
        key = (str(row["task"]), str(row["model"]), str(row["region"]))
        payload = grouped.setdefault(
            key,
            {
                "candidate_images": 0,
                "mask_available_images": 0,
                "scored_images": 0,
                "SIE_values": [],
                "SIE_weighted_sum": 0.0,
                "SIE_weight": 0,
                "LFE80_values": [],
                "LFE80_weighted_sum": 0.0,
                "LFE80_weight": 0,
            },
        )
        payload["candidate_images"] = int(payload["candidate_images"]) + 1
        if row.get("region_mask_available"):
            payload["mask_available_images"] = int(payload["mask_available_images"]) + 1

        valid_for_any_metric = False
        for metric in OUTPUT_METRICS:
            value = row.get(metric)
            if _is_finite(value):
                metric_values = payload[f"{metric}_values"]
                if not isinstance(metric_values, list):
                    raise TypeError(f"Expected list accumulator for {metric}, got {type(metric_values).__name__}")
                metric_value = float(value)
                metric_weight = _metric_weight(row, metric)
                if metric_weight > 0:
                    metric_values.append(metric_value)
                    payload[f"{metric}_weighted_sum"] = float(payload[f"{metric}_weighted_sum"]) + _metric_error_sum(
                        row, metric
                    )
                    payload[f"{metric}_weight"] = int(payload[f"{metric}_weight"]) + metric_weight
                    valid_for_any_metric = True
        if valid_for_any_metric:
            payload["scored_images"] = int(payload["scored_images"]) + 1

    summary_rows = []
    for (task, model, region), payload in sorted(grouped.items()):
        summary_row: dict[str, Any] = {
            "task": task,
            "model": model,
            "region": region,
            "candidate_images": payload["candidate_images"],
            "mask_available_images": payload["mask_available_images"],
            "scored_images": payload["scored_images"],
        }
        for metric in OUTPUT_METRICS:
            values = np.asarray(payload[f"{metric}_values"], dtype=float)
            weight = int(payload[f"{metric}_weight"])
            summary_row[f"{metric}_mean"] = (
                float(payload[f"{metric}_weighted_sum"]) / weight if weight > 0 else np.nan
            )
            summary_row[f"{metric}_median"] = float(np.median(values)) if values.size else np.nan
            summary_row[f"{metric}_std"] = float(np.std(values, ddof=1)) if values.size > 1 else np.nan
            summary_row[f"{metric}_valid_count"] = int(values.size)
            summary_row[f"{metric}_pixel_count"] = weight
        summary_rows.append(summary_row)
    return summary_rows


def write_summary_csv(summary_rows: list[dict[str, Any]], output_path: Path) -> None:
    fieldnames = [
        "task",
        "model",
        "region",
        "candidate_images",
        "mask_available_images",
        "scored_images",
        "SIE_mean",
        "SIE_median",
        "SIE_std",
        "SIE_valid_count",
        "SIE_pixel_count",
        "LFE80_mean",
        "LFE80_median",
        "LFE80_std",
        "LFE80_valid_count",
        "LFE80_pixel_count",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)


def evaluate_task(
    dataset: Any,
    ai_models: list[str],
    task: str,
    target_resolution: tuple[int, int] | None,
    max_cores: int,
    region_labels: list[str],
    exclude_labels: list[str],
) -> tuple[list[dict[str, Any]], int]:
    print(f"\n{'=' * 50}")
    print(f"Starting annotated-region evaluation for {task.upper()}")
    print(f"{'=' * 50}")

    active_models = active_models_for_task(dataset, ai_models, task)
    if not active_models:
        print(f"No valid AI model data found for {task}. Skipping.")
        return [], 0

    global_target_shape = get_global_target_resolution(dataset, active_models, target_resolution, task=task)
    sample_indices = list(range(len(dataset)))

    rows: list[dict[str, Any]] = []
    errors = []
    for result in iter_indexed_jobs(
        indices=sample_indices,
        processor=process_single_sample,
        processor_kwargs={
            "dataset": dataset,
            "ai_models": active_models,
            "global_target_shape": global_target_shape,
            "task": task,
            "region_labels": region_labels,
            "exclude_labels": exclude_labels,
        },
        max_workers=max_cores,
        desc=f"Annotated regions ({task})",
    ):
        if "error" in result:
            errors.append(result)
            print(f"\nError processing sample {result['idx']}: {result['error']}")
            continue
        rows.extend(result["rows"])

    print(f"[{task}] Processed {len(sample_indices)} samples with {len(errors)} errors.")
    return rows, len(errors)


def parse_resolution(values: list[int] | None) -> tuple[int, int] | None:
    if values is None:
        return None
    if len(values) != 2:
        raise ValueError("Resolution must be two integers: HEIGHT WIDTH.")
    if any(value <= 0 for value in values):
        raise ValueError("Resolution dimensions must be positive.")
    return values[0], values[1]


def _parse_labels(values: Iterable[str] | None, default: list[str]) -> list[str]:
    labels = list(values) if values is not None else list(default)
    labels = [label for label in labels if label]
    if not labels:
        raise ValueError("At least one label is required.")
    return labels


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate SIE and LFE80 on annotated 3DLP regions with global valid-image standardization."
    )
    parser.add_argument("--generated-root-dir", default=None, help="External root containing generated 3DLP images.")
    parser.add_argument("--dataset-name", default="timkuechler/3DLP", help="Hugging Face Hub dataset name.")
    parser.add_argument("--models", nargs="+", required=True, help="Generated model name(s) to evaluate.")
    parser.add_argument("--tasks", nargs="+", choices=["turn_on", "turn_off"], default=["turn_on", "turn_off"])
    parser.add_argument("--region-labels", nargs="+", default=REGION_LABELS)
    parser.add_argument("--exclude-labels", nargs="+", default=EXCLUDE_LABELS)
    parser.add_argument("--target-resolution", nargs=2, type=int, metavar=("HEIGHT", "WIDTH"))
    parser.add_argument(
        "--output-csv",
        default=str(Path(__file__).resolve().parent / "results" / "AnnotatedRegion_SIE_LFE80.csv"),
    )
    parser.add_argument("--max-cores", type=int, default=32)
    parser.add_argument("--correct-off-img", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    target_resolution = parse_resolution(args.target_resolution)
    region_labels = _parse_labels(args.region_labels, REGION_LABELS)
    exclude_labels = _parse_labels(args.exclude_labels, EXCLUDE_LABELS)
    max_workers = max(1, args.max_cores)
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
    print(f"Dataset loaded. Total potential pairs: {len(dataset)}")
    print(f"Region labels: {', '.join(region_labels)}")
    print(f"Exclude labels: {', '.join(exclude_labels)}")

    all_rows: list[dict[str, Any]] = []
    total_errors = 0
    for task in args.tasks:
        rows, error_count = evaluate_task(
            dataset=dataset,
            ai_models=evaluation_models,
            task=task,
            target_resolution=target_resolution,
            max_cores=max_workers,
            region_labels=region_labels,
            exclude_labels=exclude_labels,
        )
        all_rows.extend(rows)
        total_errors += error_count

    output_csv = Path(args.output_csv)
    write_summary_csv(summarize_rows(all_rows), output_csv)
    print(f"\nSaved annotated-region SIE/LFE80 summary to {output_csv}")
    if total_errors:
        print(f"Skipped {total_errors} samples due to processing errors.")

    print("\nAnnotated-region SIE/LFE80 evaluation complete.")


if __name__ == "__main__":
    main()
