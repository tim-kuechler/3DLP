"""Read-only 3DLP dataset factory."""

from __future__ import annotations

import os
from pathlib import Path

from datasets import Dataset, load_dataset
from datasets import Image as HFImage

from threedlp.data.threedlp_generation import build_generated_image_path
from threedlp.data.threedlp_io import asset_root_from_image_value
from threedlp.data.threedlp_metadata import normalize_ambient_light_type
from threedlp.data.threedlp_transform import (
    build_image_pair,
    should_include_pair,
    transform_batch,
)

__all__ = ["ThreeDLPDataset", "build_generated_image_path", "normalize_ambient_light_type"]


class ThreeDLPDataset:
    """Factory that returns a transformed Hugging Face ``Dataset``.

    The released 3DLP dataset is treated as read-only: it provides real ON/OFF
    HDR TIFFs, while generated model outputs are loaded from an optional
    external ``generated_root_dir``. Each sample contains real linear TIFF
    arrays, LDR references, metadata, and optional generated model images.
    """

    def __new__(
        cls,
        ev_shift: float = 0.0,
        correct_off_img: bool = True,
        ai_models: list[str] | None = None,
        load_only_missing: str | None = None,
        missing_condition: str = "either",
        load_only_existing: bool = False,
        loading_condition: str | None = None,
        dataset_name: str = "timkuechler/3DLP",
        size: tuple[int, int] | None = None,
        generated_root_dir: str | os.PathLike[str] | None = None,
    ) -> Dataset:
        """Build a transformed 3DLP dataset.

        ``dataset_name`` is loaded from the Hugging Face Hub.
        ``generated_root_dir`` points to generated images in the mirrored
        external layout described by ``threedlp_generation``.
        """
        if Path(dataset_name).expanduser().exists():
            raise ValueError(
                "Local dataset paths are not supported. "
                "Upload or access the dataset through the Hugging Face Hub and pass its dataset name."
            )

        ai_models = ai_models or []
        generated_root_dir = str(Path(generated_root_dir).expanduser()) if generated_root_dir is not None else None

        if load_only_existing and load_only_missing:
            raise ValueError("Cannot set both 'load_only_existing' and 'load_only_missing'.")
        if load_only_missing and load_only_missing not in ai_models:
            raise ValueError(f"Model '{load_only_missing}' must be included in 'ai_models'.")

        dataset = load_dataset(dataset_name, split="eval")
        image_feature = HFImage()
        for column_name, feature in dataset.features.items():
            if isinstance(feature, HFImage) and feature.decode:
                dataset = dataset.cast_column(column_name, HFImage(decode=False))

        if not len(dataset):
            raise ValueError(f"Dataset '{dataset_name}' has no samples in its eval split.")

        seed_row = dataset[0]
        asset_root = (
            asset_root_from_image_value(seed_row["image_on"])
            or asset_root_from_image_value(seed_row["image_off"])
        )

        image_pairs = []
        selected_indices = []
        for index in range(len(dataset)):
            row = dataset[index]
            pair = build_image_pair(
                row,
                asset_root=asset_root,
                ai_models=ai_models,
                generated_root_dir=generated_root_dir,
            )
            include_pair = should_include_pair(
                pair,
                ai_models=ai_models,
                load_only_missing=load_only_missing,
                missing_condition=missing_condition,
                load_only_existing=load_only_existing,
                loading_condition=loading_condition,
            )
            if include_pair:
                selected_indices.append(index)
                image_pairs.append(pair)

        dataset = dataset.select(selected_indices) if selected_indices else dataset.select([])

        def transform(batch):
            return transform_batch(
                batch,
                asset_root=asset_root,
                image_feature=image_feature,
                ev_shift=ev_shift,
                correct_off_img=correct_off_img,
                ai_models=ai_models,
                size=size,
                generated_root_dir=generated_root_dir,
            )

        transformed_dataset = dataset.with_transform(
            transform,
            columns=dataset.column_names,
            output_all_columns=False,
        )
        transformed_dataset.image_pairs = image_pairs
        transformed_dataset.ai_models = list(ai_models)
        transformed_dataset.asset_root = asset_root
        transformed_dataset.generated_root_dir = generated_root_dir
        return transformed_dataset
