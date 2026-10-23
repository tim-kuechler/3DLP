# 3DLP Internals

This document is a tour of the repository for people who want to understand or
modify the benchmark code. If you only want to run a model and compute scores,
start with the workflow guide: [workflow.md](workflow.md).

## Repository Layout

```text
threedlp/
  data/
    threedlp.py                         # ThreeDLPDataset factory
    export_3dlp_generation_inputs.py    # CLI entry point for exporting inputs
    threedlp_export.py                  # export implementation and manifest rows
    threedlp_generation.py              # input/output path conventions
    threedlp_transform.py               # sample decoding and generated-image loading
    threedlp_io.py                      # image, TIFF, EXIF, resize helpers
    threedlp_metadata.py                # metadata normalization
  benchmark/
    calculate_benchmark.py              # main 3DLP benchmark CLI
    affine_metrics.py                    # SIE/LFE metric implementation
    evaluation_utils.py                 # task state, image prep, multiprocessing
    evaluation_masks.py                 # clipping and light-influence masks
    label_masks.py                      # annotation-mask loading and exclusions
    dataset_utils.py                    # model availability and resolution helpers
    constants.py                        # shared metric and label constants
  experiments/
    annotated_regions/
      calculate_annotated_regions.py    # per-annotation SIE/LFE summaries
    intensity_bands/
      calculate_intensity_bands.py      # SIE/LFE by light-intensity band
  utils/
    image_utils.py                      # sRGB and linear-light conversion helpers
```

## Main Data Flow

The usual benchmark path is:

1. `threedlp.data.export_3dlp_generation_inputs` parses CLI arguments.
2. `threedlp.data.threedlp_export` loads the dataset through
   `ThreeDLPDataset`, saves tonemapped PNG inputs, and writes
   `manifest.jsonl`.
3. An external model runner reads the manifest and saves generated images.
4. `threedlp.benchmark.calculate_benchmark` loads the same dataset with
   `ai_models=[...]` and `generated_root_dir=...`.
5. The benchmark prepares reference and generated images at a common
   resolution, builds evaluation masks, computes metrics, and writes a summary
   CSV.

## Dataset Loader

`ThreeDLPDataset` lives in `threedlp/data/threedlp.py`. It is a factory around
the Hugging Face `datasets` loader. It always loads the `eval` split from
`dataset_name` and treats the released dataset as read-only.

Important constructor arguments:

- `dataset_name`: Hugging Face dataset name, defaulting to
  `timkuechler/3DLP`.
- `ai_models`: model names whose generated images should be attached to each
  sample.
- `generated_root_dir`: local root where external generated images are stored.
- `correct_off_img`: whether to apply the OFF-image EXIF brightness correction.
  Enabled by default in the dataset loader and input exporter; benchmark
  evaluation disables it by default.
- `size`: optional `(height, width)` resize applied while loading samples.
- `load_only_existing` / `load_only_missing`: optional availability filters for
  debugging generation coverage.

The loader first builds a lightweight `dataset.image_pairs` summary without
decoding every image. Benchmark code uses that summary to decide which models
have outputs for each task.

## Sample Fields

After transformation, each sample contains reference images, paths, metadata,
and optional generated images.

Reference image fields:

- `on_tiff`, `off_tiff`: real HDR references in linear light.
- `on_ldr`, `off_ldr`: tonemapped sRGB references as float arrays in `[0, 1]`.
- `on_path`, `off_path`: resolved reference-image paths.
- `auto_ev_shift_off`: EXIF-derived brightness difference between ON and OFF
  captures.

Generated image fields are added for each model name in `ai_models`:

- `<model>_on`, `<model>_off`: generated sRGB images as float arrays.
- `<model>_on_linear`, `<model>_off_linear`: generated images converted to
  linear light.
- `<model>_on_path`, `<model>_off_path`: resolved generated-image paths.

Metadata and annotation fields from the dataset are also copied onto each
sample where available.

Minimal loader example:

```python
from threedlp.data import ThreeDLPDataset


dataset = ThreeDLPDataset(
    dataset_name="timkuechler/3DLP",
    ai_models=["my_model"],
    generated_root_dir="./3dlp_generated",
    correct_off_img=False,
)

sample = dataset[0]
off_input = sample["off_ldr"]
on_reference = sample["on_tiff"]
generated_on = sample["my_model_on"]
generated_on_linear = sample["my_model_on_linear"]
```

## Generated-Image Discovery

Generated output discovery is implemented in
`threedlp/data/threedlp_generation.py` and `threedlp/data/threedlp_transform.py`.

The canonical path is:

```text
<generated-root>/images/.../images_gen/<model_name>/<sample>_<state>.png
```

For example, `turn_on` looks for the generated `on` image and `turn_off` looks
for the generated `off` image. The loader accepts `.png`, `.jpg`, and `.jpeg`.

The resolver also checks common hyphen/underscore model-name aliases and two
fallback layouts beside the exported input PNGs:

```text
<generated-root>/images/.../images_input/<model_name>/<sample>_<state>.png
<generated-root>/images/.../images_input/<sample>_<state>_<model_name>.png
```

Use the canonical `images_gen/<model_name>/...` layout for final runs. The
fallbacks are mainly there to make simple local runners easier to debug.

## Export Internals

The export command is:

```bash
python -m threedlp.data.export_3dlp_generation_inputs
```

The thin CLI wrapper calls `export_generation_inputs()` in
`threedlp/data/threedlp_export.py`.

For each dataset sample and requested target state:

1. `reverse_state(target_state)` chooses the source state.
2. The source LDR image is saved to `images_input/`.
3. A manifest row is written with `input_path`, `target_state`,
   `source_state`, reference TIFF paths, and `generated_path_template`.

The manifest is intentionally simple JSON Lines so external runners do not need
to import this repository.

## Benchmark Internals

The main benchmark lives in `threedlp/benchmark/calculate_benchmark.py`.

For each task:

1. `active_models_for_task()` checks which requested models have the needed
   generated state.
2. `get_global_target_resolution()` chooses a common evaluation resolution,
   unless `--target-resolution` was provided.
3. `prepare_reference_images()` resizes real OFF, real ON, and the real light
   contribution `max(ON - OFF, 0)`.
4. `prepare_model_images()` loads the generated image for the task and converts
   it to linear light.
5. `compute_evaluation_masks()` removes clipped pixels and keeps pixels with
   enough measured light influence.
6. `build_benchmark_mask()` removes default excluded labels, currently
   `windows`.
7. `compute_metric_scores()` computes SIE and LFE-style affine-ratio errors.

The benchmark writes one CSV row per task and model.

## Metrics

Metric code lives in `threedlp/benchmark/affine_metrics.py`.

The metric first builds task-specific ratio images:

- `turn_on`: compares `I_AI_on / I_R_off` with `I_R_on / I_R_off`.
- `turn_off`: compares `I_AI_off / I_R_on` with `I_R_off / I_R_on`.

The ratios are robustly standardized per channel using median and MAD. This
makes the score focus on spatial light-transport structure instead of a single
global brightness or color fit.

The main outputs are:

- `affine_intensity_error`: SIE over the benchmark-valid image.
- `affine_low_freq_falloff`: LFE over the 80th-percentile low-gradient subset.

Lower values are better.

## Masks and Annotations

Evaluation masks combine several filters:

- clipping checks on reference and generated images;
- minimum measured light influence;
- optional annotation masks, such as excluding windows.

Annotation masks are loaded by `threedlp/benchmark/label_masks.py`. The loader
supports masks stored directly in dataset rows, path metadata, or paths inferred
from the real image locations.

The main benchmark uses the full benchmark-valid image with default exclusions.
Region-level scoring is handled separately by
`threedlp/experiments/annotated_regions/calculate_annotated_regions.py`.

## Optional Experiment Scripts

`calculate_annotated_regions.py` computes SIE and LFE80 for specific annotation
regions such as metallic surfaces, mirrors, cast shadows, and specular
highlights. It fits the robust standardization on the full benchmark-valid
image, then averages errors inside each region.

`calculate_intensity_bands.py` groups pixels by measured light intensity and
computes metrics per band. This is useful for checking whether a model handles
strong direct light differently from weaker falloff regions.

## Development

Install the development tools and run the linter:

```bash
python -m pip install -r requirements.txt
python -m pip install -e '.[dev]'
ruff check .
```
