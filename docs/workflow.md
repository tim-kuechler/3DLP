# 3DLP Workflow

This guide is for people who want to run an image-editing model on the 3DLP
benchmark. The repository does three things:

1. exports real HDR TIFF images to ordinary sRGB PNG inputs for your editing
   model;
2. tells you where to save your generated edited images;
3. computes benchmark scores from those generated images.

It does not run your model for you. You run your model outside this codebase and
save its outputs in the expected location.

For a tour of the repository internals, see [internals.md](internals.md).

## Setup

Use Python 3.11 or newer and run commands from the repository root.

```bash
python -m pip install -r requirements.txt
```

The dataset is loaded from the Hugging Face Hub:

```bash
export DATASET_NAME=timkuechler/3DLP
export GENERATED_ROOT=./3dlp_generated
export MODEL_NAME=my_model
```

`GENERATED_ROOT` is your local working directory for exported inputs and model
outputs. Do not write generated outputs into the Hugging Face dataset cache.
The first run downloads the main HDR subset. Ensure the cache drive has enough
free space; `--limit` reduces the number of exported samples, not the download.

## Step 1: Export Model Inputs

Export PNG inputs for both benchmark tasks:

```bash
python -m threedlp.data.export_3dlp_generation_inputs \
  --dataset-name "$DATASET_NAME" \
  --output-root "$GENERATED_ROOT" \
  --states on off \
  --no-correct-off-img
```

For a quick smoke test, export only a few samples:

```bash
python -m threedlp.data.export_3dlp_generation_inputs \
  --dataset-name "$DATASET_NAME" \
  --output-root "$GENERATED_ROOT" \
  --states on off \
  --limit 5 \
  --no-correct-off-img
```

The command writes:

```text
<generated-root>/manifest.jsonl
<generated-root>/images/.../images_input/*.png
```

Each line in `manifest.jsonl` is one model run. The important fields are:

- `input_path`: the PNG image to give to your model.
- `target_state`: the lighting state your model should generate, either `on`
  or `off`.
- `generated_path_template`: where to save the generated image. Replace
  `{model_name}` with the model name you will pass to the benchmark.

The meaning of the target states is:

- `target_state == "on"`: the input is the real light-off image; generate the
  light-on edit.
- `target_state == "off"`: the input is the real light-on image; generate the
  light-off edit.

## Step 2: Run Your Model

Read `manifest.jsonl`, run your model on every `input_path`, and save the result
to `generated_path_template` after replacing `{model_name}` with your model
name.

Minimal runner skeleton:

```python
from __future__ import annotations

import json
from pathlib import Path

from PIL import Image


MODEL_NAME = "my_model"
MANIFEST_PATH = Path("3dlp_generated/manifest.jsonl")


def run_my_model(input_image: Image.Image, target_state: str) -> Image.Image:
    """Replace this with your own model inference code."""
    raise NotImplementedError


with MANIFEST_PATH.open("r", encoding="utf-8") as handle:
    for line in handle:
        job = json.loads(line)

        input_path = Path(job["input_path"])
        output_path = Path(
            job["generated_path_template"].replace("{model_name}", MODEL_NAME)
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)

        input_image = Image.open(input_path).convert("RGB")
        generated = run_my_model(input_image, target_state=job["target_state"])
        generated.convert("RGB").save(output_path)
```

Recommended output format is 8-bit RGB PNG. JPG and JPEG are also accepted.

The benchmark can resize outputs during evaluation, so the generated image does
not have to match the input resolution exactly. However, the generated image
should keep the same scene framing. Avoid cropping, padding, geometric warping,
or replacing unrelated scene content.

## Expected Output Layout

The safest option is to use the exact path from `generated_path_template`. That
creates this layout:

```text
<generated-root>/
  images/.../
    images_input/
      <sample>_off.png
      <sample>_on.png
    images_gen/
      my_model/
        <sample>_on.png
        <sample>_off.png
```

The model name in the path must match the name passed to `--models`. For
example, if outputs are saved under `images_gen/my_model/`, evaluate with:

```bash
--models my_model
```

## Step 3: Check Coverage

Before running the benchmark, check that every manifest row has an output file:

```bash
python - <<'PY'
import json
from pathlib import Path

model_name = "my_model"
manifest = Path("3dlp_generated/manifest.jsonl")

missing = []
total = 0
with manifest.open("r", encoding="utf-8") as handle:
    for line in handle:
        total += 1
        row = json.loads(line)
        path = Path(row["generated_path_template"].replace("{model_name}", model_name))
        if not path.exists():
            missing.append(path)

print(f"checked={total}")
print(f"missing={len(missing)}")
for path in missing[:10]:
    print(path)
PY
```

Missing files are skipped for that model and task. Partial runs are fine for
debugging, but final results should have the expected coverage.

You can also count generated files directly:

```bash
find "$GENERATED_ROOT" -path "*/images_gen/$MODEL_NAME/*" -type f | wc -l
```

## Step 4: Run the Benchmark

Evaluate both tasks:

```bash
python -m threedlp.benchmark.calculate_benchmark \
  --dataset-name "$DATASET_NAME" \
  --generated-root-dir "$GENERATED_ROOT" \
  --models "$MODEL_NAME" \
  --tasks turn_on turn_off \
  --output-csv "results/${MODEL_NAME}_3dlp.csv" \
  --max-cores 8 \
  --no-correct-off-img
```

For debugging, use one worker and one task:

```bash
python -m threedlp.benchmark.calculate_benchmark \
  --dataset-name "$DATASET_NAME" \
  --generated-root-dir "$GENERATED_ROOT" \
  --models "$MODEL_NAME" \
  --tasks turn_on \
  --output-csv "results/${MODEL_NAME}_debug.csv" \
  --max-cores 1 \
  --no-correct-off-img
```

To evaluate several models at once, list all model names:

```bash
python -m threedlp.benchmark.calculate_benchmark \
  --dataset-name "$DATASET_NAME" \
  --generated-root-dir "$GENERATED_ROOT" \
  --models my_model baseline_model other_model \
  --tasks turn_on turn_off \
  --output-csv results/multi_model_3dlp.csv \
  --max-cores 8 \
  --no-correct-off-img
```

## Benchmark Output

The benchmark writes one CSV row per task and model.

Important columns:

- `task`: `turn_on` or `turn_off`.
- `Model`: evaluated model name.
- `Data_Coverage`: number of images with at least one valid metric value.
- `<metric>_dataset_avg`: mean score across valid images.
- `<metric>_dataset_std`: standard deviation across valid images.
- `<metric>_valid_count`: number of images that produced that metric.

Default metric columns are:

- `affine_intensity_error`: Standardised Intensity Error, or SIE.
- `affine_low_freq_falloff`: Low-Frequency Error, or LFE, on the
  80th-percentile low-gradient subset.

Lower scores are better.

## Useful Options

For `threedlp.benchmark.calculate_benchmark`:

- `--generated-root-dir`: directory containing your generated images.
- `--models`: one or more model names to evaluate.
- `--tasks`: `turn_on`, `turn_off`, or both.
- `--target-resolution HEIGHT WIDTH`: force evaluation at one resolution. If
  omitted, the benchmark chooses a common resolution from the data.
- `--max-cores`: number of worker processes.
- `--correct-off-img` / `--no-correct-off-img`: OFF-image correction is enabled
  by default in the dataset loader and input exporter, and disabled by default
  in benchmark evaluation. The examples above explicitly disable it.

## Optional Analyses

The main benchmark above is the result most users need. Two additional scripts
are available for deeper analysis.

Annotated-region scores:

```bash
python -m threedlp.experiments.annotated_regions.calculate_annotated_regions \
  --dataset-name "$DATASET_NAME" \
  --generated-root-dir "$GENERATED_ROOT" \
  --models "$MODEL_NAME" \
  --tasks turn_on turn_off \
  --output-csv "results/${MODEL_NAME}_annotated_regions.csv" \
  --max-cores 8 \
  --no-correct-off-img
```

Light-intensity-band scores:

```bash
python -m threedlp.experiments.intensity_bands.calculate_intensity_bands \
  --dataset-name "$DATASET_NAME" \
  --generated-root-dir "$GENERATED_ROOT" \
  --models "$MODEL_NAME" \
  --task turn_on \
  --num-bands 6 \
  --output-csv "results/${MODEL_NAME}_intensity_bands_turn_on.csv" \
  --max-cores 8
```

The intensity-band script currently uses `correct_off_img=False` internally,
matching the `--no-correct-off-img` examples above.

## Command Reference

```bash
python -m threedlp.data.export_3dlp_generation_inputs --help
python -m threedlp.benchmark.calculate_benchmark --help
python -m threedlp.experiments.annotated_regions.calculate_annotated_regions --help
python -m threedlp.experiments.intensity_bands.calculate_intensity_bands --help
```
