"""Regenerate website previews from the original images (requires Pillow).

Run from any directory: python scripts/generate_webp.py
The original JPEGs and pipeline PNG are retained.
"""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image, ImageOps


ROOT = Path(__file__).resolve().parents[1]
COMPARISON_WIDTHS = (360, 720, 1080)


def generate(source, widths, lossless=False):
    total = 0
    with Image.open(source) as original:
        image = ImageOps.exif_transpose(original)
        profile = original.info.get("icc_profile")
        for width in widths:
            if width > image.width:
                raise ValueError(f"Preview would upscale {source}: {width}px")
            height = round(image.height * width / image.width)
            preview = image.resize((width, height), Image.Resampling.LANCZOS)
            target = source.with_name(f"{source.stem}-{width}.webp")
            options = {"quality": 90, "method": 6, "lossless": lossless}
            if profile:
                options["icc_profile"] = profile
            preview.save(target, "WEBP", **options)
            total += target.stat().st_size
    return total


def generate_comparison(source):
    return generate(source, COMPARISON_WIDTHS)


def main():
    sources = sorted((ROOT / "static/videos/comparison").rglob("*.jpg"))
    with ThreadPoolExecutor(max_workers=4) as pool:
        sizes = list(pool.map(generate_comparison, sources))
    diagram_bytes = generate(
        ROOT / "static/images/method_overview.png", (960, 1920), lossless=True
    )
    print(f"Generated {len(sources) * len(COMPARISON_WIDTHS)} comparison previews "
          f"({sum(sizes) / 1_000_000:.2f} MB across all sizes)")
    print(f"Generated 2 lossless pipeline previews ({diagram_bytes / 1_000_000:.2f} MB)")


if __name__ == "__main__":
    main()
