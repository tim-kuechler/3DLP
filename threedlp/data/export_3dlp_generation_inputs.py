"""CLI entry point for exporting 3DLP generation input PNGs."""

from __future__ import annotations

from threedlp.data.threedlp_export import build_export_parser, export_generation_inputs


def main() -> None:
    """Parse CLI arguments and export the generation inputs."""
    export_generation_inputs(build_export_parser().parse_args())


if __name__ == "__main__":
    main()

