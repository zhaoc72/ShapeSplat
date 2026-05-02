from __future__ import annotations

from pathlib import Path

from shapesplat.datasets.converters.base import DatasetConverter


class Pix3DConverterTemplate(DatasetConverter):
    """Pix3D-style real-image diagnostic converter template."""

    name = "pix3d_template"

    def convert(self, src: str | Path, out: str | Path, cfg: dict | None = None) -> Path:
        if not Path(src).exists():
            raise FileNotFoundError(f"source not found: {src}")
        raise NotImplementedError("This is a template. Please adapt raw dataset parsing to your local Pix3D format.")

