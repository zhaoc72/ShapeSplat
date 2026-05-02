from __future__ import annotations

from pathlib import Path

from shapesplat.datasets.converters.base import DatasetConverter


class CO3DConverterTemplate(DatasetConverter):
    """CO3D / LVIS-like real-image diagnostic converter template."""

    name = "co3d_template"

    def convert(self, src: str | Path, out: str | Path, cfg: dict | None = None) -> Path:
        if not Path(src).exists():
            raise FileNotFoundError(f"source not found: {src}")
        raise NotImplementedError("This is a template. Please adapt raw dataset parsing to your local CO3D/LVIS-like format.")

