from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import torch

from shapesplat.benchmarks.stress_generator import create_stress_dataset, generate_stress_sample
from shapesplat.benchmarks.stress_metrics import compute_stress_metrics
from shapesplat.benchmarks.stress_runner import run_stress_benchmark


def _tmp_dir(name: str) -> Path:
    path = Path("outputs") / "test_stress_tmp" / f"{name}_{uuid4().hex[:8]}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def test_generate_stress_sample() -> None:
    image, masks, meta = generate_stress_sample("occ_000", "heavy_occlusion", size=64, seed=7)
    assert image.shape == (3, 64, 64)
    assert masks.ndim == 3
    assert meta.subset == "heavy_occlusion"
    assert meta.num_objects >= 2
    assert meta.occlusion_pairs


def test_create_stress_dataset() -> None:
    out = _tmp_dir("dataset")
    manifest = create_stress_dataset(out, num_per_subset=1, size=64, subsets=["normal", "same_category"], seed=11)
    assert manifest.exists()
    assert any((out / "images").glob("*.png"))
    assert any((out / "masks").glob("*.npy"))
    assert any((out / "metadata").glob("*.json"))


def test_stress_metrics() -> None:
    _, masks, meta = generate_stress_sample("samecat_000", "same_category", size=64, seed=9)
    ownership = masks.float()
    ownership = ownership / ownership.sum(dim=0, keepdim=True).clamp_min(1e-6)
    render = SimpleNamespace(ownership=ownership, depth=torch.ones(64, 64))
    metrics = compute_stress_metrics(render, masks, meta)
    assert "SwapRateProxy" in metrics
    assert "OrderAccProxy" in metrics
    assert "OcclusionRecallProxy" in metrics
    assert metrics["Subset"] == "same_category"
    for value in metrics.values():
        assert value is None or isinstance(value, (str, int, float))


def test_run_stress_benchmark_small() -> None:
    root = _tmp_dir("run")
    manifest = create_stress_dataset(root / "data", num_per_subset=1, size=64, subsets=["normal", "same_category"], seed=13)
    out = root / "out"
    rows = run_stress_benchmark(
        "configs/stress_benchmark.yaml",
        manifest,
        out,
        max_images=2,
        run_comparison=False,
        save_visuals=False,
    )
    assert len(rows) == 2
    assert (out / "stress_subset_summary.json").exists()
