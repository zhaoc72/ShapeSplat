# Command Matrix

## Setup

```bash
pip install -r requirements.txt
pip install -e .
```

## Minimal

```bash
python scripts/run_minimal.py --config configs/minimal.yaml --out outputs/minimal --eval
```

## Real Input

```bash
python scripts/create_example_image.py --out examples/test_image.png --size 128
python scripts/run_minimal.py --config configs/minimal.yaml --input examples/test_image.png --out outputs/real_input --eval
```

## Sanity Checks

```bash
python scripts/run_all_checks.py --config configs/minimal.yaml --out outputs/real_input
python scripts/check_renderer_backend.py --config configs/minimal.yaml --backend soft --out outputs/check_renderer_soft
```

## Same-Mask

```bash
python scripts/create_example_dataset.py --out examples/example_dataset --num-images 4 --size 128
python scripts/run_minimal.py --config configs/same_mask.yaml --input examples/example_dataset/images/example_000.png --mask examples/example_dataset/masks/example_000.npy --out outputs/same_mask_single --eval
```

## Dataset Runner

```bash
python scripts/run_dataset.py --config configs/same_mask.yaml --manifest examples/example_dataset/manifest.csv --out outputs/same_mask_dataset --mask-source file --max-images 3
```

## Ablation

```bash
python scripts/run_ablation.py --config configs/minimal.yaml --ablations configs/ablations.yaml --input examples/test_image.png --out outputs/ablations
```

## Baseline Protocol

```bash
python scripts/run_dummy_baselines.py --config configs/same_mask.yaml --input examples/example_dataset/images/example_000.png --mask examples/example_dataset/masks/example_000.npy --out outputs/dummy_baselines/example_000 --image-id example_000
```

## Comparison

```bash
python scripts/run_comparison.py --config configs/comparison_minimal.yaml --manifest examples/example_dataset/manifest.csv --out outputs/comparison_run --max-images 3
```

## Reporting

```bash
python scripts/generate_report.py --root outputs/comparison_run --out outputs/comparison_run/report --title "ShapeSplat++ Comparison Report"
```

## External Baselines

```bash
python scripts/list_baseline_adapters.py
python scripts/run_external_baseline.py --config configs/same_mask.yaml --external-config configs/external_baselines.yaml --adapter dummy_external --input examples/example_dataset/images/example_000.png --mask examples/example_dataset/masks/example_000.npy --out outputs/external_baseline/example_000/dummy_external --image-id example_000
```

## Stress Benchmark

```bash
python scripts/create_stress_dataset.py --out examples/stress_dataset --num-per-subset 4 --size 128
python scripts/run_stress_benchmark.py --config configs/stress_benchmark.yaml --manifest examples/stress_dataset/manifest.csv --out outputs/stress_benchmark --max-images 12
```

## Editing Suite

```bash
python scripts/run_edit_demo.py --config configs/editing.yaml --input examples/example_dataset/images/example_000.png --mask examples/example_dataset/masks/example_000.npy --out outputs/edit_demo --object-id 0
```

## Reproducibility

```bash
python scripts/finalize_run.py --out outputs/comparison_run --config configs/comparison_minimal.yaml --run-type comparison --manifest examples/example_dataset/manifest.csv --status success
python scripts/list_runs.py --registry runs/run_registry.jsonl --max-rows 20
```

## Shape Bank

```bash
python scripts/prepare_shape_bank.py --source toy --out outputs/shape_bank_prepared --num-points 512 --descriptor-dim 16 --descriptor-mode point_stats
python scripts/check_shape_retrieval.py --config configs/real_shape_bank.yaml --input examples/test_image.png --shape-root outputs/shape_bank_prepared --out outputs/check_shape_retrieval
```

## Paper Experiments

```bash
python scripts/check_paper_ready.py --profile debug --out outputs/check_paper_ready
python scripts/run_paper_experiments.py --profile debug --out outputs/paper_debug --generate-tables --generate-report
```

## Artifact Validation

```bash
python scripts/check_project_health.py
python scripts/run_quick_tests.py
python scripts/validate_artifact.py --matrix configs/command_matrix.yaml --groups quick smoke --out outputs/artifact_validation
```

## Final Paper Experiments

```bash
python scripts/check_final_paper_ready.py --profile configs/paper/final_debug.yaml --out outputs/check_final_ready
python scripts/run_final_paper.py --profile configs/paper/final_debug.yaml --out outputs/final_paper_debug --dry-run
python scripts/run_final_paper.py --profile configs/paper/final_debug.yaml --out outputs/final_paper_debug --generate-tables --generate-report
python scripts/export_all_final_tables.py --root outputs/final_paper_debug --out outputs/final_paper_debug/tables
python scripts/generate_final_report.py --root outputs/final_paper_debug --out outputs/final_paper_debug/report --title "ShapeSplat++ Final Paper Report"
python scripts/run_experiment.py --preset final_paper_debug --out outputs/exp_final_paper_debug
```

Use these commands for the final orchestration smoke test. `final_debug` is intentionally small and can auto-create the example benchmark; replace it with a local final profile before making paper claims.
## CO3Dv2 High-Resolution Workflow

```bat
conda activate shapesplat
cd /d C:\Users\zhaoc\ShapeSplat\shapesplat
python scripts/check_co3dv2_highres_ready.py --config configs/final_ours_co3dv2_highres.yaml --manifest data/co3dv2_single_benchmark/manifest.csv --cache-manifest outputs/cache_co3dv2_real_frontend_vits16_highres/cache_manifest.csv --out outputs/check_co3dv2_highres_ready
python scripts/cache_co3dv2_highres_frontend.py --config configs/co3dv2_real_frontend_highres.yaml --manifest data/co3dv2_single_benchmark/manifest.csv --out-cache outputs/cache_co3dv2_real_frontend_vits16_highres --max-images 5 --write-manifest outputs/cache_co3dv2_real_frontend_vits16_highres/cache_manifest.csv --validate --require-cuda --check-deps-first
python scripts/run_co3dv2_highres_ours.py --config configs/final_ours_co3dv2_highres.yaml --manifest data/co3dv2_single_benchmark/manifest.csv --out outputs/ours_co3dv2_vits16_highres --max-images 5 --frontend-cache-manifest outputs/cache_co3dv2_real_frontend_vits16_highres/cache_manifest.csv
python scripts/inspect_output_resolution.py --out outputs/ours_co3dv2_vits16_highres --max-items 5
python scripts/generate_co3dv2_highres_report.py --root outputs/ours_co3dv2_vits16_highres --out outputs/ours_co3dv2_vits16_highres/report
```

These commands keep CO3Dv2 file masks fixed, use nearest mask resize, and record original/working/render resolution diagnostics.
