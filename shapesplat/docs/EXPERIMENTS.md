# Experiments

ShapeSplat++ 当前实验结构围绕 same-mask setting 展开：所有方法共享同一张图像和同一组 retained visible instance masks，从而减少 proposal quality 对 reconstruction / ownership / editing 指标的干扰。

## Main Comparison

`run_comparison.py` 运行 Ours、dummy baselines 和可选 independent Gaussian baseline，输出 per-image comparison 和 per-method summary。

## Ablation

`run_ablation.py` 使用 `configs/ablations.yaml` 切换 loss/branch 配置，用于检查模块贡献。

## Stress Benchmark

stress dataset 覆盖 occlusion、same-category、contact-heavy、truncation、scale variation 和 small-object 场景。它是 synthetic diagnostic，不是真实最终 benchmark。

## Editing Benchmark

editing suite 对 object buffers 执行 remove/translate/scale/isolate/object_only，并统计 CollateralL1、EditLocality、DeletionResidual 等指标。

## Baseline Protocol

baseline input/output protocol 规定 image、masks、crops、render、alpha、ownership 等文件格式，方便后续接入外部方法。

## Metrics Groups

指标分为 object consistency、rendering、editing、stress 和 optional geometry。geometry 只有预测点云和 GT 点云都存在时才启用。

## Output Structure

常见输出包括 `metrics.json`、`summary.json`、`per_method_summary.json`、`stress_subset_summary.json`、`edit_summary.json`、`report.md` 和 `tables/*.tex`。

dummy baselines 和 toy datasets 只用于 smoke tests，不是论文最终实验结果。
## Final Benchmark Format

Benchmark v2 standardizes paper-ready same-mask inputs:

```text
benchmark_root/
  images/
  masks/
  metadata/
  depth/ cameras/ gt_pointclouds/ gt_meshes/ frontend_cache/  # optional
  manifest.csv
  splits.json
  benchmark_info.json
```

Required manifest fields:
- `image_id`
- `image_path`
- `mask_path`
- `split`

Recommended fields include `metadata_path`, `subset`, `category`, `num_objects`, `scene_id`, and `source_dataset`. Optional GT fields such as `gt_pointcloud_path` and `gt_mesh_path` are only used when available; they are not required for standard same-mask experiments.

Convert a generic folder:

```bash
python scripts/convert_benchmark.py --converter generic_folder --src examples/example_dataset --out data/example_benchmark_v2 --source-dataset example --overwrite
```

Validate:

```bash
python scripts/validate_benchmark_v2.py --manifest data/example_benchmark_v2/manifest.csv --config configs/final_benchmark.yaml --out outputs/benchmark_v2_validation
```

Bind frontend cache:

```bash
python scripts/bind_cache_to_benchmark.py --manifest data/example_benchmark_v2/manifest.csv --cache-manifest outputs/frontend_cache/cache_manifest.csv --out-manifest data/example_benchmark_v2/manifest_with_cache.csv
```

Dataset-specific GSO / Objaverse / Pix3D / CO3D converters are templates. Users should adapt raw parsing to their local dataset layout, then output benchmark v2.


## Ours Reconstruction Core Runner

`run_ours_benchmark.py` is the formal entry point for running the ShapeSplat++ Ours method on benchmark v2 manifests. It is an orchestration layer over the existing frontend, Trainer, renderer, metrics, and output protocol.

Recommended workflow:

```powershell
python scripts/check_ours_core_ready.py --config configs/final_ours.yaml --manifest data/example_benchmark_v2/manifest.csv --out outputs/check_ours_core_ready
python scripts/run_ours_benchmark.py --config configs/final_ours.yaml --manifest data/example_benchmark_v2/manifest.csv --out outputs/ours_benchmark --max-images 3
python scripts/run_ours_variants.py --config configs/final_ours.yaml --variants configs/ours_variants.yaml --manifest data/example_benchmark_v2/manifest.csv --out outputs/ours_variants --variant full --variant visible_only --max-images 3
```

The runner saves baseline-compatible outputs, so comparison can reuse a previously computed Ours directory with `--ours-output-dir` instead of retraining. The readiness checker warns when a run still uses stub frontends, ToyShapeBank, or SoftGaussianRenderer; those settings are acceptable for smoke tests but not final paper claims.

## Final Baseline and Geometry Metrics

The final evaluation layer reads method outputs from `configs/final_method_outputs.yaml` and method metadata from `configs/method_catalog.yaml`. It can evaluate Ours, Ours variants, internal baselines, dummy baselines, and externally generated outputs as long as they satisfy the baseline output protocol.

```powershell
python scripts/evaluate_method_outputs.py --method ours_full --outputs outputs/ours_benchmark --manifest data/example_benchmark_v2/manifest.csv --config configs/final_ours.yaml --out outputs/eval_ours_full --max-images 3
python scripts/run_final_comparison.py --manifest data/example_benchmark_v2/manifest.csv --methods configs/method_catalog.yaml --outputs-config configs/final_method_outputs.yaml --config configs/final_ours.yaml --out outputs/final_comparison --max-images 3
python scripts/export_final_tables.py --summary outputs/final_comparison/final_method_summary.json --out outputs/final_comparison/tables
```

Geometry metrics are optional. They are reported only when `pred_pointcloud.npy` and a GT pointcloud field such as `gt_pointcloud_path` both exist. The current pointcloud evaluator is lightweight and does not replace a final, explicitly aligned geometry protocol.

## Final Paper Experiments

`scripts/run_final_paper.py` is the final paper-style orchestration entry. It does not introduce new algorithm logic; it serially calls the existing benchmark validator, Ours runner, variants runner, internal baselines, final comparison evaluator, stress/editing runners, table exporters, readiness checker, and report generator.

```powershell
python scripts/check_final_paper_ready.py --profile configs/paper/final_debug.yaml --out outputs/check_final_ready
python scripts/run_final_paper.py --profile configs/paper/final_debug.yaml --out outputs/final_paper_debug --dry-run
python scripts/run_final_paper.py --profile configs/paper/final_debug.yaml --out outputs/final_paper_debug --generate-tables --generate-report
python scripts/export_all_final_tables.py --root outputs/final_paper_debug --out outputs/final_paper_debug/tables
python scripts/generate_final_report.py --root outputs/final_paper_debug --out outputs/final_paper_debug/report --title "ShapeSplat++ Final Paper Report"
```

`configs/paper/final_debug.yaml` intentionally uses tiny example data and fallback components so the whole chain can be smoke-tested on CPU. Real paper runs should use `configs/paper/final_all.yaml` or a local profile with fixed benchmark manifests, cached real frontend outputs, prepared shape banks, renderer settings, and external baseline output roots.

The final readiness report warns about stub frontends, ToyShapeBank, SoftGaussianRenderer, disabled external baselines, and missing geometry GT. In non-strict mode those are warnings so debug runs remain usable; in strict mode they can be promoted to blockers.

## CO3Dv2 Single Subset Diagnostics

CO3Dv2 single subset is treated as a real-image diagnostic dataset. It is usually object-centric with one visible foreground mask per frame, so it should not be described as the multi-object occlusion benchmark. ShapeSplat++ fixes the CO3D mask as a retained visible mask and runs the same reconstruction/output protocol.

Local path example:

```powershell
D:\projects\datasets\co3dv2_single
```

Inspect the local structure before conversion:

```powershell
python scripts/inspect_co3dv2_single.py --root D:\projects\datasets\co3dv2_single --out outputs/inspect_co3dv2_single
```

Convert a small subset first:

```powershell
python scripts/convert_co3dv2_single.py --root D:\projects\datasets\co3dv2_single --out data/co3dv2_single_benchmark --max-categories 3 --max-sequences 2 --max-frames-per-sequence 5 --copy-files --overwrite
```

Validate and run diagnostics:

```powershell
python scripts/validate_benchmark_v2.py --manifest data/co3dv2_single_benchmark/manifest.csv --config configs/final_benchmark.yaml --out outputs/validate_co3dv2_single
python scripts/run_co3dv2_diagnostics.py --manifest data/co3dv2_single_benchmark/manifest.csv --config configs/final_ours.yaml --out outputs/co3dv2_diagnostics --max-images 20 --generate-report
```

The converter first tries lightweight CO3D annotations such as `frame_annotations.jgz`; if that fails, it scans `images/`, `masks/`, `depths/`, `depth_masks/`, and `pointcloud.ply`. Depth, camera, and pointcloud fields are optional and missing values should not block conversion. Geometry metrics should only be reported after a clear pointcloud conversion and alignment protocol is fixed.

### CO3Dv2 Real Frontend Cache

For CO3Dv2 single, the main protocol keeps `mask_source: file` and uses official DINOv3 only for descriptors. Start with ViT-S/16:

```powershell
python scripts/check_dinov3_dependencies.py
python scripts/check_dinov3_weights.py --config configs/co3dv2_real_frontend_debug.yaml --input examples/test_image.png --out outputs/check_dinov3_vits16 --device cuda
python scripts/cache_co3dv2_real_frontend.py --config configs/co3dv2_real_frontend_debug.yaml --manifest data/co3dv2_single_benchmark/manifest.csv --out-cache outputs/cache_co3dv2_real_frontend_vits16 --max-images 5 --write-manifest outputs/cache_co3dv2_real_frontend_vits16/cache_manifest.csv --validate --require-cuda --check-deps-first
```

Then switch to `configs/co3dv2_real_frontend.yaml` for ViT-L/16. SAM3 is available through `scripts/check_sam3_vs_co3d_masks.py`, but it remains an automatic-mask diagnostic and is not the default CO3Dv2 mask source.
## CO3Dv2 High-Resolution Diagnostics

Run from Anaconda Prompt:

```bat
conda activate shapesplat
cd /d C:\Users\zhaoc\ShapeSplat\shapesplat
```

CO3Dv2 single is treated as a real-image diagnostic / single foreground visible-mask benchmark. It is not the multi-object main benchmark. Use file masks, not SAM3, for the main diagnostic path.

```bat
python scripts/check_co3dv2_highres_ready.py --config configs/final_ours_co3dv2_highres.yaml --manifest data/co3dv2_single_benchmark/manifest.csv --cache-manifest outputs/cache_co3dv2_real_frontend_vits16_highres/cache_manifest.csv --out outputs/check_co3dv2_highres_ready
```

```bat
python scripts/cache_co3dv2_highres_frontend.py --config configs/co3dv2_real_frontend_highres.yaml --manifest data/co3dv2_single_benchmark/manifest.csv --out-cache outputs/cache_co3dv2_real_frontend_vits16_highres --max-images 5 --write-manifest outputs/cache_co3dv2_real_frontend_vits16_highres/cache_manifest.csv --validate --require-cuda --check-deps-first
```

```bat
python scripts/run_co3dv2_highres_ours.py --config configs/final_ours_co3dv2_highres.yaml --manifest data/co3dv2_single_benchmark/manifest.csv --out outputs/ours_co3dv2_vits16_highres --max-images 5 --frontend-cache-manifest outputs/cache_co3dv2_real_frontend_vits16_highres/cache_manifest.csv
```

```bat
python scripts/run_co3dv2_highres_variants.py --config configs/final_ours_co3dv2_highres.yaml --manifest data/co3dv2_single_benchmark/manifest.csv --out outputs/ours_variants_co3dv2_vits16_highres --variant full --variant visible_only --max-images 5 --frontend-cache-manifest outputs/cache_co3dv2_real_frontend_vits16_highres/cache_manifest.csv
```

```bat
python scripts/inspect_output_resolution.py --out outputs/ours_co3dv2_vits16_highres --max-items 5
python scripts/generate_co3dv2_highres_report.py --root outputs/ours_co3dv2_vits16_highres --out outputs/ours_co3dv2_vits16_highres/report
```

Readiness warnings about ToyShapeBank and SoftGaussianRenderer mean the run is diagnostic-only. Paper claims need a prepared shape bank and an explicitly reported renderer setting.
