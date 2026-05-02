# Baselines

## Input Protocol

baseline inputs 包含：

- `image.png`
- `masks.npy`
- `masks.png`
- `overlay.png`
- `metadata.json`
- `crops/object_XXX_rgb.png`
- `crops/object_XXX_mask.png`
- `crops/object_XXX_rgba.png`

这些 masks 是 retained visible instance masks，不是 amodal masks。

## Output Protocol

baseline outputs 至少包含：

- `render.png` 或 `render_final.png`
- `alpha.png` 或 `alpha_final.png`
- `ownership.npy` 或 `object_XXX_alpha.png`
- `metrics.json` 可选
- `output_spec.json` 可选

验证：

```bash
python scripts/validate_baseline_output.py --output outputs/dummy_baselines/example_000/identity_mask --num-objects 2 --strict
```

## Dummy Baselines

`identity_mask`、`independent_blob`、`scene_union` 只用于 protocol smoke test。

## Independent Gaussian Baseline

`independent_gaussian` 是最小可运行 baseline：每个 object 独立优化，不使用 scene-coupled ownership。

```bash
python scripts/run_independent_gaussian_baseline.py --config configs/benchmark_baseline.yaml --input examples/example_dataset/images/example_000.png --mask examples/example_dataset/masks/example_000.npy --out outputs/independent_gaussian/example_000 --image-id example_000
```

## Command Template Adapter

外部 baseline 可以通过 command template 接入：

```yaml
command: "python external_methods/my_method/run.py --image {image} --masks {masks} --out {output_dir}"
```

## Future Real Baselines

SPAR3D / TRELLIS / VGGT / DUSt3R 等目前只是模板。真实接入需要用户提供外部 repo、模型权重和输出协议适配。
# Final Baseline and Geometry Metrics

`configs/method_catalog.yaml` records method family, output type, editability, and whether a method is enabled by default. Real SPAR3D / TRELLIS / VGGT / DUSt3R entries are disabled templates until users provide external outputs.

Final comparison expects method outputs in baseline-compatible folders:

```text
method_outputs/
  method_name/
    image_id/
      render.png or render_final.png
      alpha.png or alpha_final.png
      ownership.npy or object_i_alpha.png
      pred_pointcloud.npy optional
      output_spec.json optional
```

Run the unified evaluator:

```powershell
python scripts/evaluate_method_outputs.py --method ours_full --outputs outputs/ours_benchmark --manifest data/example_benchmark_v2/manifest.csv --config configs/final_ours.yaml --out outputs/eval_ours_full --max-images 3
python scripts/run_final_comparison.py --manifest data/example_benchmark_v2/manifest.csv --methods configs/method_catalog.yaml --outputs-config configs/final_method_outputs.yaml --config configs/final_ours.yaml --out outputs/final_comparison --max-images 3
```

Chamfer/F-score are optional and only run when both prediction and GT pointclouds are present. No LPIPS, pandas, open3d, or trimesh dependency is required.

