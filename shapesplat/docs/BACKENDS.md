# Backends

## SAM Backend

支持 `stub`、`real`、`auto`。默认使用 stub；`auto` 在真实模型不可用时 fallback 到 stub。

检查：

```bash
python scripts/check_sam_backend.py --config configs/minimal.yaml
```

## DINO Backend

支持 `stub`、`real`、`auto`。真实 DINOv3 不会自动下载，需要用户配置 checkpoint/model_name。

检查：

```bash
python scripts/check_dino_backend.py --config configs/minimal.yaml
```

## Depth Backend

支持 `stub`、`real`、`auto`。默认深度为轻量 stub。

检查：

```bash
python scripts/check_depth_backend.py --config configs/minimal.yaml
```

## Shape Bank

支持 `toy`、`file`、`auto`。prepared file bank 推荐包含 `points`、`descriptor`、`category`。

检查：

```bash
python scripts/check_shape_bank.py --config configs/file_shape_bank.yaml
```

## Renderer

当前默认 `soft` renderer；`real` renderer 仍是后续扩展入口。

检查：

```bash
python scripts/check_renderer_backend.py --config configs/minimal.yaml --backend soft --out outputs/check_renderer_soft
```

## Real Frontend Check

统一检查 SAM/DINO/Depth：

```bash
python scripts/check_real_frontend.py --config configs/local_real_frontend.yaml --input examples/test_image.png --out outputs/check_real_frontend --save-cache
```

真实 backend fallback 到 stub 是预期行为，除非用户显式要求 real-only。

## Local Real Integration Smoke Test

生成本地 backend 模板：

```bash
python scripts/create_local_backend_template.py --out configs/my_local_backend.yaml
```

检查 backend capability：

```bash
python scripts/check_backend_capabilities.py --config configs/local_backend_template.yaml --out outputs/backend_capabilities
```

运行真实集成 smoke test：

```bash
python scripts/run_real_integration_smoke.py --config configs/local_backend_template.yaml --input examples/test_image.png --out outputs/real_integration_smoke --save-cache
```

填写 `local_backend_template.yaml` 时，常用字段包括：

- `frontend.sam3_checkpoint`
- `frontend.dino_model_name` / `frontend.dino_checkpoint`
- `frontend.depth_model_name` / `frontend.depth_checkpoint`
- `shape_bank.root`
- `renderer.real_renderer_module`
- `renderer.real_renderer_class`

Fallback 规则：

- `auto` 会优先尝试真实 backend；
- 真实 backend 不可用时 fallback 到 `stub` / `soft`；
- `real` 表示强制真实 backend，失败时会报错；
- 真实 backend 接入失败不应影响默认 minimal pipeline。
## Real 3DGS Renderer Adapter

默认仍使用 soft renderer：

```yaml
renderer:
  backend: soft
```

本地检查真实 3D Gaussian Splatting renderer 时可以使用 auto：

```yaml
renderer:
  backend: auto
  fallback_to_soft: true
  real_3dgs:
    library: auto
```

检查命令：

```bash
python scripts/check_real_renderer.py --config configs/real_3dgs_renderer.yaml --backend auto --out outputs/check_real_renderer --allow-fallback
```

真实 renderer 必须输出标准 `RenderOutput`：
- `rgb`
- `alpha`
- `depth`
- `contributions`
- `ownership`
- `bg_ownership`

如果真实 renderer 不支持 native object contributions，adapter 预留 `object_wise_alpha` fallback：逐 object 渲染 alpha，再归一化得到 ownership。该策略较慢，但能保持 ShapeSplat++ 的 object ownership supervision 协议。

当前 artifact 不强制安装 `diff-gaussian-rasterization` / `gsplat`。用户后续可以在 `Real3DGSRendererAdapter._render_with_xxx` 中接入本地 CUDA renderer。
## Windows + RTX 5090 Runtime

Backend availability and CUDA runtime are separate checks. A renderer may fallback to soft even when CUDA works, and CUDA may fail even when `torch.cuda.is_available()` is true.

```powershell
python scripts/print_gpu_info.py
python scripts/check_gpu_runtime.py --config configs/local_windows_rtx5090.yaml --device cuda --require-cuda --out outputs/check_gpu_runtime
python scripts/run_gpu_smoke_experiment.py --config configs/local_windows_rtx5090.yaml --out outputs/gpu_smoke --require-cuda --iters 2
```

Use `--device cuda --require-cuda` for GPU-required experiments. Use `--allow-cpu-fallback` only when CPU fallback is intentional for debugging.
## CO3Dv2 + DINOv3 Official Frontend

CO3Dv2 single diagnostics use CO3D file masks as the default same-mask protocol input. DINOv3 is a frozen descriptor backend; SAM3 is optional for automatic-mask diagnostics only.

Check ViT-S/16 first:

```powershell
python scripts/check_dinov3_dependencies.py
python scripts/check_dinov3_weights.py --config configs/co3dv2_real_frontend_debug.yaml --input examples/test_image.png --out outputs/check_dinov3_vits16 --device cuda
```

If the dependency check reports `torchmetrics` or related DINOv3 helper packages missing, install only:

```powershell
python -m pip install torchmetrics omegaconf ftfy regex scikit-learn submitit termcolor
```

Cache descriptors with CO3D file masks:

```powershell
python scripts/cache_co3dv2_real_frontend.py --config configs/co3dv2_real_frontend_debug.yaml --manifest data/co3dv2_single_benchmark/manifest.csv --out-cache outputs/cache_co3dv2_real_frontend_vits16 --max-images 5 --write-manifest outputs/cache_co3dv2_real_frontend_vits16/cache_manifest.csv --validate --require-cuda --check-deps-first
```

SAM3 diagnostic:

```powershell
python scripts/check_sam3_vs_co3d_masks.py --config configs/co3dv2_real_frontend.yaml --manifest data/co3dv2_single_benchmark/manifest.csv --out outputs/check_sam3_vs_co3d_masks --max-images 10 --device cuda
```

The config paths point to local checkpoints under `D:\projects\dinov3\checkpoint` and `D:\projects\sam3\checkpoint`. The project never downloads these weights.
