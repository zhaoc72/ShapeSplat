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
