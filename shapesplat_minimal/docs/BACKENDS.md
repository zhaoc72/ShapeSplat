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

