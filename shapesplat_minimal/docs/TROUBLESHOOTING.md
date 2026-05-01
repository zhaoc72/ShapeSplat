# Troubleshooting

## `git` is not recognized

现象：reproducibility 模块无法记录 git commit。

原因：Git 未安装或不在 PATH。

解决：

```bash
git --version
```

没有 git 时实验仍可运行，只是 `environment.json` 中会记录 unavailable。

## PowerShell 不能使用 `python - <<'PY'`

现象：Linux heredoc 命令在 PowerShell 失败。

解决：使用脚本文件，或：

```bash
python -c "import shapesplat; print('ok')"
```

## `ModuleNotFoundError: shapesplat`

原因：没有安装 editable package，或当前目录不对。

解决：

```bash
pip install -e .
python -c "import shapesplat; print('import ok')"
```

## `ModuleNotFoundError: scripts`

原因：测试不应 import `scripts` 目录。可复用逻辑应放在 `src/shapesplat`。

解决：从 `shapesplat.*` 导入正式模块。

## `masks.png` 一片红

原因：mask label map 或 stack 可能只有一个大前景，或 RGB 可视化 palette 叠加过强。

解决：

```bash
python scripts/check_file_masks.py --image examples/example_dataset/images/example_000.png --mask examples/example_dataset/masks/example_000.npy --config configs/minimal.yaml --out outputs/check_file_masks
```

## loss 出现 NaN

原因：学习率过大、mask 为空、输入尺寸异常。

解决：先跑 quick tests，再检查 mask：

```bash
python scripts/run_quick_tests.py
python scripts/validate_benchmark.py --manifest examples/example_dataset/manifest.csv --config configs/same_mask.yaml --out outputs/benchmark_validation
```

## renderer 输出全黑

原因：opacity 太低、camera/shape 不匹配或 renderer backend 配置错误。

解决：

```bash
python scripts/check_renderer_backend.py --config configs/minimal.yaml --backend soft --out outputs/check_renderer_soft
```

## checkpoint 能否加载

当前 artifact 不包含大 checkpoint。真实模型权重需要用户单独配置。

## ownership shape 不匹配

现象：metrics 或 renderer contract 报 `[N,H,W]` 不一致。

解决：检查 renderer contract：

```bash
python scripts/check_renderer_backend.py --config configs/minimal.yaml --backend soft --out outputs/check_renderer_soft
```

## pytest collected 0 items

原因：目录不对或 pytest 未安装。

解决：

```bash
pip install pytest
pytest tests/test_smoke.py -v
```

## 真实 backend fallback 到 stub

原因：checkpoint/model_name 未配置或真实依赖不可用。

解决：

```bash
python scripts/check_real_frontend.py --config configs/local_real_frontend.yaml --input examples/test_image.png --out outputs/check_real_frontend
```

## CUDA 不可用

当前 minimal pipeline 可在 CPU 上运行。真实 backend 需要单独配置 CUDA/PyTorch。

## Windows 路径空格问题

使用命令参数列表脚本，不要手写复杂 shell 字符串。路径有空格时用引号包裹。

## output 文件不存在

先看对应 `logs/*stderr.txt`，再运行 artifact validation dry run。

```bash
python scripts/validate_artifact.py --matrix configs/command_matrix.yaml --groups quick smoke --dry-run --out outputs/artifact_validation_dry
```

## `run_all_checks` 失败

先运行 quick tests 定位基础问题：

```bash
python scripts/run_quick_tests.py
```

