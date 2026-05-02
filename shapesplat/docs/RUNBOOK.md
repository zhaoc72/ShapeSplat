# ShapeSplat++ Runbook

本手册用于从零开始运行当前 artifact-ready 版本。所有命令都可以在 Windows PowerShell 下执行。

## 1. Environment

```bash
conda create -n shapesplat python=3.11
conda activate shapesplat
pip install -r requirements.txt
pip install -e .
```

## 2. Minimal

```bash
python scripts/run_minimal.py --config configs/minimal.yaml --out outputs/minimal --eval
```

## 3. Real Input

```bash
python scripts/create_example_image.py --out examples/test_image.png --size 128
python scripts/run_minimal.py --config configs/minimal.yaml --input examples/test_image.png --out outputs/real_input --eval
```

## 4. Same-Mask Dataset

```bash
python scripts/create_example_dataset.py --out examples/example_dataset --num-images 4 --size 128
python scripts/run_dataset.py --config configs/same_mask.yaml --manifest examples/example_dataset/manifest.csv --out outputs/same_mask_dataset --mask-source file --max-images 3
```

## 5. Comparison

```bash
python scripts/run_comparison.py --config configs/comparison_minimal.yaml --manifest examples/example_dataset/manifest.csv --out outputs/comparison_run --max-images 3
```

## 6. Stress Benchmark

```bash
python scripts/create_stress_dataset.py --out examples/stress_dataset --num-per-subset 4 --size 128
python scripts/run_stress_benchmark.py --config configs/stress_benchmark.yaml --manifest examples/stress_dataset/manifest.csv --out outputs/stress_benchmark --max-images 12
```

## 7. Editing

```bash
python scripts/run_edit_demo.py --config configs/editing.yaml --input examples/example_dataset/images/example_000.png --mask examples/example_dataset/masks/example_000.npy --out outputs/edit_demo --object-id 0
```

## 8. Paper Debug

```bash
python scripts/run_paper_experiments.py --profile debug --out outputs/paper_debug --generate-tables --generate-report
```

## 9. One-Command Orchestration

```bash
python scripts/run_experiment.py --preset debug_all --out outputs/exp_debug_all
```

## 10. Windows + RTX 5090 GPU Runtime

```powershell
conda activate shapesplat
python scripts/print_gpu_info.py
python scripts/check_gpu_runtime.py --config configs/local_windows_rtx5090.yaml --device cuda --require-cuda --out outputs/check_gpu_runtime
python scripts/run_gpu_smoke_experiment.py --config configs/local_windows_rtx5090.yaml --out outputs/gpu_smoke --require-cuda --iters 2
powershell -ExecutionPolicy Bypass -File scripts/run_windows_gpu_paper_debug.ps1
```

中文注释：如果 PyTorch CUDA build 不支持 RTX 5090 / sm_120，GPU check 会明确失败；普通 debug 可用 `--allow-cpu-fallback` 显式允许 CPU。

这些命令使用 stub/minimal pipeline，不需要下载真实模型。
