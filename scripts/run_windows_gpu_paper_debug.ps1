# 中文注释：Windows + RTX 5090 一键 GPU debug。请先激活 conda env shapesplat。
$ErrorActionPreference = "Stop"

Write-Host "Conda env: $env:CONDA_DEFAULT_ENV"
if ($env:CONDA_DEFAULT_ENV -ne "shapesplat") {
  Write-Warning "当前 CONDA_DEFAULT_ENV 不是 shapesplat，请确认你已运行 conda activate shapesplat。"
}

python scripts/print_gpu_info.py
python scripts/check_gpu_runtime.py --config configs/local_windows_rtx5090.yaml --device cuda --require-cuda --out outputs/check_gpu_runtime
python scripts/run_gpu_smoke_experiment.py --config configs/local_windows_rtx5090.yaml --out outputs/gpu_smoke --require-cuda --iters 2
python scripts/run_final_paper.py --profile configs/paper/final_debug.yaml --out outputs/windows_gpu_paper_debug --generate-tables --generate-report --device cuda --require-cuda

Write-Host "Windows GPU paper debug completed."
