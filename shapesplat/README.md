# ShapeSplat++ Minimal

ShapeSplat++ Minimal 是一个最小可运行的单图多前景物体 3D Gaussian 重建框架。当前目标不是追求真实重建效果，而是把完整 pipeline 稳定跑通，并为后续替换真实 SAM3、DINOv3、深度模型、真实 shape bank 和 CUDA 3D Gaussian renderer 留出清晰接口。

## 最小技术路线

SAM3-DINOv3 frozen front-end + visible-hidden Gaussian buffers + scene-coupled ownership rendering + confidence-weighted hidden support prior + differentiable edit-consistency optimization。

当前版本默认使用 stub：

- `Sam3Stub`：用前景启发式和 connected components 产生 retained visible masks。
- `DinoV3Stub`：用 RGB、坐标和正弦位置编码构造 dense features，并做 mask pooling。
- `DepthStub`：生成 canonical weak monocular depth。
- `ToyShapeBank`：提供 sphere/box/cylinder toy point clouds。
- `SoftGaussianRenderer`：纯 PyTorch 2D soft splatting，可反向传播，但不是 CUDA 3DGS renderer。

## 安装

```bash
pip install -r requirements.txt
pip install -e .
```

## 运行

```bash
python scripts/run_minimal.py --config configs/minimal.yaml --out outputs/minimal
```

如果没有提供 `--input`，脚本会自动生成 synthetic 多物体 RGB 图像。

## 输出文件

- `input.png`：输入图像。
- `masks.png`：SAM backend 的 visible instance masks。
- `input_mask_overlay.png`：输入图和 masks 的半透明叠加。
- `render_final.png`：最终 soft renderer RGB。
- `alpha_final.png`：最终 alpha。
- `ownership_argmax.png`：每个像素归属哪个 object 的可视化。
- `object_0_alpha.png`, `object_1_alpha.png` 等：per-object ownership/alpha-like map。
- `loss_log.json`：训练损失日志。
- `checkpoint_minimal.pt`：最小 checkpoint。

## Sanity Checks / 最小版本检查

```bash
python scripts/run_minimal.py --config configs/minimal.yaml --out outputs/minimal
python scripts/check_loss.py --log outputs/minimal/loss_log.json
python scripts/check_checkpoint.py --checkpoint outputs/minimal/checkpoint_minimal.pt
python scripts/check_renderer_shape.py --config configs/minimal.yaml
python scripts/check_backward.py --config configs/minimal.yaml --stage visible
python scripts/run_all_checks.py --config configs/minimal.yaml --out outputs/minimal
```

这些检查通过意味着 front-end 输出正常、Gaussian scene 初始化正常、renderer 输出 shape 正常、loss 没有 NaN/Inf、checkpoint 可以加载、反向传播链路是通的。

## Real Image Input / 真实图像输入测试

生成示例图片：

```bash
python scripts/create_example_image.py --out examples/test_image.png --size 128
```

使用真实输入路径运行：

```bash
python scripts/run_minimal.py --config configs/minimal.yaml --input examples/test_image.png --out outputs/real_input
```

或者使用 demo：

```bash
python scripts/run_real_input_demo.py --config configs/minimal.yaml --input examples/test_image.png --out outputs/real_input
```

检查输出：

```bash
python scripts/run_all_checks.py --config configs/minimal.yaml --out outputs/real_input
```

当前仍然默认使用 `Sam3Stub`、`DinoV3Stub`、`DepthStub` 和 `SoftGaussianRenderer`。这一步只验证真实图像输入、resize、front-end stub、Gaussian 初始化、renderer、loss 和保存逻辑是否稳定。

## Optional Real SAM3 Backend / 可选真实 SAM3 前端

默认仍然使用 stub：

```yaml
frontend:
  sam_backend: stub
```

自动模式会优先尝试真实 SAM3，如果依赖或 checkpoint 不可用，会 fallback 到 `Sam3Stub`：

```yaml
frontend:
  sam_backend: auto
  sam3_checkpoint: path/to/checkpoint
```

真实模式会强制使用 `RealSAM3Wrapper`，如果 checkpoint 或依赖缺失会报错：

```yaml
frontend:
  sam_backend: real
  sam3_checkpoint: path/to/checkpoint
```

## CO3Dv2 Single Subset Diagnostics

CO3Dv2 single subset is supported as a local real-image diagnostic dataset. It is usually object-centric with one visible foreground mask per frame, so this path is not treated as the multi-object occlusion benchmark. ShapeSplat++ uses the CO3D mask as a fixed retained visible mask for same-mask diagnostics.

Example local root:

```powershell
D:\projects\datasets\co3dv2_single
```

Inspect the folder structure:

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

Preset:

```powershell
python scripts/run_experiment.py --preset co3dv2_cache --out outputs/exp_co3dv2_cache
```

The converter does not require PyTorch3D or the official CO3D package. It tries lightweight `.jgz` annotations first and falls back to folder scanning. Depth, camera, and pointcloud fields are optional; geometry metrics should only be reported after a clear pointcloud conversion and alignment protocol is fixed.

## CO3Dv2 + DINOv3 Official Frontend

CO3Dv2 main experiments should use `mask_source: file`, so the benchmark keeps the CO3D visible foreground masks fixed. DINOv3 is used as a frozen dense descriptor extractor. SAM3 is optional and only used for automatic-mask diagnostics, not as the default CO3Dv2 mask source.

Local paths used by the provided configs:

```powershell
D:\projects\dinov3\checkpoint\dinov3_vits16_pretrain_lvd1689m.pth
D:\projects\dinov3\checkpoint\dinov3_vitl16_pretrain_lvd1689m.pth
D:\projects\sam3\checkpoint\sam3.pt
```

Recommended workflow:

```powershell
python scripts/check_dinov3_dependencies.py
python scripts/check_dinov3_weights.py --config configs/co3dv2_real_frontend_debug.yaml --input examples/test_image.png --out outputs/check_dinov3_vits16 --device cuda
python scripts/cache_co3dv2_real_frontend.py --config configs/co3dv2_real_frontend_debug.yaml --manifest data/co3dv2_single_benchmark/manifest.csv --out-cache outputs/cache_co3dv2_real_frontend_vits16 --max-images 5 --write-manifest outputs/cache_co3dv2_real_frontend_vits16/cache_manifest.csv --validate --require-cuda --check-deps-first
```

If dependencies are missing, install only helper packages and do not reinstall torch:

```powershell
python -m pip install torchmetrics omegaconf ftfy regex scikit-learn submitit termcolor
```

After ViT-S/16 works, switch to `configs/co3dv2_real_frontend.yaml` for ViT-L/16:

```powershell
python scripts/check_dinov3_weights.py --config configs/co3dv2_real_frontend.yaml --input examples/test_image.png --out outputs/check_dinov3_vitl16 --device cuda
```

Optional SAM3 diagnostic:

```powershell
python scripts/check_sam3_vs_co3d_masks.py --config configs/co3dv2_real_frontend.yaml --manifest data/co3dv2_single_benchmark/manifest.csv --out outputs/check_sam3_vs_co3d_masks --max-images 10 --device cuda
```

The scripts do not download weights. If local repos or checkpoints are missing, they write a clear report or raise a clear error depending on `--allow-missing`.

检查 SAM backend：

```bash
python scripts/check_sam_backend.py --config configs/minimal.yaml --backend stub --out outputs/check_sam_stub
```

如果有真实 SAM3：

```bash
python scripts/check_sam_backend.py --config configs/minimal.yaml --backend real --input examples/test_image.png --out outputs/check_sam_real
```

`RealSAM3Wrapper` 是可选接口；真实 SAM3 API 可能需要根据本地安装方式调整。项目其余模块只依赖统一的 `MaskSet` 输出，不关心具体 SAM 实现。

## Optional Real DINOv3 Backend / 可选真实 DINOv3 前端

默认仍然使用 stub：

```yaml
frontend:
  dino_backend: stub
```

自动模式会优先尝试真实 DINOv3，如果依赖或 checkpoint 不可用，会 fallback 到 `DinoV3Stub`：

```yaml
frontend:
  dino_backend: auto
  dino_checkpoint: path/to/checkpoint
```

真实模式会强制使用 `RealDINOv3Wrapper`，如果 checkpoint 或依赖缺失会报错：

```yaml
frontend:
  dino_backend: real
  dino_model_name: your_model_name
  dino_checkpoint: path/to/checkpoint
```

检查 DINO backend：

```bash
python scripts/check_dino_backend.py --config configs/minimal.yaml --backend stub --out outputs/check_dino_stub
```

如果有真实 DINOv3：

```bash
python scripts/check_dino_backend.py --config configs/minimal.yaml --backend real --input examples/test_image.png --out outputs/check_dino_real
```

`RealDINOv3Wrapper` 是可选接口；真实 DINOv3 API 可能需要根据本地安装方式调整。项目其余模块只依赖统一的 dense features 和 descriptors，不关心具体 DINO 实现。DINOv3 在本项目中始终是 frozen dense descriptor extractor。

## Optional Real Depth Backend / 可选真实深度前端

默认仍然使用 stub：

```yaml
frontend:
  depth_backend: stub
```

自动模式会优先尝试真实 depth model，如果依赖或 checkpoint 不可用，会 fallback 到 `DepthStub`：

```yaml
frontend:
  depth_backend: auto
  depth_checkpoint: path/to/checkpoint
```

真实模式会强制使用 `RealDepthWrapper`，如果 checkpoint 或依赖缺失会报错：

```yaml
frontend:
  depth_backend: real
  depth_model_name: your_model_name
  depth_checkpoint: path/to/checkpoint
```

检查 depth backend：

```bash
python scripts/check_depth_backend.py --config configs/minimal.yaml --backend stub --out outputs/check_depth_stub
```

如果有真实 depth model：

```bash
python scripts/check_depth_backend.py --config configs/minimal.yaml --backend real --input examples/test_image.png --out outputs/check_depth_real
```

`RealDepthWrapper` 是可选接口；真实 depth API 可能需要根据本地安装方式调整。Depth 在 ShapeSplat++ 中只作为 weak initialization / layout cue，不是 oracle geometry；所有 depth 都会归一化到 canonical camera range。

## Evaluation Metrics / 最小评估指标

当前 minimal 版本支持：

- Inst-IoU
- AttrAcc
- AttrPurity
- Leakage
- Iso-IoU
- DeletionResidual
- EditLocality
- CollateralL1

这些指标主要用于检查 object ownership、foreground leakage 和 editing stability。当前没有实现 Chamfer / F-score / LPIPS；Chamfer / F-score 需要 GT mesh 和对齐协议，后续实验版本再加入。`CollateralL1` 是 Collateral LPIPS 的 lightweight proxy。

```bash
python scripts/evaluate_minimal.py --config configs/minimal.yaml --input examples/test_image.png --out outputs/eval_real_input
python scripts/run_minimal.py --config configs/minimal.yaml --input examples/test_image.png --out outputs/real_input --eval
```

输出：

- `metrics.json`

## Ablation Experiments / 消融实验

当前支持的消融开关：

- `use_visible_hidden_split`：visible-hidden factorized Gaussian representation。
- `use_hidden_branch`：hidden Gaussian branch。
- `use_hidden_prior`：confidence-weighted hidden support prior。
- `use_confidence_weighting`：用 retrieval confidence 控制 hidden budget 和 prior strength。
- `use_dino_retrieval`：mask-guided DINO descriptor retrieval。
- `use_scene_loss`：scene-level compositional RGB loss。
- `use_ownership_loss`：identity / ownership loss。
- `use_depth_loss`：weak depth consistency loss。
- `use_bg_loss`：background leakage loss。
- `use_bridge_loss`：visible-hidden bridge loss。
- `use_edit_consistency`：differentiable edit-consistency loss。

运行默认消融：

```bash
python scripts/run_ablation.py --config configs/minimal.yaml --ablations configs/ablations.yaml --input examples/test_image.png --out outputs/ablations
```

调试只跑前 3 个：

```bash
python scripts/run_ablation.py --config configs/minimal.yaml --ablations configs/ablations.yaml --input examples/test_image.png --out outputs/ablations_debug --max-experiments 3
```

打印结果表：

```bash
python scripts/print_ablation_table.py --summary outputs/ablations/ablation_summary.json
```

当前 ablation 只是 minimal pipeline 的工程级消融，用于验证代码路径和指标是否工作；正式论文实验需要真实 SAM3/DINOv3、真实 shape bank、真实 renderer 和正式 benchmark。

## 后续真实模块替换路线

- `Sam3Stub` -> `RealSAM3Wrapper`
- `DinoV3Stub` -> `RealDINOv3Wrapper`
- `DepthStub` -> `DepthAnythingWrapper`
- `ToyShapeBank` -> `RealShapeBank`
- `SoftGaussianRenderer` -> `CUDA3DGSRenderer`

替换时建议保持当前 wrapper 方法签名和输出字段不变，尤其是 `MaskSet`、instance descriptors、weak depth、per-object contributions/ownership。

## 实验注意

主实验应采用 same-mask setting，保证和 baselines 公平：不同方法使用相同 visible masks，避免 segmentation 质量差异混入 3D reconstruction/editability 的比较。
## Shape Bank / Hidden Support Prior

默认配置使用 `ToyShapeBank`，只包含 sphere / box / cylinder，目的是让 hidden support prior 的代码路径可以 smoke test。它不是正式实验用的 shape prior。

创建一个 file backend 可读取的 toy 点云库：

```bash
python scripts/create_toy_shape_bank.py --out examples/shape_bank --num-points 512 --descriptor-dim 16
```

检查默认 toy shape bank：

```bash
python scripts/check_shape_bank.py --config configs/minimal.yaml --backend toy --out outputs/check_shape_toy
```

检查 file shape bank：

```bash
python scripts/check_shape_bank.py --config configs/minimal.yaml --backend file --root examples/shape_bank --out outputs/check_shape_file
```

使用 file shape bank 运行最小 pipeline：

```bash
python scripts/run_minimal.py --config configs/file_shape_bank.yaml --input examples/test_image.png --out outputs/file_shape_bank --eval
```

`FileShapeBank` 目前支持 `.npz` / `.npy` point cloud。`.npz` 至少需要包含 `points [P,3]`，可选包含 `descriptor [D]`、`descriptors [V,D]` 和 `category`。如果没有 descriptor，当前 minimal 版本可以生成 deterministic random descriptor，后续应替换为真实 DINOv3 多视角 shape descriptor 预计算。

正式论文实验时，shape bank 必须与测试实例 train/test instance-disjoint，避免检索泄漏。hidden support prior 是 soft prior，不是 hard template fitting。

## Renderer Backend / 渲染器后端

默认仍然使用 `SoftGaussianRenderer`：

```yaml
renderer:
  backend: soft
```

检查 soft renderer contract：

```bash
python scripts/check_renderer_backend.py --config configs/minimal.yaml --backend soft --out outputs/check_renderer_soft
```

真实 CUDA 3DGS renderer 预留接口：

```yaml
renderer:
  backend: real
  real_renderer_module: shapesplat.renderer.real_3dgs_renderer
  real_renderer_class: RealGaussianRenderer
```

也可以使用 auto fallback：

```yaml
renderer:
  backend: auto
  fallback_to_soft: true
```

如果真实 renderer 不可用，auto 模式会 fallback 到 soft renderer。真实 renderer 必须返回统一的 `RenderOutput`：

```text
rgb [3,H,W]
alpha [H,W]
depth [H,W]
contributions [N,H,W]
ownership [N,H,W]
bg_ownership [H,W]
```

`contributions` 和 `ownership` 是 scene-coupled object ownership optimization 的核心。后续真实 CUDA 3DGS renderer 必须支持 per-object contribution maps，不能只输出 RGB。

## Dataset / Batch Experiment Runner

创建 example dataset：

```bash
python scripts/create_example_dataset.py --out examples/example_dataset --num-images 4 --size 128
```

运行 batch experiment：

```bash
python scripts/run_dataset.py --config configs/dataset_minimal.yaml --manifest examples/example_dataset/manifest.csv --out outputs/dataset_run --max-images 3
```

打印 summary：

```bash
python scripts/print_metrics_table.py --summary outputs/dataset_run/summary.json
```

输出结构：

```text
outputs/dataset_run/
  image_id/
    input.png
    masks.png
    render_final.png
    alpha_final.png
    ownership_argmax.png
    metrics.json
  per_image_metrics.json
  per_image_metrics.csv
  summary.json
  summary.csv
```

当前 batch runner 仍然使用 minimal pipeline，用于验证多图运行、结果保存和 metrics 汇总。正式论文实验后续需要接真实 backend、正式数据集和 baseline。
## Same-Mask Protocol / 固定 Mask 实验协议

same-mask setting 用于公平比较 reconstruction quality、ownership 和 editing stability：所有方法共享同一组 retained visible instance masks，避免 proposal quality 干扰后续 3D 重建指标。这里的 masks 是可见实例 masks，不是 amodal masks；hidden branch 仍然只负责 plausible hidden support。

单图 file mask 运行：
```bash
python scripts/run_minimal.py --config configs/same_mask.yaml --input examples/test_image.png --mask examples/test_mask.npy --out outputs/same_mask_single --eval
```

创建带 masks 的 example dataset：
```bash
python scripts/create_example_dataset.py --out examples/example_dataset --num-images 4 --size 128
```

检查 file masks：
```bash
python scripts/check_file_masks.py --image examples/example_dataset/images/example_000.png --mask examples/example_dataset/masks/example_000.npy --config configs/minimal.yaml --out outputs/check_file_masks
```

dataset same-mask 运行：
```bash
python scripts/run_dataset.py --config configs/same_mask.yaml --manifest examples/example_dataset/manifest.csv --out outputs/same_mask_dataset --mask-source file --max-images 3
```

支持的 mask 格式：
- `.npy` stack: `[N,H,W]`
- `.npy` label map: `[H,W]`
- `.npz` with `masks` / `labels` / `instance_map`
- `.png` label map
- RGB instance PNG

`frontend.mask_source` 控制 mask 从哪里来：`sam` 使用现有 SAM backend，`file` 必须读取给定 mask，`auto` 在存在 mask 文件时用 file，否则回退到 SAM。正式论文主实验建议默认使用 same-mask protocol。

## Baseline Protocol / 基线方法输入输出协议

baseline protocol 用于 same-mask setting：所有方法共享同一张 image 和同一组 retained visible masks，从而公平比较 reconstruction、ownership 和 editing，而不是比较 proposal 质量。

导出单图 baseline inputs：
```bash
python scripts/export_baseline_inputs.py --config configs/same_mask.yaml --input examples/example_dataset/images/example_000.png --mask examples/example_dataset/masks/example_000.npy --out outputs/baseline_inputs/example_000 --image-id example_000
```

运行 dummy baselines：
```bash
python scripts/run_dummy_baselines.py --config configs/same_mask.yaml --input examples/example_dataset/images/example_000.png --mask examples/example_dataset/masks/example_000.npy --out outputs/dummy_baselines/example_000 --image-id example_000
```

批量运行：
```bash
python scripts/run_baseline_dataset.py --config configs/baseline_protocol.yaml --manifest examples/example_dataset/manifest.csv --out outputs/baseline_dataset --max-images 3 --run-dummy
```

打印表格：
```bash
python scripts/print_baseline_table.py --summary outputs/baseline_dataset/baseline_summary.json
```

真实 baseline 后续只要按协议输出以下文件，就可以被统一评估：
- `render.png`
- `alpha.png`
- `ownership.npy` 或 `object_i_alpha.png`
- `metrics.json` 可选

当前 `identity_mask`、`independent_blob`、`scene_union` 都只是 protocol smoke-test dummy baselines，不是论文正式对比方法。正式 SPAR3D / SF3D / TRELLIS / Hunyuan3D / VGGT / DUSt3R / AnySplat 接入将在后续版本实现。

## Comparison Runner / 统一对比实验

comparison runner 在 same-mask dataset 上同时运行 Ours 和 dummy baselines，并保存 per-image comparison、per-method summary 和定性网格图。

创建 same-mask example dataset：
```bash
python scripts/create_example_dataset.py --out examples/example_dataset --num-images 4 --size 128
```

运行 Ours + dummy baselines comparison：
```bash
python scripts/run_comparison.py --config configs/comparison_minimal.yaml --manifest examples/example_dataset/manifest.csv --out outputs/comparison_run --max-images 3
```

打印表格：
```bash
python scripts/print_comparison_table.py --summary outputs/comparison_run/per_method_summary.json
```

查看定性结果：
```text
outputs/comparison_run/qualitative_index.md
```

当前 dummy baselines 只是协议测试，不是论文正式 baseline。正式 baseline 后续只需要按照 baseline output protocol 输出 `render` / `alpha` / `ownership`，即可被 comparison runner 统一评估。

## Reporting and Diagnostics / 实验报告与诊断

Reporting 工具用于把 comparison / ablation / dataset / baseline 的零散 JSON/CSV 输出整理成实验报告、论文表格草稿和 failure diagnostics。它不是新的算法模块。

先运行 comparison：
```bash
python scripts/run_comparison.py --config configs/comparison_minimal.yaml --manifest examples/example_dataset/manifest.csv --out outputs/comparison_run --max-images 3
```

生成报告：
```bash
python scripts/generate_report.py --root outputs/comparison_run --out outputs/comparison_run/report --title "ShapeSplat++ Comparison Report"
```

生成 LaTeX 表格：
```bash
python scripts/make_latex_table.py --input outputs/comparison_run/per_method_summary.json --out outputs/comparison_run/report/tables/comparison_table.tex --caption "Comparison on the same-mask setting." --label tab:same_mask_comparison --kind comparison
```

诊断 metrics：
```bash
python scripts/diagnose_metrics.py --metrics outputs/comparison_run/per_image_comparison.json --out outputs/comparison_run/report/diagnostics --metric AttrAcc --top-k 5
```

主要输出：
- `report.md`：实验报告；
- `tables/*.tex`：论文表格草稿；
- `diagnostics/failure_cases.json`：失败案例；
- `qualitative/qualitative_index.md`：定性结果索引。

这些工具用于整理结果和发现 failure modes；正式论文表格仍需要人工检查和排版。

## External Baseline Adapter / 外部基线方法适配器

external baseline adapter 是后续接入 SPAR3D / SF3D / TRELLIS / Hunyuan3D / VGGT / DUSt3R / AnySplat 等方法的统一入口。当前版本只提供接口、dummy adapter、command template、输出验证和 dry-run，不接真实 baseline、不下载模型。

列出可用 adapters：
```bash
python scripts/list_baseline_adapters.py
```

验证 baseline output：
```bash
python scripts/validate_baseline_output.py --output outputs/dummy_baselines/example_000/identity_mask --num-objects 2 --strict
```

运行 dummy external adapter：
```bash
python scripts/run_external_baseline.py --config configs/same_mask.yaml --external-config configs/external_baselines.yaml --adapter dummy_external --input examples/example_dataset/images/example_000.png --mask examples/example_dataset/masks/example_000.npy --out outputs/external_baseline/example_000/dummy_external --image-id example_000
```

批量运行 dummy external adapter：
```bash
python scripts/run_external_baseline_dataset.py --config configs/same_mask.yaml --external-config configs/external_baselines.yaml --manifest examples/example_dataset/manifest.csv --adapter dummy_external --out outputs/external_baseline_dataset --max-images 3
```

`command_template` 示例：
```yaml
external_baselines:
  - name: my_method
    adapter: command_template
    enabled: true
    command: "python external_methods/my_method/run.py --image {image} --masks {masks} --out {output_dir}"
```

真实 baseline 后续只需要写 adapter 或 command template，并输出符合 baseline output protocol 的文件：`render.png`、`alpha.png`、`ownership.npy` 或 `object_i_alpha.png`。
## Reproducibility and Run Registry / 可复现实验追踪

每次主要实验入口默认会在输出目录写入轻量元数据：`run_info.json`、`config_resolved.yaml`、`environment.json`、`command.txt`、`output_index.json`、`metrics_summary.json` 和 `file_hashes.json`。这些文件用于记录命令、配置快照、环境、关键指标和输出哈希，方便论文实验复查。

手动补充已有输出目录：

```bash
python scripts/finalize_run.py --out outputs/comparison_run --config configs/comparison_minimal.yaml --run-type comparison --manifest examples/example_dataset/manifest.csv --status success
```

查看历史 runs：

```bash
python scripts/list_runs.py --registry runs/run_registry.jsonl --max-rows 20
```

比较两个 run：

```bash
python scripts/compare_runs.py --run-a outputs/run_a --run-b outputs/run_b --out outputs/run_compare
```

这是轻量级实验追踪工具，不替代 MLflow/W&B；主要用于 ShapeSplat++ minimal 论文实验的可复现记录和调试。

## Stress Benchmark / 遮挡与编辑稳定性压力测试

创建 synthetic stress dataset：

```bash
python scripts/create_stress_dataset.py --out examples/stress_dataset --num-per-subset 4 --size 128
```

运行 stress benchmark：

```bash
python scripts/run_stress_benchmark.py --config configs/stress_benchmark.yaml --manifest examples/stress_dataset/manifest.csv --out outputs/stress_benchmark --max-images 12
```

打印 subset 表格：

```bash
python scripts/print_stress_table.py --summary outputs/stress_benchmark/stress_subset_summary.json
```

支持 subset：`normal`、`light_occlusion`、`heavy_occlusion`、`same_category`、`contact_heavy`、`truncation`、`scale_variation`、`small_object`、`mixed`。

诊断指标包括 `SwapRateProxy`、`OrderAccProxy`、`OcclusionRecallProxy`、`TruncationStabilityProxy`。这些是 synthetic diagnostic metrics，不是最终真实 3D reconstruction metrics；它们用于调试 visible-hidden split、ownership rendering 和 edit consistency。

## Object Editing Suite / 物体级编辑评估

单图编辑 demo：

```bash
python scripts/run_edit_demo.py --config configs/editing.yaml --input examples/example_dataset/images/example_000.png --mask examples/example_dataset/masks/example_000.npy --out outputs/edit_demo --object-id 0
```

打印编辑指标：

```bash
python scripts/print_edit_table.py --summary outputs/edit_demo/edit_summary.json
```

批量编辑评估：

```bash
python scripts/run_edit_dataset.py --config configs/editing.yaml --manifest examples/example_dataset/manifest.csv --out outputs/edit_dataset --max-images 3 --max-objects 2
```

支持编辑操作：`remove`、`translate`、`scale`、`isolate`、`object_only`。

主要输出包括 `original_render.png`、`edited_render.png`、`diff_heatmap.png`、`edit_region.png`、`non_edit_region.png`、`object_alpha_before.png`、`object_alpha_after.png`、`edit_metrics.json` 和 `edit_summary.json`。

指标说明：`CollateralL1` 衡量非编辑区域 RGB 变化；`AlphaCollateral` 衡量非编辑区域 alpha 变化；`EditLocality` 越高越好；`DeletionResidual` 衡量删除后原物体区域残留；`ObjectSupportIoU` 衡量编辑后 object support 与原 mask 的一致性。这些是 minimal editing diagnostics；正式论文中可以替换 CollateralL1 为 Collateral LPIPS。
## Experiment Orchestration / 统一实验入口

查看可用 presets：

```bash
python scripts/list_presets.py
```

检查实验是否就绪：

```bash
python scripts/check_experiment_ready.py --preset comparison --out outputs/check_ready
```

dry run：

```bash
python scripts/run_experiment.py --preset comparison --out outputs/exp_comparison --dry-run
```

运行 comparison：

```bash
python scripts/run_experiment.py --preset comparison --out outputs/exp_comparison
```

运行 stress benchmark：

```bash
python scripts/run_experiment.py --preset stress --out outputs/exp_stress
```

运行 editing suite：

```bash
python scripts/run_experiment.py --preset editing --out outputs/exp_editing
```

运行 debug_all：

```bash
python scripts/run_experiment.py --preset debug_all --out outputs/exp_debug_all
```

输出包括 `experiment_plan.json`、`command_log.json`、`run_summary.json`、`readiness.json`、`logs/`、`run_info.json` 和 `metrics_summary.json`。orchestrator 只是统一调用已有脚本，不改变算法，用于减少实验命令复杂度。
## Real Backend and Shape Prior Pack

v2.0 adds utility code for checking optional real frontends, caching frontend outputs, and preparing file shape banks. These tools do not download checkpoints and do not require real SAM3 / DINOv3 / Depth models. The default minimal pipeline still uses stub backends.

Check a frontend backend configuration:

```bash
python scripts/check_real_frontend.py --config configs/local_real_frontend.yaml --input examples/test_image.png --out outputs/check_real_frontend --save-cache
```

Cache frontend outputs for a dataset:

```bash
python scripts/cache_frontend_outputs.py --config configs/local_real_frontend.yaml --manifest examples/example_dataset/manifest.csv --out-cache outputs/frontend_cache --max-images 3
```

Prepare a descriptor-ready toy shape bank:

```bash
python scripts/prepare_shape_bank.py --source toy --out outputs/shape_bank_prepared --num-points 512 --descriptor-dim 16 --descriptor-mode point_stats
```

Check image-to-shape retrieval:

```bash
python scripts/check_shape_retrieval.py --config configs/real_shape_bank.yaml --input examples/test_image.png --shape-root outputs/shape_bank_prepared --out outputs/check_shape_retrieval
```

Run through presets:

```bash
python scripts/run_experiment.py --preset real_backend_check --out outputs/exp_real_backend_check
python scripts/run_experiment.py --preset shape_prior_check --out outputs/exp_shape_prior_check
```

Frontend cache files include `masks.npy`, `mask_confidences.npy`, `boxes.npy`, `descriptors.npy`, `depth.npy`, `frontend_meta.json`, and optional `dino_features.pt`. Shape descriptor precompute currently supports `point_stats` and `random`; a future real paper pipeline can replace this with multi-view DINO descriptors.
## Benchmark and Baseline Pack

v2.1 adds standard same-mask benchmark validation, renderer contract checks, a runnable independent Gaussian-style baseline, and command templates for future external baselines.

Validate a benchmark manifest:

```bash
python scripts/validate_benchmark.py --manifest examples/example_dataset/manifest.csv --config configs/same_mask.yaml --out outputs/benchmark_validation
```

Build a standard same-mask benchmark directory:

```bash
python scripts/build_same_mask_benchmark.py --source-manifest examples/example_dataset/manifest.csv --out data/same_mask_example --copy-files --overwrite
```

Check the renderer contract:

```bash
python scripts/check_renderer_backend.py --config configs/minimal.yaml --backend soft --out outputs/check_renderer_soft
```

Run the independent Gaussian baseline:

```bash
python scripts/run_independent_gaussian_baseline.py --config configs/benchmark_baseline.yaml --input examples/example_dataset/images/example_000.png --mask examples/example_dataset/masks/example_000.npy --out outputs/independent_gaussian/example_000 --image-id example_000
```

Include the independent baseline in comparison:

```bash
python scripts/run_comparison.py --config configs/benchmark_baseline.yaml --manifest examples/example_dataset/manifest.csv --out outputs/comparison_with_independent --max-images 3 --run-independent-gaussian
```

List external baseline command templates:

```bash
python scripts/list_baseline_templates.py
```

`independent_gaussian` is a minimal runnable baseline. SPAR3D / TRELLIS / VGGT / DUSt3R style methods are currently templates only; real integration still requires the external repository, model files, and an adapter that writes the baseline output protocol.

## Paper Experiment Pack / 论文实验一键运行

v2.2 adds a paper-style orchestration layer for running the current smoke-test versions of the main comparison, ablations, stress benchmark, editing evaluation, baseline checks, report generation, and LaTeX table export. It does not install or run real SPAR3D / TRELLIS / VGGT style baselines.

Check whether a paper profile is ready:

```bash
python scripts/check_paper_ready.py --profile debug --out outputs/check_paper_ready
```

Dry-run the paper workflow:

```bash
python scripts/run_paper_experiments.py --profile debug --out outputs/paper_debug --dry-run
```

Run the debug paper experiments and export reports/tables:

```bash
python scripts/run_paper_experiments.py --profile debug --out outputs/paper_debug --generate-tables --generate-report
```

Export paper tables from an existing paper output directory:

```bash
python scripts/export_paper_tables.py --root outputs/paper_debug --kind all --out outputs/paper_debug/tables
```

The same workflow is also available through the experiment orchestrator:

```bash
python scripts/run_experiment.py --preset paper_debug --out outputs/exp_paper_debug
```

Typical outputs:

```text
outputs/paper_debug/
  main_comparison/
  ablation/
  stress/
  editing/
  baselines/
  tables/
  paper_report.md
  paper_run_summary.json
```

`debug` is a smoke-test profile, not a formal paper result. Optional geometry metrics run only when GT point clouds are available. The soft renderer is useful for validating the experimental protocol, but it is not a final high-quality CUDA 3DGS renderer.

## Artifact Readiness / 工程验收

v2.3 adds final artifact validation, documentation, command matrix checks, lightweight CI, cleanup helpers, and packaging utilities. These tools do not add algorithms and do not require real model checkpoints.

Run a quick project health check:

```bash
python scripts/check_project_health.py
```

Run quick tests:

```bash
python scripts/run_quick_tests.py
```

Run the full test suite:

```bash
python scripts/run_quick_tests.py --full
```

Run artifact validation from the command matrix:

```bash
python scripts/validate_artifact.py --matrix configs/command_matrix.yaml --groups quick smoke --out outputs/artifact_validation
```

Dry-run validation:

```bash
python scripts/validate_artifact.py --matrix configs/command_matrix.yaml --groups quick smoke --dry-run --out outputs/artifact_validation_dry
```

Package a lightweight artifact:

```bash
python scripts/package_artifact.py --out dist/shapesplat_minimal_artifact.zip --include-examples --include-tests --include-docs
```

Clean generated outputs:

```bash
python scripts/clean_outputs.py --dry-run
```

Lightweight CI lives at `.github/workflows/ci.yml` and runs import plus CPU pytest smoke tests. The artifact package excludes `outputs/`, `runs/`, `.git/`, checkpoints, and model weights. Real SAM3 / DINOv3 / renderer checkpoints are intentionally not bundled and must be configured separately.

## Real Integration Smoke Test / 真实组件本地集成测试

v2.4 adds a local integration smoke layer for optional real SAM3 / DINOv3 / Depth / ShapeBank / Renderer / external baseline components. It does not download models and does not require real checkpoints.

Create a local backend template:

```bash
python scripts/create_local_backend_template.py --out configs/my_local_backend.yaml
```

Check backend capability:

```bash
python scripts/check_backend_capabilities.py --config configs/my_local_backend.yaml --out outputs/backend_capabilities
```

Run the integration smoke test:

```bash
python scripts/run_real_integration_smoke.py --config configs/my_local_backend.yaml --input examples/test_image.png --out outputs/real_integration_smoke --save-cache
```

Then inspect:

```text
outputs/real_integration_smoke/integration_report.md
```

If no real models are configured, `auto` mode falls back to stub / soft backends and should still complete. This smoke test validates local integration plumbing; it is not a final paper-quality experiment.

## Frontend Cache / 前端缓存

v2.5 lets the main experiment runners reuse cached frontend outputs. This is useful because real SAM3 / DINOv3 / Depth inference can be slow, while comparison, ablation, stress, editing, and paper experiments often reuse the same masks, descriptors, and depth.

Cache an example dataset:

```bash
python scripts/cache_frontend_outputs.py --config configs/local_real_frontend.yaml --manifest examples/example_dataset/manifest.csv --out-cache outputs/frontend_cache --max-images 3 --write-manifest outputs/frontend_cache/cache_manifest.csv --update-dataset-manifest outputs/frontend_cache/manifest_with_cache.csv --validate
```

Validate the cache:

```bash
python scripts/validate_frontend_cache.py --cache-root outputs/frontend_cache --out outputs/frontend_cache_validation
```

Convert cached masks into a same-mask dataset:

```bash
python scripts/cache_to_same_mask_dataset.py --image-manifest examples/example_dataset/manifest.csv --cache-manifest outputs/frontend_cache/cache_manifest.csv --out data/cached_same_mask_dataset --copy-images --overwrite
```

Run comparison with the cached frontend outputs:

```bash
python scripts/run_comparison.py --config configs/cache_experiment.yaml --manifest data/cached_same_mask_dataset/manifest.csv --out outputs/cached_comparison --max-images 3 --use-frontend-cache --frontend-cache-manifest outputs/frontend_cache/cache_manifest.csv
```

Or use the preset:

```bash
python scripts/run_experiment.py --preset cache_frontend --out outputs/exp_cache_frontend
```

The cache stores frontend outputs only: retained visible masks, descriptors, depth, and small visualizations. It is not a trained model output, and Gaussian optimization still runs normally.

## Real 3DGS Renderer Adapter / 真实 3D Gaussian Renderer 适配

The default renderer is still the CPU-friendly soft renderer:

```yaml
renderer:
  backend: soft
```

For local integration checks you can use auto mode. It tries a real 3DGS adapter first, then falls back to soft when no CUDA renderer is installed:

```yaml
renderer:
  backend: auto
  fallback_to_soft: true
  real_3dgs:
    library: auto
```

Check the real renderer adapter and fallback path:

```bash
python scripts/check_real_renderer.py --config configs/real_3dgs_renderer.yaml --backend auto --out outputs/check_real_renderer --allow-fallback
```

A real renderer must return the standard `RenderOutput`: `rgb`, `alpha`, `depth`, `contributions`, `ownership`, and `bg_ownership`. If the CUDA renderer cannot provide native object contributions, the adapter reserves an object-wise alpha fallback strategy: render each object separately, stack `alpha_n`, and normalize ownership maps. This is slower, but keeps ShapeSplat++ ownership supervision compatible.

This project does not require `diff-gaussian-rasterization` or `gsplat`; users can add those locally later.

## Final Benchmark Format / 正式 Benchmark 协议

v3.0 adds a benchmark manifest v2 format for paper-ready same-mask experiments:

```text
benchmark_root/
  images/
  masks/
  metadata/
  depth/                 optional
  cameras/               optional
  gt_pointclouds/         optional
  gt_meshes/              optional
  frontend_cache/         optional
  manifest.csv
  splits.json
  benchmark_info.json
```

Required manifest fields are `image_id`, `image_path`, `mask_path`, and `split`. Optional fields include metadata, source dataset, GT geometry/depth/camera paths, frontend cache bindings, and diagnostic tags such as occlusion or same-category flags.

Convert a generic folder:

```bash
python scripts/convert_benchmark.py --converter generic_folder --src examples/example_dataset --out data/example_benchmark_v2 --source-dataset example --overwrite
```

Validate benchmark v2:

```bash
python scripts/validate_benchmark_v2.py --manifest data/example_benchmark_v2/manifest.csv --config configs/final_benchmark.yaml --out outputs/benchmark_v2_validation
```

Bind frontend cache:

```bash
python scripts/bind_cache_to_benchmark.py --manifest data/example_benchmark_v2/manifest.csv --cache-manifest outputs/frontend_cache/cache_manifest.csv --out-manifest data/example_benchmark_v2/manifest_with_cache.csv
```

Summarize:

```bash
python scripts/summarize_benchmark.py --manifest data/example_benchmark_v2/manifest.csv --out outputs/benchmark_summary
```

GSO / Objaverse / Pix3D / CO3D converters are templates for local adaptation. Formal paper experiments should use fixed manifests and fixed visible masks.

## Ours Reconstruction Core Runner

The Ours core runner runs ShapeSplat++ directly on a benchmark v2 manifest and saves outputs that are compatible with the baseline protocol.

Check readiness:

```powershell
python scripts/check_ours_core_ready.py --config configs/final_ours.yaml --manifest data/example_benchmark_v2/manifest.csv --out outputs/check_ours_core_ready
```

Run Ours on a benchmark:

```powershell
python scripts/run_ours_benchmark.py --config configs/final_ours.yaml --manifest data/example_benchmark_v2/manifest.csv --out outputs/ours_benchmark --max-images 3
```

Run Ours variants:

```powershell
python scripts/run_ours_variants.py --config configs/final_ours.yaml --variants configs/ours_variants.yaml --manifest data/example_benchmark_v2/manifest.csv --out outputs/ours_variants --variant full --variant visible_only --max-images 3
```

Use frontend cache:

```powershell
python scripts/run_ours_benchmark.py --config configs/final_ours.yaml --manifest data/example_benchmark_v2/manifest_with_cache.csv --out outputs/ours_benchmark_cached --use-frontend-cache --frontend-cache-manifest outputs/frontend_cache_v2/cache_manifest.csv --max-images 3
```

Notes:
- If the run still uses stub / toy / soft fallback, the result is only for debugging.
- Final submission experiments should run `--strict-ready` and explicitly configure real frontend cache, prepared shape bank, and renderer settings.
- Ours outputs include `render_final.png`, `alpha_final.png`, `ownership.npy`, `metrics.json`, `output_spec.json`, `reconstruction_meta.json`, and `diagnostics.json`.

## Final Baseline and Geometry Metrics

Run Ours and export a lightweight predicted pointcloud:

```powershell
python scripts/run_ours_benchmark.py --config configs/final_ours.yaml --manifest data/example_benchmark_v2/manifest.csv --out outputs/ours_benchmark --max-images 3
```

Evaluate one method output root:

```powershell
python scripts/evaluate_method_outputs.py --method ours_full --outputs outputs/ours_benchmark --manifest data/example_benchmark_v2/manifest.csv --config configs/final_ours.yaml --out outputs/eval_ours_full --max-images 3
```

Run final comparison:

```powershell
python scripts/run_final_comparison.py --manifest data/example_benchmark_v2/manifest.csv --methods configs/method_catalog.yaml --outputs-config configs/final_method_outputs.yaml --config configs/final_ours.yaml --out outputs/final_comparison --max-images 3
```

Export final tables:

```powershell
python scripts/export_final_tables.py --summary outputs/final_comparison/final_method_summary.json --out outputs/final_comparison/tables
```

Notes:
- Chamfer/F-score are computed only when both `pred_pointcloud.npy` and manifest GT pointcloud fields exist.
- Real-image diagnostics do not report geometry without GT.
- Real external baselines are disabled in `configs/method_catalog.yaml` until users provide outputs that satisfy the baseline protocol.

## Final Paper Experiments / 最终论文实验

The final paper runner is a top-level orchestration layer. It validates the benchmark, runs Ours, variants, internal baselines, final comparison, stress/editing checks, table export, report generation, and readiness summaries.

Check final readiness:

```powershell
python scripts/check_final_paper_ready.py --profile configs/paper/final_debug.yaml --out outputs/check_final_ready
```

Dry run the final plan:

```powershell
python scripts/run_final_paper.py --profile configs/paper/final_debug.yaml --out outputs/final_paper_debug --dry-run
```

Run the final debug pipeline:

```powershell
python scripts/run_final_paper.py --profile configs/paper/final_debug.yaml --out outputs/final_paper_debug --generate-tables --generate-report
```

Export all final tables:

```powershell
python scripts/export_all_final_tables.py --root outputs/final_paper_debug --out outputs/final_paper_debug/tables
```

Generate the final report:

```powershell
python scripts/generate_final_report.py --root outputs/final_paper_debug --out outputs/final_paper_debug/report --title "ShapeSplat++ Final Paper Report"
```

Run through the experiment preset:

```powershell
python scripts/run_experiment.py --preset final_paper_debug --out outputs/exp_final_paper_debug
```

Notes:
- `final_debug` is a smoke test, not a submission result.
- Submission runs should replace the example benchmark, frontend cache, shape bank, renderer, and external baseline output paths with real experiment artifacts.
- `--strict-ready` is intended to catch remaining stub / toy / soft fallback settings before final reporting.

## Windows + RTX 5090 GPU Runtime

Activate the conda environment first:

```powershell
conda activate shapesplat
```

Print PyTorch / CUDA / GPU information:

```powershell
python scripts/print_gpu_info.py
```

Check CUDA runtime compatibility:

```powershell
python scripts/check_gpu_runtime.py --config configs/local_windows_rtx5090.yaml --device cuda --require-cuda --out outputs/check_gpu_runtime
```

Run a tiny GPU smoke experiment:

```powershell
python scripts/run_gpu_smoke_experiment.py --config configs/local_windows_rtx5090.yaml --out outputs/gpu_smoke --require-cuda --iters 2
```

Run the PowerShell one-shot debug:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_windows_gpu_paper_debug.ps1
```

Notes:
- The project does not install PyTorch, CUDA, or NVIDIA drivers automatically.
- RTX 5090 needs a PyTorch CUDA build that supports the GPU architecture. `torch.cuda.is_available()` is not enough; `check_gpu_runtime.py` runs a real CUDA matmul/backward smoke test.
- If `--device cuda` or `--require-cuda` is used, CUDA failure is reported clearly. CPU fallback happens only when `--allow-cpu-fallback` or `runtime.allow_cpu_fallback=true` is explicit.
- `renderer.backend=auto` can still fallback to the soft renderer; this is reported separately from CUDA device availability.

## CO3Dv2 High-Resolution Diagnostics

Run from Anaconda Prompt:

```bat
conda activate shapesplat
cd /d C:\Users\zhaoc\ShapeSplat\shapesplat
```

Do not use the old minimal `128x128` config for CO3Dv2 diagnostics. CO3Dv2 images and file masks are around `640x479`, so the high-resolution workflow uses keep-aspect resizing, nearest mask resize, DINOv3 ViT-S/16 dense descriptors, and explicit resolution diagnostics.

Check readiness:

```bat
python scripts/check_co3dv2_highres_ready.py --config configs/final_ours_co3dv2_highres.yaml --manifest data/co3dv2_single_benchmark/manifest.csv --cache-manifest outputs/cache_co3dv2_real_frontend_vits16_highres/cache_manifest.csv --out outputs/check_co3dv2_highres_ready
```

Rebuild high-resolution frontend cache:

```bat
python scripts/cache_co3dv2_highres_frontend.py --config configs/co3dv2_real_frontend_highres.yaml --manifest data/co3dv2_single_benchmark/manifest.csv --out-cache outputs/cache_co3dv2_real_frontend_vits16_highres --max-images 5 --write-manifest outputs/cache_co3dv2_real_frontend_vits16_highres/cache_manifest.csv --validate --require-cuda --check-deps-first
```

Run high-resolution Ours:

```bat
python scripts/run_co3dv2_highres_ours.py --config configs/final_ours_co3dv2_highres.yaml --manifest data/co3dv2_single_benchmark/manifest.csv --out outputs/ours_co3dv2_vits16_highres --max-images 5 --frontend-cache-manifest outputs/cache_co3dv2_real_frontend_vits16_highres/cache_manifest.csv
```

Run high-resolution variants:

```bat
python scripts/run_co3dv2_highres_variants.py --config configs/final_ours_co3dv2_highres.yaml --manifest data/co3dv2_single_benchmark/manifest.csv --out outputs/ours_variants_co3dv2_vits16_highres --variant full --variant visible_only --max-images 5 --frontend-cache-manifest outputs/cache_co3dv2_real_frontend_vits16_highres/cache_manifest.csv
```

Inspect output resolution:

```bat
python scripts/inspect_output_resolution.py --out outputs/ours_co3dv2_vits16_highres --max-items 5
```

Generate a diagnostic report:

```bat
python scripts/generate_co3dv2_highres_report.py --root outputs/ours_co3dv2_vits16_highres --out outputs/ours_co3dv2_vits16_highres/report
```

CO3Dv2 single is a real-image diagnostic / single foreground visible-mask benchmark, not the multi-object main benchmark. If readiness or diagnostics show ToyShapeBank or SoftGaussianRenderer fallback, the run is useful for engineering diagnostics but not paper-final 3DGS quality.
