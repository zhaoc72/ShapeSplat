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
