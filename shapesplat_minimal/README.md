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
