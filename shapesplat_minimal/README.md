# ShapeSplat++ Minimal

ShapeSplat++ Minimal 是一个最小可运行的单图多前景物体 3D Gaussian 重建框架。当前目标不是追求真实重建效果，而是把完整 pipeline 稳定跑通，并为后续替换真实 SAM3、DINOv3、深度模型、真实 shape bank 和 CUDA 3D Gaussian renderer 留出清晰接口。

## 最小技术路线

SAM3-DINOv3 frozen front-end + visible-hidden Gaussian buffers + scene-coupled ownership rendering + confidence-weighted hidden support prior + differentiable edit-consistency optimization。

当前版本全部使用 stub：

- `Sam3Stub`：用颜色阈值和 connected components 产生 retained visible masks。
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

如果没有提供 `--input`，脚本会自动生成一张 synthetic 多物体 RGB 图像。

## 输出文件

- `input.png`：输入图像。
- `masks.png`：SAM3 stub 的 visible instance masks。
- `render_final.png`：最终 soft renderer RGB。
- `alpha_final.png`：最终 alpha。
- `ownership_argmax.png`：每个像素归属哪个 object 的可视化。
- `object_0_alpha.png`, `object_1_alpha.png` 等：per-object ownership/alpha-like map。
- `loss_log.json`：训练损失日志。
- `checkpoint_minimal.pt`：最小 checkpoint。

## 后续真实模块替换路线

- `Sam3Stub` -> `RealSAM3Wrapper`
- `DinoV3Stub` -> `RealDINOv3Wrapper`
- `DepthStub` -> `DepthAnythingWrapper`
- `ToyShapeBank` -> `RealShapeBank`
- `SoftGaussianRenderer` -> `CUDA3DGSRenderer`

替换时建议保持当前 wrapper 方法签名和输出字段不变，尤其是 `MaskSet`、instance descriptors、weak depth、per-object contributions/ownership。

## 实验注意

主实验应采用 same-mask setting，保证和 baselines 公平：不同方法使用相同 visible masks，避免 segmentation 质量差异混入 3D reconstruction/editability 的比较。
