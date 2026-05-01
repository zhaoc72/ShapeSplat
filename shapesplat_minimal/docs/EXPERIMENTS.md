# Experiments

ShapeSplat++ 当前实验结构围绕 same-mask setting 展开：所有方法共享同一张图像和同一组 retained visible instance masks，从而减少 proposal quality 对 reconstruction / ownership / editing 指标的干扰。

## Main Comparison

`run_comparison.py` 运行 Ours、dummy baselines 和可选 independent Gaussian baseline，输出 per-image comparison 和 per-method summary。

## Ablation

`run_ablation.py` 使用 `configs/ablations.yaml` 切换 loss/branch 配置，用于检查模块贡献。

## Stress Benchmark

stress dataset 覆盖 occlusion、same-category、contact-heavy、truncation、scale variation 和 small-object 场景。它是 synthetic diagnostic，不是真实最终 benchmark。

## Editing Benchmark

editing suite 对 object buffers 执行 remove/translate/scale/isolate/object_only，并统计 CollateralL1、EditLocality、DeletionResidual 等指标。

## Baseline Protocol

baseline input/output protocol 规定 image、masks、crops、render、alpha、ownership 等文件格式，方便后续接入外部方法。

## Metrics Groups

指标分为 object consistency、rendering、editing、stress 和 optional geometry。geometry 只有预测点云和 GT 点云都存在时才启用。

## Output Structure

常见输出包括 `metrics.json`、`summary.json`、`per_method_summary.json`、`stress_subset_summary.json`、`edit_summary.json`、`report.md` 和 `tables/*.tex`。

dummy baselines 和 toy datasets 只用于 smoke tests，不是论文最终实验结果。

