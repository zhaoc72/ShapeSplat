# Examples

这里用于存放用户自己的测试图片。当前项目不会提交大图或真实数据。

你可以将图片命名为 `test_image.png`，然后运行：

```bash
python scripts/run_minimal.py --config configs/minimal.yaml --input examples/test_image.png --out outputs/real_input
```

如果没有真实图片，可以先生成一张 synthetic-realistic 示例图：

```bash
python scripts/create_example_image.py --out examples/test_image.png --size 128
```
