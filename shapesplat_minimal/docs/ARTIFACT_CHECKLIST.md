# Artifact Checklist

- [ ] `python -c "import shapesplat; print('import ok')"` passes.
- [ ] `python scripts/check_project_health.py` passes.
- [ ] minimal demo runs.
- [ ] real input demo runs.
- [ ] same-mask single-image run works.
- [ ] dataset runner works.
- [ ] comparison runner works.
- [ ] stress benchmark works.
- [ ] editing demo works.
- [ ] report generation works.
- [ ] `python scripts/run_quick_tests.py` passes.
- [ ] run registry can be written.
- [ ] outputs can be cleaned or regenerated.
- [ ] artifact package excludes `outputs/`, `runs/`, checkpoints, and model weights.
- [ ] README points to current commands.
- [ ] `.github/workflows/ci.yml` runs lightweight CPU tests.

当前 artifact 不包含真实模型 checkpoint；真实 SAM3 / DINOv3 / Depth / renderer 需要用户另行配置。
