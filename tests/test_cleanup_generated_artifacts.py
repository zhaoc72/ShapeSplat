from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from shapesplat.cleanup.generated_artifacts import (
    delete_candidates_permanently,
    move_candidates_to_trash,
    scan_generated_artifacts,
)
from shapesplat.cleanup.rules import load_cleanup_rules


def _make_root(name: str) -> Path:
    root = Path("outputs/test_cleanup_generated_artifacts") / name
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)
    return root.resolve()


def _rules() -> dict:
    rules = load_cleanup_rules(None)
    rules["protected_paths"] = ["src", "configs", "tests", "docs", "README.md", "agents.md", "data/co3dv2_single_benchmark"]
    return rules


def test_scan_generated_artifacts():
    root = _make_root("scan")
    (root / "__pycache__").mkdir()
    (root / "__pycache__" / "x.pyc").write_text("x", encoding="utf-8")
    (root / ".pytest_cache").mkdir()
    (root / "outputs" / "minimal").mkdir(parents=True)
    candidates = scan_generated_artifacts(root, _rules())
    paths = {c.path for c in candidates}
    assert "__pycache__" in paths
    assert ".pytest_cache" in paths
    assert "outputs/minimal" in paths


def test_protected_paths_not_candidates():
    root = _make_root("protected")
    (root / "src").mkdir()
    (root / "src" / "file.py").write_text("print(1)", encoding="utf-8")
    (root / "configs").mkdir()
    (root / "configs" / "a.yaml").write_text("x: 1", encoding="utf-8")
    (root / "tests").mkdir()
    (root / "tests" / "test_x.py").write_text("def test_x(): pass", encoding="utf-8")
    candidates = scan_generated_artifacts(root, _rules())
    assert all(not c.path.startswith(("src/", "configs/", "tests/")) for c in candidates)


def test_dry_run_does_not_delete():
    root = _make_root("dry")
    pycache = root / "__pycache__"
    pycache.mkdir()
    candidates = scan_generated_artifacts(root, _rules())
    move_candidates_to_trash(candidates, root / ".trash", execute=False)
    assert pycache.exists()


def test_execute_moves_to_trash():
    root = _make_root("execute")
    pycache = root / "__pycache__"
    pycache.mkdir()
    (pycache / "x.pyc").write_text("x", encoding="utf-8")
    candidates = scan_generated_artifacts(root, _rules())
    old_cwd = Path.cwd()
    try:
        # 中文注释：move 函数保持相对路径结构，因此测试在假项目根目录下执行。
        import os

        os.chdir(root)
        report = move_candidates_to_trash(candidates, ".trash/generated_artifacts", execute=True)
    finally:
        os.chdir(old_cwd)
    assert not pycache.exists()
    assert any(Path(m["dst"]).exists() for m in report["mappings"])


def test_permanent_requires_confirm():
    root = _make_root("confirm")
    pycache = root / "__pycache__"
    pycache.mkdir()
    candidates = scan_generated_artifacts(root, _rules())
    with pytest.raises(ValueError, match="DELETE_GENERATED_ARTIFACTS"):
        delete_candidates_permanently(candidates, execute=True, require_confirm_token=None)
