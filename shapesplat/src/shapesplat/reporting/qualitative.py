from __future__ import annotations

from pathlib import Path


def find_qualitative_grids(root: str | Path) -> list[Path]:
    """查找常见定性结果图片。"""

    root = Path(root)
    names = ["qualitative_grid.png", "ownership_argmax.png", "render_final.png", "alpha_final.png"]
    paths: list[Path] = []
    for name in names:
        paths.extend(sorted(root.glob(f"**/{name}")))
    # 去重并保持顺序。
    seen = set()
    unique = []
    for p in paths:
        if p not in seen:
            unique.append(p)
            seen.add(p)
    return unique


def make_qualitative_markdown(grids: list[Path], out_path: str | Path, title: str = "Qualitative Results") -> None:
    """生成定性结果 Markdown 索引。"""

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    root = out_path.parent
    lines = [f"# {title}", ""]
    for grid in grids:
        name = grid.parent.name
        try:
            rel = grid.relative_to(root).as_posix()
        except ValueError:
            rel = grid.as_posix()
        lines.extend([f"## {name}", f"![]({rel})", ""])
    out_path.write_text("\n".join(lines), encoding="utf-8")


def make_selected_cases_markdown(cases: list[dict], root: str | Path, out_path: str | Path) -> None:
    """为 best/worst/failure cases 生成 Markdown。

    如果 case 有 image_id，会尝试链接 per_image/{image_id}/qualitative_grid.png。
    """

    root = Path(root)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# Selected Cases", ""]
    for case in cases:
        image_id = case.get("image_id", "unknown")
        method = case.get("method", "")
        lines.append(f"## {image_id} {method}".strip())
        grid = root / "per_image" / str(image_id) / "qualitative_grid.png"
        if grid.exists():
            try:
                rel = grid.relative_to(out_path.parent).as_posix()
            except ValueError:
                rel = grid.as_posix()
            lines.append(f"![]({rel})")
        lines.append("```json")
        lines.append(str(case))
        lines.append("```")
        lines.append("")
    out_path.write_text("\n".join(lines), encoding="utf-8")

