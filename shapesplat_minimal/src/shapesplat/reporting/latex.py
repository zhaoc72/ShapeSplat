from __future__ import annotations

from pathlib import Path


def escape_latex(text: str) -> str:
    """转义 LaTeX 特殊字符。"""

    repl = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(repl.get(ch, ch) for ch in str(text))


def _fmt(value, float_format: str) -> str:
    try:
        if isinstance(value, str) and value.strip() == "":
            return ""
        return format(float(value), float_format)
    except (TypeError, ValueError):
        return escape_latex(str(value))


def make_latex_table(
    rows: list[dict],
    columns: list[str],
    caption: str,
    label: str,
    float_format: str = ".4f",
) -> str:
    """生成简单 LaTeX table 草稿。

    使用普通 hline，避免依赖 booktabs；正式论文排版时可手动调整。
    """

    align = "l" + "c" * max(0, len(columns) - 1)
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        rf"\caption{{{escape_latex(caption)}}}",
        rf"\label{{{escape_latex(label)}}}",
        rf"\begin{{tabular}}{{{align}}}",
        r"\hline",
        " & ".join(escape_latex(c) for c in columns) + r" \\",
        r"\hline",
    ]
    for row in rows:
        lines.append(" & ".join(_fmt(row.get(c, ""), float_format) for c in columns) + r" \\")
    lines.extend([r"\hline", r"\end{tabular}", r"\end{table}"])
    return "\n".join(lines)


def save_latex_table(rows: list[dict], columns: list[str], caption: str, label: str, path: str | Path) -> None:
    """保存 LaTeX table 文件。"""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(make_latex_table(rows, columns, caption, label), encoding="utf-8")

