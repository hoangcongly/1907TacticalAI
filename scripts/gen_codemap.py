#!/usr/bin/env python3
"""
Sinh bản đồ code cho AI agent: .claude/skills/aegis-nav/references/codemap.md

Chạy lại sau mỗi lần refactor để bản đồ không bao giờ lỗi thời:
    python scripts/gen_codemap.py

Thiết kế tối ưu token: 1 dòng / 1 symbol, không văn xuôi, có sẵn số dòng để
agent nhảy thẳng tới đích thay vì phải grep dò dẫm.
"""
import ast
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
SRC = ROOT / "src" / "aegis"
OUT = ROOT / ".claude" / "skills" / "aegis-nav" / "references" / "codemap.md"

# File <= ngưỡng này coi như chưa triển khai (chỉ có docstring).
STUB_MAX_LINES = 3


def first_doc_line(node) -> str:
    """Dòng đầu docstring, cắt ngắn — đủ để agent quyết định có mở file không."""
    doc = ast.get_docstring(node)
    if not doc:
        return ""
    line = doc.strip().splitlines()[0].strip()
    return line[:78]


def scan(path: pathlib.Path):
    text = path.read_text(encoding="utf-8")
    n_lines = len(text.splitlines())
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return n_lines, "", []

    module_doc = first_doc_line(tree)
    symbols = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name.startswith("_"):
                continue
            symbols.append(("def", node.name, node.lineno, first_doc_line(node)))
        elif isinstance(node, ast.ClassDef):
            symbols.append(("class", node.name, node.lineno, first_doc_line(node)))
            for sub in node.body:
                if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if sub.name.startswith("_") and sub.name != "__init__":
                        continue
                    symbols.append(("  .", sub.name, sub.lineno, ""))
    return n_lines, module_doc, symbols


def main() -> int:
    files = sorted(SRC.rglob("*.py"))
    stubs, mapped = [], []

    for path in files:
        rel = path.relative_to(ROOT)
        n_lines, module_doc, symbols = scan(path)
        if path.name == "__init__.py" and n_lines <= STUB_MAX_LINES:
            continue
        if n_lines <= STUB_MAX_LINES or not symbols:
            stubs.append((str(rel), n_lines, module_doc))
            continue
        mapped.append((str(rel), n_lines, module_doc, symbols))

    out = [
        "# Aegis codemap (SINH TỰ ĐỘNG — đừng sửa tay)",
        "",
        "Sinh lại: `python scripts/gen_codemap.py`",
        "",
        "Cách dùng: tìm symbol -> đọc `file:line` -> mở thẳng bằng `sed -n 'START,+40p' file`.",
        "",
        "---",
        "",
        f"## 1. CHƯA TRIỂN KHAI ({len(stubs)} file — chỉ có docstring/stub)",
        "",
        "Không có logic. Đừng phí token đọc; đây là danh sách việc phải làm.",
        "",
        "| file | dòng | ý định |",
        "|---|---|---|",
    ]
    for rel, n, doc in stubs:
        out.append(f"| `{rel}` | {n} | {doc or '—'} |")

    out += ["", "---", "", f"## 2. ĐÃ TRIỂN KHAI ({len(mapped)} file)", ""]
    for rel, n, module_doc, symbols in mapped:
        out.append(f"### `{rel}` ({n} dòng)")
        if module_doc:
            out.append(f"_{module_doc}_")
        out.append("")
        for kind, name, lineno, doc in symbols:
            suffix = f" — {doc}" if doc else ""
            out.append(f"- `{kind} {name}` :{lineno}{suffix}")
        out.append("")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(out) + "\n", encoding="utf-8")

    total = sum(n for _, n, _, _ in mapped) + sum(n for _, n, _ in stubs)
    print(f"[codemap] {len(mapped)} file có logic, {len(stubs)} stub, {total} dòng -> {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
