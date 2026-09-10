#!/usr/bin/env python3
"""
Chống mục tài liệu: kiểm tra mọi tham chiếu `file:line` trong .claude/ còn hợp lệ,
và mọi marker [FIX Fxx] mà defects.md tuyên bố "đã vá" thực sự tồn tại trong src/.

    python scripts/check_docs.py     # exit 1 nếu có tham chiếu hỏng

Chạy sau mỗi lần refactor, cùng lúc với scripts/gen_codemap.py.
"""
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
DOCS = sorted((ROOT / ".claude").rglob("*.md")) + [ROOT / "CLAUDE.md"]

# `path/to/file.py:123` hoặc `file.py:123` trong backtick
REF = re.compile(r"`([\w./-]+\.py):(\d+)`|(\b[\w./-]+\.py):(\d+)")
FIX = re.compile(r"\[FIX (F\d+)")

# Tiền tố ngầm định khi doc viết đường dẫn rút gọn.
PREFIXES = ["", "src/aegis/", "src/"]


def resolve(rel: str) -> pathlib.Path | None:
    for prefix in PREFIXES:
        candidate = ROOT / (prefix + rel)
        if candidate.is_file():
            return candidate
    # Dự phòng: doc chỉ ghi tên file trần -> dò trong src/, chỉ chấp nhận khi duy nhất.
    matches = [m for m in (ROOT / "src").rglob(pathlib.PurePath(rel).name) if m.is_file()]
    return matches[0] if len(matches) == 1 else None


def main() -> int:
    errors: list[str] = []
    checked = 0

    for doc in DOCS:
        if not doc.is_file() or "codemap.md" in doc.name:
            continue  # codemap sinh tự động, luôn đúng theo định nghĩa
        for lineno, line in enumerate(doc.read_text(encoding="utf-8").splitlines(), 1):
            for match in REF.finditer(line):
                rel = match.group(1) or match.group(3)
                target_line = int(match.group(2) or match.group(4))
                path = resolve(rel)
                rel_doc = doc.relative_to(ROOT)
                if path is None:
                    errors.append(f"{rel_doc}:{lineno} -> không tìm thấy file `{rel}`")
                    continue
                n = len(path.read_text(encoding="utf-8").splitlines())
                if target_line > n:
                    errors.append(
                        f"{rel_doc}:{lineno} -> `{rel}:{target_line}` vượt quá số dòng ({n})"
                    )
                checked += 1

    # Marker [FIX Fxx] mà defects.md nói đã vá phải thật sự có trong src/
    src_text = "\n".join(
        p.read_text(encoding="utf-8") for p in (ROOT / "src").rglob("*.py")
    )
    present = set(FIX.findall(src_text))

    defects = ROOT / ".claude/skills/aegis-nav/references/defects.md"
    if defects.is_file():
        for line in defects.read_text(encoding="utf-8").splitlines():
            if "✅ VÁ" not in line:
                continue
            claimed = re.search(r"\|\s*(F\d+)\s*\|", line)
            if claimed and claimed.group(1) not in present:
                errors.append(
                    f"defects.md tuyên bố {claimed.group(1)} đã vá "
                    f"nhưng không có marker [FIX {claimed.group(1)}] trong src/"
                )

    if errors:
        print(f"[check-docs] ❌ {len(errors)} tham chiếu hỏng:")
        for e in errors:
            print(f"  - {e}")
        return 1

    print(f"[check-docs] ✅ {checked} tham chiếu file:line hợp lệ, "
          f"{len(present)} marker FIX khớp ({', '.join(sorted(present))})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
