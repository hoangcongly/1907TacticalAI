import ast
import os
import sys

def status(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            tree = ast.parse(f.read())
    except Exception:
        return "ERROR"
        
    defs = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))]
    if not defs:
        return "STUB"
        
    raises_ni = any(
        isinstance(n, ast.Raise) and getattr(getattr(n.exc, 'func', None), 'id', '') == 'NotImplementedError'
        for n in ast.walk(tree)
    )
    return "NOT_IMPLEMENTED" if raises_ni and len(defs) <= 1 else "IMPLEMENTED"

def main():
    src_dir = os.path.join(os.path.dirname(__file__), "..", "src")
    report_lines = ["# Bảng Kiểm Kê Trạng Thái Module\n\n| Module | Trạng Thái |\n|---|---|"]
    
    for root, _, files in os.walk(src_dir):
        for file in sorted(files):
            if file.endswith(".py") and file != "__init__.py":
                path = os.path.join(root, file)
                rel_path = os.path.relpath(path, start=os.path.dirname(src_dir))
                mod_status = status(path)
                report_lines.append(f"| `{rel_path}` | {mod_status} |")
                
    report_content = "\n".join(report_lines) + "\n"
    report_path = os.path.join(os.path.dirname(__file__), "..", "docs", "module_status.md")
    
    if "--check" in sys.argv:
        if not os.path.exists(report_path):
            print("docs/module_status.md không tồn tại.")
            sys.exit(1)
        with open(report_path, "r", encoding="utf-8") as f:
            existing_content = f.read()
        if existing_content != report_content:
            print("FAIL: docs/module_status.md chưa được cập nhật. Hãy chạy scripts/audit_module_status.py")
            sys.exit(1)
        else:
            print("OK: docs/module_status.md đã cập nhật.")
            sys.exit(0)
    else:
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report_content)
        print("Đã tạo docs/module_status.md thành công.")

if __name__ == "__main__":
    main()
