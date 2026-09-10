#!/bin/bash
echo "=== BẮT ĐẦU QUÁ TRÌNH BIÊN DỊCH LOCAL ==="

# Kiểm tra xem xelatex đã được cài đặt chưa, nếu chưa thì tự động cài đặt
if ! command -v xelatex &> /dev/null; then
    echo "Phát hiện hệ thống chưa có LaTeX trong PATH."
    # Cập nhật biến môi trường PATH mặc định của MacTeX/BasicTeX
    if [ -x "/usr/libexec/path_helper" ]; then
        eval "$(/usr/libexec/path_helper)"
    fi
    export PATH="/Library/TeX/texbin:$PATH"
fi

if ! command -v xelatex &> /dev/null; then
    echo "Vẫn chưa tìm thấy xelatex. Đang tự động cài đặt BasicTeX qua Homebrew..."
    brew install --cask basictex
    export PATH="/Library/TeX/texbin:$PATH"
    
    echo "Đang tải các gói LaTeX phụ thuộc..."
    sudo tlmgr update --self
    sudo tlmgr install pdflscape amsmath ulem hyperref xcolor booktabs graphicx fancyhdr unicode-math lualatex-math setspace indentfirst
fi

echo "Đang tải Sơ đồ kiến trúc (Mermaid)..."
curl -s -o architecture.png "https://mermaid.ink/img/Zmxvd2NoYXJ0IFRECiAgICBjbGFzc0RlZiBjb21wbGV0ZWQgZmlsbDojMWEzYzIyLHN0cm9rZTojNTBmYTdiLHN0cm9rZS13aWR0aDoycHgsY29sb3I6IzUwZmE3YgogICAgY2xhc3NEZWYgaW5wcm9ncmVzcyBmaWxsOiMzNDI0MWEsc3Ryb2tlOiNmZmI4NmMsc3Ryb2tlLXdpZHRoOjEuNXB4LGNvbG9yOiNmZmI4NmMsc3Ryb2tlLWRhc2hhcnJheTogNCA0CiAgICBjbGFzc0RlZiByb2FkbWFwIGZpbGw6IzE2MTYxZCxzdHJva2U6IzQ0NDQ1NCxzdHJva2Utd2lkdGg6MXB4LGNvbG9yOiM3YTdhOGEsc3Ryb2tlLWRhc2hhcnJheTogNSA1CgogICAgUkFXWyJSYXcgVGljayBEYXRhIl0gLS0+IE1BRFsiTUFEIDVcc2lnbWEgRmlsdGVyIl06OjppbnByb2dyZXNzCiAgICBNQUQgLS0-IEtBTE1BTlsiS2FsbWFuIFByZWRpY3QtT25seSJdOjo6aW5wcm9ncmVzcwogICAgS0FMTUFOIC0tPiBEVlsiRG9sbGFyLVZvbHVtZSBCYXJzIl06OjppbnByb2dyZXNzCiAgICBEViAtLT4gRkZEWyJGRkQgV2luZG93ZWQvUHJvbnkiXTo6OmlucHJvZ3Jlc3MKICAgIEZGRCAtLT4gSE1NWyJDYXVzYWwgSE1NIDJEIl06OjppbnByb2dyZXNzCiAgICBGRkQgLS0-IElNTVsiSU1NIEthbG1hbiAyRCJdOjo6aW5wcm9ncmVzcwogICAgSE1NIC0tPiBDVVNVTVsiQ1VTVU0gRXZlbnRzIl06OjppbnByb2dyZXNzCiAgICBJTU0gLS0-IENVU1VNCiAgICBDVVNVTSAtLT4gU0xbIlNMIMSQ4buRaSB44bupbmciXTo6OmNvbXBsZXRlZAogICAgU0wgLS0-IFRSQUlMWyJUcmFpbGluZyBFeGl0IHYzIl06Ojpjb21wbGV0ZWQKICAgIFRSQUlMIC0tPiBNT0RFWyJjbGFzc2lmeV90cmFkZV9tb2RlIl06Ojpjb21wbGV0ZWQKICAgIE1PREUgLS0-IEtFTExZWyIzLUxheWVyIEtlbGx5Il06Ojpjb21wbGV0ZWQKICAgIEtFTExZIC0tPiBDUENWWyJQdXJnZWRLRm9sZCAvIENQQ1YiXTo6OmlucHJvZ3Jlc3MKICAgIENQQ1YgLS0-IEVYRUNbIkV4ZWN1dGlvbiBTaW11bGF0b3IiXTo6OmlucHJvZ3Jlc3MKICAgIEVYRUMgLS0-IENCWyJDaXJjdWl0IEJyZWFrZXIiXTo6OmlucHJvZ3Jlc3MKICAgIENCIC0tPiBSVEtbIlJ1c3QgUlRLIEhhbmRvZmYiXTo6OnJvYWRtYXAK?bgColor=ffffff"

if [ ! -f "architecture.png" ]; then
    echo "Cảnh báo: Không thể tải ảnh sơ đồ. Báo cáo vẫn sẽ biên dịch nhưng thiếu hình ảnh này."
fi

echo "Đang tạo lại file .tex từ .md qua pandoc (tích hợp chuẩn Báo cáo Khoa học)..."
pandoc SCIENTIFIC_REPORT.md \
    -s \
    --pdf-engine=xelatex \
    -V mainfont="Times New Roman" \
    -V mathfont="Times New Roman" \
    -o SCIENTIFIC_REPORT.tex

if [ ! -f "SCIENTIFIC_REPORT.tex" ]; then
    echo "Lỗi: Không tạo được file SCIENTIFIC_REPORT.tex từ pandoc."
    exit 1
fi

echo "Đang biên dịch bằng xelatex (quá trình này có thể mất 15-30 giây)..."
xelatex -interaction=nonstopmode SCIENTIFIC_REPORT.tex
xelatex -interaction=nonstopmode SCIENTIFIC_REPORT.tex

if [ -f "SCIENTIFIC_REPORT.pdf" ]; then
    echo "=== HOÀN TẤT! ==="
    echo "File PDF đã được tạo thành công: SCIENTIFIC_REPORT.pdf"
    open SCIENTIFIC_REPORT.pdf
else
    echo "Biên dịch thất bại, vui lòng kiểm tra log lỗi ở trên."
fi
