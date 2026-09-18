"""
Báo cáo chi tiết từng vị thế cho Telegram.

MỘT ĐIỀU PHẢI NÓI RÕ NGAY: chiến lược này KHÔNG có chốt lời / chốt lỗ theo giá.

Đó không phải thiếu sót mà là thiết kế. Đây là danh mục cross-sectional
market-neutral: vị thế được mở vì tài sản đó xếp hạng cao/thấp trong mặt cắt ngang,
và được đóng khi nó rơi khỏi nhóm ở lần tái cân bằng kế tiếp. Điều kiện thoát là
THỨ HẠNG và THỜI GIAN, không phải MỨC GIÁ.

Đặt chốt lỗ theo giá cho từng chân sẽ phá vỡ tính trung lập: nếu chân long bị cắt
còn chân short vẫn giữ, danh mục lập tức thành cược có hướng — đúng cơ chế đã làm
sổ lệch +35% ngày 10/09 và khiến toàn bộ kết quả kiểm định mất hiệu lực.

Cái THAY THẾ cho chốt lỗ, và báo cáo này hiển thị đúng chúng:

  - GIÁ THANH LÝ từng vị thế (sàn tính, có thật)
  - NGẮT MẠCH DANH MỤC theo drawdown: TIER1 −10% giảm nửa vị thế,
    TIER2 −20% đóng băng 24h, TIER3 −30% kill switch. Đây mới là "chốt lỗ" thật
    của chiến lược, và nó đặt ở cấp DANH MỤC nên không phá vỡ trung lập.

Cũng cần phân biệt hai loại đòn bẩy, vì chúng khác nhau rất xa và dễ gây hiểu nhầm:
  - đòn bẩy SÀN (mỗi cặp, thường 20x): chỉ quyết định ký quỹ tối thiểu
  - đòn bẩy THỰC (gross notional / equity, ~2x): đây mới là rủi ro thật đang gánh
"""

from typing import Any, Dict, List, Optional

__all__ = ["build_position_report", "position_rows", "chunk_message", "TELEGRAM_LIMIT"]

# Telegram từ chối tin nhắn quá 4096 ký tự và chỉ trả về lỗi HTTP — thư viện
# gửi tin trả về False, KHÔNG ném ngoại lệ. Nghĩa là báo cáo sẽ im lặng không
# bao giờ tới nơi. Danh mục 12 vị thế dài 4482 ký tự, tức vượt ngay từ lần đầu.
# [FIX F34]
TELEGRAM_LIMIT = 4000   # chừa biên cho ký tự HTML và emoji nhiều byte

TIER1_DD, TIER2_DD, TIER3_DD = 0.10, 0.20, 0.30


def position_rows(positions: List[Dict[str, Any]], equity: float) -> List[Dict[str, Any]]:
    """
    Chuẩn hoá dữ liệu vị thế thô từ sàn thành các trường để hiển thị.

    `margin` = notional / đòn bẩy sàn — số vốn thực sự bị khoá cho vị thế đó.
    `pct_account` tính theo NOTIONAL chứ không theo margin: notional mới là mức
    rủi ro đang gánh, margin chỉ là số tiền ký quỹ.
    """
    rows = []
    for p in positions:
        amt = float(p.get("positionAmt", 0) or 0)
        if abs(amt) <= 0:
            continue
        notional = abs(float(p.get("notional", 0) or 0))
        lev = float(p.get("leverage", 1) or 1)
        entry = float(p.get("entryPrice", 0) or 0)
        mark = float(p.get("markPrice", 0) or 0)
        liq = float(p.get("liquidationPrice", 0) or 0)
        upnl = float(p.get("unRealizedProfit", 0) or 0)

        # Khoảng cách tới thanh lý. Sàn trả 0 khi vị thế quá an toàn để tính được
        # (cross margin, phần còn lại của ví đủ đỡ) — khi đó để None chứ không
        # hiển thị 0, vì "0" trông như sắp bị thanh lý.
        dist = (abs(mark - liq) / mark * 100.0) if (liq > 0 and mark > 0) else None

        rows.append({
            "symbol": p["symbol"],
            "side": "LONG" if amt > 0 else "SHORT",
            "qty": abs(amt),
            "notional": notional,
            "margin": notional / lev if lev > 0 else 0.0,
            "pct_account": (notional / equity * 100.0) if equity > 0 else 0.0,
            "exchange_leverage": lev,
            "entry": entry,
            "mark": mark,
            "liq": liq if liq > 0 else None,
            "liq_distance_pct": dist,
            "upnl": upnl,
            # Hai cách quy đổi, CẢ HAI đều cần vì chúng nói hai chuyện khác nhau:
            #   `price_move_pct` — giá đã chạy bao nhiêu (con số trực giác)
            #   `upnl_pct_margin` — lãi/lỗ trên ký quỹ. Ở đòn bẩy sàn 20x, con số này
            #     bị khuếch đại 20 lần và nhìn rất đáng sợ (+582%) dù giá chỉ chạy 41%.
            #     Hiển thị riêng nó mà không kèm mức giá là gây hiểu nhầm.
            "price_move_pct": (((mark - entry) / entry * 100.0 * (1 if amt > 0 else -1))
                               if entry > 0 else 0.0),
            "upnl_pct_margin": (upnl / (notional / lev) * 100.0) if (notional > 0 and lev > 0) else 0.0,
        })
    return sorted(rows, key=lambda r: -r["notional"])


def _fmt_price(x: Optional[float]) -> str:
    """Giá crypto trải từ 0.0000005 tới 100000 — chọn số chữ số theo độ lớn."""
    if x is None or x <= 0:
        return "—"
    if x >= 1000:
        return f"{x:,.2f}"
    if x >= 1:
        return f"{x:,.4f}"
    if x >= 0.001:
        return f"{x:.6f}"
    return f"{x:.8f}"


def build_position_report(
    positions: List[Dict[str, Any]],
    equity: float,
    wallet: float,
    peak_equity: float,
    next_rebalance: Optional[str] = None,
    target_leverage: float = 2.0,
    max_rows: int = 15,
) -> str:
    """
    Dựng báo cáo HTML cho Telegram: từng vị thế + ngưỡng rủi ro cấp danh mục.
    """
    rows = position_rows(positions, equity)
    gross = sum(r["notional"] for r in rows)
    net = sum(r["notional"] * (1 if r["side"] == "LONG" else -1) for r in rows)
    real_lev = gross / equity if equity > 0 else 0.0
    dd = (1.0 - equity / peak_equity) if peak_equity > 0 else 0.0

    out = [
        f"<b>DANH MỤC — {len(rows)} vị thế</b>",
        f"Equity <b>${equity:,.2f}</b> | ví ${wallet:,.2f}",
        f"Gross ${gross:,.0f} | net {net/gross*100 if gross else 0:+.2f}% gross",
        f"Đòn bẩy THỰC <b>{real_lev:.2f}x</b> (mục tiêu {target_leverage:.1f}x)",
        f"Cách đỉnh vốn: {-dd*100:+.2f}%  (đỉnh ${peak_equity:,.2f})",
        "",
    ]

    for r in rows[:max_rows]:
        icon = "🟢" if r["side"] == "LONG" else "🔴"
        liq = (f"{_fmt_price(r['liq'])} (cách {r['liq_distance_pct']:.0f}%)"
               if r["liq_distance_pct"] is not None else "— (ví đủ đỡ)")
        out += [
            f"{icon} <b>{r['symbol']}</b> {r['side']}",
            f"   vốn vào lệnh : ${r['margin']:,.2f}  "
            f"({r['pct_account']:.1f}% tài khoản theo notional)",
            f"   notional     : ${r['notional']:,.2f} | đòn bẩy sàn {r['exchange_leverage']:.0f}x",
            f"   giá vào      : {_fmt_price(r['entry'])}",
            f"   giá hiện tại : {_fmt_price(r['mark'])}  "
            f"(giá chạy {r['price_move_pct']:+.1f}% theo hướng vị thế)",
            f"   lãi/lỗ       : ${r['upnl']:+,.2f}  "
            f"({r['upnl_pct_margin']:+.0f}% trên ký quỹ, "
            f"{r['upnl']/r['notional']*100 if r['notional'] else 0:+.1f}% trên notional)",
            f"   giá thanh lý : {liq}",
            "",
        ]

    if len(rows) > max_rows:
        out.append(f"<i>... và {len(rows)-max_rows} vị thế nữa</i>\n")

    out += [
        "<b>CHỐT LỜI / CHỐT LỖ</b>",
        "Chiến lược này KHÔNG dùng TP/SL theo giá — vị thế đóng khi tài sản rơi khỏi",
        "nhóm xếp hạng ở lần tái cân bằng, không khi giá chạm mức nào.",
        "Cắt một chân theo giá sẽ phá vỡ trung lập và biến danh mục thành cược có hướng.",
        "",
        f"Thoát lệnh kế tiếp: <b>{next_rebalance or 'chưa xác định'}</b>",
        "",
        "<b>CHỐT LỖ THẬT — ngắt mạch cấp danh mục:</b>",
        f"  TIER1 −{TIER1_DD*100:.0f}% equity → giảm nửa vị thế "
        f"= ${peak_equity*(1-TIER1_DD):,.2f}",
        f"  TIER2 −{TIER2_DD*100:.0f}% → đóng băng 24h "
        f"= ${peak_equity*(1-TIER2_DD):,.2f}",
        f"  TIER3 −{TIER3_DD*100:.0f}% → kill switch "
        f"= ${peak_equity*(1-TIER3_DD):,.2f}",
    ]
    return "\n".join(out)


def chunk_message(text: str, limit: int = TELEGRAM_LIMIT) -> List[str]:
    """
    Chia tin dài thành nhiều phần, CẮT Ở RANH GIỚI DÒNG.

    Cắt giữa dòng sẽ làm hỏng thẻ HTML (`<b>` mở ở phần này, `</b>` ở phần kia) và
    Telegram từ chối cả tin. Vì vậy luôn cắt ở `\n`, và nếu một dòng đơn lẻ đã dài
    hơn giới hạn thì mới cắt cứng.
    """
    if len(text) <= limit:
        return [text]

    parts, cur = [], []
    size = 0
    for line in text.split("\n"):
        # Dòng đơn lẻ quá dài: cắt cứng, không còn lựa chọn nào khác.
        while len(line) > limit:
            if cur:
                parts.append("\n".join(cur)); cur, size = [], 0
            parts.append(line[:limit])
            line = line[limit:]
        if size + len(line) + 1 > limit and cur:
            parts.append("\n".join(cur)); cur, size = [], 0
        cur.append(line)
        size += len(line) + 1
    if cur:
        parts.append("\n".join(cur))

    n = len(parts)
    return [f"{p}\n<i>({i+1}/{n})</i>" if n > 1 else p for i, p in enumerate(parts)]
