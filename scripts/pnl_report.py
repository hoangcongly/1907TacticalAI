#!/usr/bin/env python3
"""
BÁO CÁO LÃI/LỖ CHI TIẾT — lấy từ sổ kế toán của sàn, không suy ra từ equity.

Vì sao không nhìn equity: equity gộp lẫn lãi đã chốt, lãi chưa chốt, funding, phí,
và cả tiền nạp vào. Nhìn nó rồi kết luận "lãi X%" là cách dễ nhất để tự lừa mình.
`/fapi/v1/income` tách riêng từng khoản, và đó mới là sổ kế toán thật.

Báo cáo tách hai giai đoạn, vì trộn chúng lại sẽ trả lời sai câu hỏi
"hệ thống MỚI lời lỗ thế nào":

  - TRƯỚC mốc vá : chạy bằng code có lỗi F21-F31 (nhân đôi vị thế, lệnh mồ côi)
  - SAU mốc vá   : code hiện tại

    python scripts/pnl_report.py
    python scripts/pnl_report.py --days 7 --since "2026-09-13 03:53"
"""
import argparse
import json
import pathlib
import sys
import time
from typing import Dict, List

import numpy as np
import pandas as pd

from aegis.data.ingestion.binance_rest import BinanceFuturesREST

# Mốc lượt tái cân bằng đầu tiên chạy bằng code đã vá đầy đủ (UTC).
FIX_BOUNDARY = "2026-09-13 03:53"

# Funding testnet cao hơn mainnet — đo trực tiếp 13/09: hệ số trung vị 1.8x.
TESTNET_FUNDING_INFLATION = 1.8
# Phí mainnet đắt hơn testnet (VIP0 thật: maker 2bp / taker 5bp).
MAINNET_FEE_MULTIPLIER = 1.5


def fetch_income(c: BinanceFuturesREST, start_ms: int, end_ms: int) -> pd.DataFrame:
    rows, cur = [], start_ms
    while cur < end_ms:
        d = c._request("GET", "/fapi/v1/income",
                       {"startTime": cur, "endTime": end_ms, "limit": 1000}, signed=True)
        if not d:
            break
        rows += d
        if len(d) < 1000:
            break
        cur = int(d[-1]["time"]) + 1
        time.sleep(0.1)
    if not rows:
        return pd.DataFrame(columns=["symbol", "incomeType", "income", "time"])
    df = pd.DataFrame(rows)
    df["income"] = df["income"].astype(float)
    df["time"] = df["time"].astype("int64")
    df["t"] = pd.to_datetime(df["time"], unit="ms")
    return df


def section(title: str) -> None:
    print("\n" + "=" * 86)
    print(title)
    print("=" * 86)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=float, default=3.5)
    ap.add_argument("--since", default=FIX_BOUNDARY,
                    help="Mốc UTC chia 'hệ thống cũ' / 'hệ thống mới'")
    ap.add_argument("--testnet", action="store_true", default=True)
    a = ap.parse_args(argv)

    c = BinanceFuturesREST(testnet=a.testnet)
    end = int(time.time() * 1000)
    start = end - int(a.days * 86_400_000)
    boundary = int(pd.Timestamp(a.since, tz="UTC").value // 1_000_000)

    df = fetch_income(c, start, end)
    if df.empty:
        print("Không có bản ghi thu nhập nào trong khoảng này.")
        return 1

    trades = df[df.incomeType != "TRANSFER"]
    deposits = df[df.incomeType == "TRANSFER"]["income"].sum()

    bal = c.balance_usdt()
    equity = bal["wallet_balance"] + bal["unrealized_pnl"]
    positions = [p for p in c.position_risk()
                 if abs(float(p.get("positionAmt", 0) or 0)) > 0]

    # ---------------------------------------------------------------- 1
    section("1. TỔNG QUAN")
    realized = trades["income"].sum()
    upnl = bal["unrealized_pnl"]
    base = bal["wallet_balance"] - realized      # vốn trước khi có lãi/lỗ
    print(f"  Khoảng thời gian  : {df['t'].min()} -> {df['t'].max()} UTC")
    print(f"  Tiền nạp vào      : ${deposits:,.2f}")
    print(f"  Vốn gốc (ước)     : ${base:,.2f}")
    print()
    print(f"  Lãi/lỗ ĐÃ CHỐT    : ${realized:+,.2f}")
    print(f"  Lãi/lỗ CHƯA CHỐT  : ${upnl:+,.2f}   <- chưa phải tiền, có thể bay")
    print(f"  TỔNG              : ${realized+upnl:+,.2f}"
          f"   = {(realized+upnl)/base*100 if base else 0:+.2f}% trên vốn gốc")
    print(f"  Equity hiện tại   : ${equity:,.2f}")

    # ---------------------------------------------------------------- 2
    section("2. PHÂN RÃ THEO NGÀY VÀ LOẠI KHOẢN")
    d = trades.copy()
    d["ngày"] = d["t"].dt.strftime("%m-%d")
    piv = d.pivot_table(index="ngày", columns="incomeType", values="income",
                        aggfunc="sum").fillna(0.0)
    for col in ("COMMISSION", "FUNDING_FEE", "REALIZED_PNL"):
        if col not in piv.columns:
            piv[col] = 0.0
    piv = piv[["COMMISSION", "FUNDING_FEE", "REALIZED_PNL"]]
    piv.columns = ["phí", "funding", "lãi từ giá"]
    piv["TỔNG"] = piv.sum(axis=1)
    print(piv.round(2).to_string())
    print("-" * 60)
    print(piv.sum().round(2).to_string())

    # ---------------------------------------------------------------- 3
    section(f"3. TÁCH HAI GIAI ĐOẠN (mốc vá: {a.since} UTC)")
    old, new = trades[trades.time < boundary], trades[trades.time >= boundary]
    print(f"{'giai đoạn':<28}{'phí':>10}{'funding':>10}{'lãi từ giá':>12}{'TỔNG':>11}{'số ngày':>9}")
    print("-" * 80)
    for label, part in (("HỆ THỐNG CŨ (có lỗi)", old), ("HỆ THỐNG MỚI (đã vá)", new)):
        if part.empty:
            print(f"{label:<28}{'(không có dữ liệu)':>52}")
            continue
        g = part.groupby("incomeType")["income"].sum()
        span = (part["time"].max() - part["time"].min()) / 86_400_000
        print(f"{label:<28}{g.get('COMMISSION',0):>10.2f}{g.get('FUNDING_FEE',0):>10.2f}"
              f"{g.get('REALIZED_PNL',0):>12.2f}{part['income'].sum():>11.2f}{span:>9.2f}")
    print("-" * 80)
    if not new.empty:
        print(f"  => Hệ thống MỚI chạy {(end-boundary)/86_400_000:.2f} ngày, "
              f"{len(new[new.incomeType=='REALIZED_PNL'].symbol.unique())} cặp có lãi/lỗ chốt.")
        print(f"     Đây mới là con số trả lời 'hệ thống mới lời lỗ thế nào'.")

    # ---------------------------------------------------------------- 4
    section("4. TOP CẶP ĐÓNG GÓP (đã chốt)")
    by_sym = trades[trades.symbol != ""].groupby("symbol")["income"].sum().sort_values()
    print(f"{'cặp':<16}{'tổng':>10}{'funding':>10}{'lãi từ giá':>12}{'phí':>9}")
    print("-" * 58)
    fund_s = trades[trades.incomeType == "FUNDING_FEE"].groupby("symbol")["income"].sum()
    pnl_s = trades[trades.incomeType == "REALIZED_PNL"].groupby("symbol")["income"].sum()
    fee_s = trades[trades.incomeType == "COMMISSION"].groupby("symbol")["income"].sum()
    for s in list(by_sym.tail(5).index[::-1]) + list(by_sym.head(3).index):
        print(f"{s:<16}{by_sym[s]:>+10.2f}{fund_s.get(s,0):>+10.2f}"
              f"{pnl_s.get(s,0):>+12.2f}{fee_s.get(s,0):>+9.2f}")
    top = by_sym.abs().idxmax()
    print("-" * 58)
    print(f"  Tập trung: {top} chiếm {abs(by_sym[top])/max(abs(by_sym).sum(),1e-9)*100:.0f}% "
          f"độ lớn tổng đóng góp")

    # ---------------------------------------------------------------- 5
    section("5. LÃI CHƯA CHỐT ĐANG NẰM Ở ĐÂU")
    rows = []
    for p in positions:
        rows.append({"cặp": p["symbol"],
                     "hướng": "LONG" if float(p["positionAmt"]) > 0 else "SHORT",
                     "notional": abs(float(p.get("notional", 0) or 0)),
                     "uPnL": float(p.get("unRealizedProfit", 0) or 0)})
    pos_df = pd.DataFrame(rows).sort_values("uPnL", ascending=False)
    print(f"{'cặp':<16}{'hướng':<7}{'notional':>11}{'uPnL':>10}{'% tổng uPnL':>13}")
    print("-" * 58)
    tot_up = pos_df["uPnL"].sum()
    for _, r in pos_df.iterrows():
        print(f"{r['cặp']:<16}{r['hướng']:<7}{r['notional']:>11,.0f}{r['uPnL']:>+10.2f}"
              f"{r['uPnL']/tot_up*100 if tot_up else 0:>12.0f}%")
    print("-" * 58)
    big = pos_df.iloc[0]
    print(f"  {big['cặp']} chiếm {big['uPnL']/tot_up*100 if tot_up else 0:.0f}% toàn bộ lãi chưa chốt.")
    lg = pos_df[pos_df["hướng"] == "LONG"]["uPnL"].sum()
    sh = pos_df[pos_df["hướng"] == "SHORT"]["uPnL"].sum()
    print(f"  Chân LONG {lg:+.2f} | chân SHORT {sh:+.2f}")
    print("  Sổ trung lập đúng nghĩa thì hai chân cùng đóng góp; một chân ăn hết là")
    print("  dấu hiệu kết quả đến từ một cú đi giá, không từ chênh lệch xếp hạng.")

    # ---------------------------------------------------------------- 6
    section("6. QUY VỀ ĐIỀU KIỆN MAINNET")
    fund = trades[trades.incomeType == "FUNDING_FEE"]["income"].sum()
    fee = trades[trades.incomeType == "COMMISSION"]["income"].sum()
    pnl = trades[trades.incomeType == "REALIZED_PNL"]["income"].sum()
    adj_fund = fund / TESTNET_FUNDING_INFLATION
    adj_fee = fee * MAINNET_FEE_MULTIPLIER
    adj_total = adj_fund + adj_fee + pnl + upnl
    print(f"  Funding testnet cao hơn mainnet {TESTNET_FUNDING_INFLATION:.1f}x (đo trực tiếp 13/09)")
    print(f"  Phí mainnet đắt hơn ~{MAINNET_FEE_MULTIPLIER:.1f}x (VIP0: maker 2bp / taker 5bp)")
    print()
    print(f"{'':<24}{'testnet':>12}{'ước mainnet':>15}")
    print("-" * 52)
    print(f"{'funding':<24}{fund:>+12.2f}{adj_fund:>+15.2f}")
    print(f"{'lãi từ giá':<24}{pnl:>+12.2f}{pnl:>+15.2f}")
    print(f"{'phí':<24}{fee:>+12.2f}{adj_fee:>+15.2f}")
    print(f"{'chưa chốt':<24}{upnl:>+12.2f}{upnl:>+15.2f}")
    print("-" * 52)
    print(f"{'TỔNG':<24}{realized+upnl:>+12.2f}{adj_total:>+15.2f}")
    if base:
        print(f"{'trên vốn gốc':<24}{(realized+upnl)/base*100:>11.2f}%{adj_total/base*100:>14.2f}%")

    # ---------------------------------------------------------------- 7
    section("7. CON SỐ NÀY NÓI LÊN ĐIỀU GÌ")
    days = (df["time"].max() - df["time"].min()) / 86_400_000
    sharpe_assumed = 1.3
    t_stat = sharpe_assumed * np.sqrt(days / 365.0)
    print(f"  Số ngày quan sát   : {days:.2f}")
    print(f"  t-stat kỳ vọng     : {t_stat:.3f}   (cần ~2.0 để có ý nghĩa thống kê)")
    print(f"  Số ngày cần        : {(2.0/sharpe_assumed)**2*365:.0f} ngày chạy liên tục")
    print()
    print("  Ở Sharpe 1.3 và đòn bẩy 2x, một chuỗi 3 ngày ±15% nằm hoàn toàn trong")
    print("  dải bình thường. Con số lãi/lỗ của vài ngày KHÔNG phân biệt được chiến")
    print("  lược tốt với chiến lược vô dụng — nó chỉ đo may rủi.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
