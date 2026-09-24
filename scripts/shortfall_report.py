#!/usr/bin/env python3
"""
CHI PHÍ KHỚP LỆNH THẬT (implementation shortfall) — thứ backtest giả định mà chưa ai đo.

VÌ SAO CẦN: backtest (`run_v3`) giả định mỗi lượt tái cân bằng khớp ĐÚNG giá đóng nến
4h của mốc tái cân bằng, rồi trừ chi phí 4-5bp/chiều. `ExecutionRecord` chỉ ghi phí
sàn và khối lượng maker/taker — KHÔNG ghi giá khớp so với giá lúc ra quyết định. Lượt
23/09 cho thấy vì sao đó là lỗ hổng: phí sàn chỉ $16 (3,62bp) nhưng giá trôi tới 778bp
trong cửa sổ chờ 900s, và equity mất $417 trong 6 giờ sau lượt. Khoản trôi giá đó
KHÔNG nằm trong dòng "phí" — nó nằm lẫn trong lãi/lỗ, nên chưa ai thấy nó.

Tách chi phí thật làm BA khoản, mỗi khoản sửa ở một chỗ khác nhau:

  1. TRỄ    = giá lúc QUYẾT ĐỊNH so với giá đóng NẾN 4H mà backtest giả định khớp.
              Lớn -> dữ liệu/lịch chạy chậm (máy ngủ, nến cũ). Sửa ở lịch tái cân bằng.
  2. KHỚP   = giá KHỚP so với giá lúc quyết định. Lớn -> cửa sổ chờ thụ động bị chọn
              ngược (lệnh chờ chỉ khớp khi giá đi NGƯỢC mình). Sửa ở `oms/`.
  3. PHÍ    = commission sàn trả về trong từng giao dịch.

Dấu: DƯƠNG = tốn tiền. Mua cao hơn tham chiếu hay bán thấp hơn tham chiếu đều dương.

Giá tham chiếu lấy từ CÙNG sàn với lệnh (testnet thì nến testnet) — so giá khớp testnet
với nến mainnet là đo độ lệch giữa hai sàn chứ không phải chi phí.

    python scripts/shortfall_report.py                 # mọi lượt trong 28 ngày qua
    python scripts/shortfall_report.py --days 14 --mainnet
Cần khoá API chỉ-đọc (`BINANCE_TESTNET_API_KEY/SECRET`, xem `core/credentials.py`).
"""
import argparse
import json
import time
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

EXEC_LOG = "artifacts/execution_log.jsonl"
BAR_MS = 4 * 3_600_000
#: Lệnh của một lượt khớp trong cửa sổ này tính từ lúc quyết định (chờ thụ động 900s
#: + đuổi giá + taker dự phòng). Rộng tay để không cắt mất đuôi lệnh taker muộn.
WINDOW_MS = 2 * 3_600_000
#: Giả định chi phí một chiều của backtest (CLAUDE.md: "chi phí thật 4-5bp/chiều").
BACKTEST_ASSUMED_BPS = 4.5


# ---------------------------------------------------------------------------
# Phần TÍNH — hàm thuần, không gọi mạng (test ở tests/scripts/test_shortfall_report.py)
# ---------------------------------------------------------------------------
def last_closed_bar_open(ts_ms: int, bar_ms: int = BAR_MS) -> int:
    """Mốc MỞ của nến 4h cuối cùng đã ĐÓNG trước `ts_ms` (nến căn theo UTC)."""
    return (ts_ms // bar_ms) * bar_ms - bar_ms


def shortfall_table(trades: pd.DataFrame, ref_decision: Dict[str, float],
                    ref_bar: Dict[str, float]) -> pd.DataFrame:
    """
    Chi phí từng giao dịch, bp và USD. `trades` cần: symbol, side, price, qty,
    commission, commissionAsset, maker.

    Giao dịch thiếu giá tham chiếu bị BỎ và đếm riêng — điền giá khớp làm tham chiếu
    sẽ cho chi phí 0 và kéo trung bình về 0 một cách giả tạo.
    """
    t = trades.copy()
    t["price"] = t["price"].astype(float)
    t["qty"] = t["qty"].astype(float)
    t["notional"] = t["price"] * t["qty"]
    t["sgn"] = np.where(t["side"].str.upper() == "BUY", 1.0, -1.0)
    t["ref_dec"] = t["symbol"].map(ref_decision).astype(float)
    t["ref_bar"] = t["symbol"].map(ref_bar).astype(float)
    t = t[np.isfinite(t["ref_dec"]) & np.isfinite(t["ref_bar"])
          & (t["ref_dec"] > 0) & (t["ref_bar"] > 0)].copy()
    # Khớp: giá khớp so với lúc quyết định. Trễ: lúc quyết định so với nến backtest.
    t["exec_bps"] = t["sgn"] * (t["price"] - t["ref_dec"]) / t["ref_dec"] * 1e4
    t["delay_bps"] = t["sgn"] * (t["ref_dec"] - t["ref_bar"]) / t["ref_bar"] * 1e4
    t["exec_usd"] = t["sgn"] * (t["price"] - t["ref_dec"]) * t["qty"]
    t["delay_usd"] = t["sgn"] * (t["ref_dec"] - t["ref_bar"]) * t["qty"]
    usdt = t["commissionAsset"].astype(str).str.upper().isin(["USDT", "USDC"])
    t["fee_usd"] = np.where(usdt, t["commission"].astype(float), np.nan)
    return t


def summarize(t: pd.DataFrame) -> Dict[str, float]:
    """Gộp theo notional. bp = USD / notional x 1e4, nên cặp lệnh lớn nặng hơn."""
    if t.empty:
        return {"turnover": 0.0, "n_trades": 0}
    n = t["notional"].sum()

    def bps(col, mask=None):
        s = t if mask is None else t[mask]
        d = s["notional"].sum()
        return float(s[col].sum() / d * 1e4) if d > 0 else np.nan

    fee_known = t["fee_usd"].notna()
    fee_bps = (float(t.loc[fee_known, "fee_usd"].sum() / t.loc[fee_known, "notional"].sum()
                     * 1e4) if fee_known.any() else np.nan)
    maker = t["maker"].astype(bool)
    return {
        "turnover": float(n), "n_trades": int(len(t)),
        "maker_share": float(t.loc[maker, "notional"].sum() / n),
        "delay_bps": bps("delay_usd"), "exec_bps": bps("exec_usd"), "fee_bps": fee_bps,
        "exec_maker_bps": bps("exec_usd", maker), "exec_taker_bps": bps("exec_usd", ~maker),
        "delay_usd": float(t["delay_usd"].sum()), "exec_usd": float(t["exec_usd"].sum()),
        "fee_usd": float(t["fee_usd"].sum(skipna=True)),
    }


# ---------------------------------------------------------------------------
# Phần GỌI SÀN
# ---------------------------------------------------------------------------
def _signed(c, path: str, params: dict):
    return c._request("GET", path, params, signed=True)


def traded_symbols(c, start_ms: int, end_ms: int) -> List[str]:
    """Cặp CÓ phát sinh phí trong cửa sổ — rẻ hơn hỏi userTrades cho cả universe."""
    rows = _signed(c, "/fapi/v1/income", {"incomeType": "COMMISSION", "startTime": start_ms,
                                          "endTime": end_ms, "limit": 1000}) or []
    return sorted({r["symbol"] for r in rows if r.get("symbol")})


def fetch_trades(c, symbols: Iterable[str], start_ms: int, end_ms: int) -> pd.DataFrame:
    rows = []
    for s in symbols:
        d = _signed(c, "/fapi/v1/userTrades", {"symbol": s, "startTime": start_ms,
                                               "endTime": end_ms, "limit": 1000}) or []
        rows += d
        time.sleep(0.12)                      # userTrades nặng 5 -> giữ dưới trần 2400/phút
    return pd.DataFrame(rows)


def _kline_price(c, symbol: str, interval: str, open_ms: int, field: int) -> Optional[float]:
    k = c.klines(symbol, interval, start_ms=open_ms, end_ms=open_ms, limit=1) or []
    return float(k[0][field]) if k and int(k[0][0]) == open_ms else None


def reference_prices(c, symbols: Iterable[str], decision_ms: int):
    """(giá MỞ nến 1m chứa lúc quyết định, giá ĐÓNG nến 4h cuối đã đóng)."""
    minute = decision_ms // 60_000 * 60_000
    bar = last_closed_bar_open(decision_ms)
    dec, barp = {}, {}
    for s in symbols:
        p1 = _kline_price(c, s, "1m", minute, 1)
        p4 = _kline_price(c, s, "4h", bar, 4)
        if p1:
            dec[s] = p1
        if p4:
            barp[s] = p4
    return dec, barp


def load_rebalances(days: float, now_ms: int) -> List[dict]:
    recs = [json.loads(x) for x in open(EXEC_LOG, encoding="utf-8") if x.strip()]
    lo = now_ms - int(days * 86_400_000)
    return [r for r in recs if r["timestamp_ms"] >= lo]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=float, default=28.0)
    ap.add_argument("--mainnet", action="store_true")
    a = ap.parse_args(argv)

    from aegis.data.ingestion.binance_rest import BinanceFuturesREST
    c = BinanceFuturesREST(testnet=not a.mainnet)
    now = int(time.time() * 1000)
    recs = load_rebalances(a.days, now)
    if not recs:
        print(f"Không có lượt tái cân bằng nào trong {a.days:.0f} ngày ({EXEC_LOG}).")
        return 1

    print("=" * 112)
    print(f"CHI PHÍ KHỚP LỆNH THẬT — {len(recs)} lượt, {'MAINNET' if a.mainnet else 'TESTNET'}"
          f" | dương = tốn tiền | backtest giả định {BACKTEST_ASSUMED_BPS}bp/chiều")
    print("=" * 112)
    print(f"{'lượt (UTC)':<18}{'equity':>9}{'turnover':>10}{'maker':>7}{'TRỄ bp':>8}{'KHỚP bp':>9}"
          f"{' maker':>8}{' taker':>8}{'PHÍ bp':>8}{'TỔNG bp':>9}{'TỔNG $':>9}{'% vốn':>8}")
    print("-" * 112)
    allt, rows = [], []
    for r in recs:
        t0 = int(r["timestamp_ms"])
        syms = traded_symbols(c, t0, t0 + WINDOW_MS)
        if not syms:
            continue
        tr = fetch_trades(c, syms, t0, t0 + WINDOW_MS)
        if tr.empty:
            continue
        dec, barp = reference_prices(c, syms, t0)
        t = shortfall_table(tr, dec, barp)
        s = summarize(t)
        if not s.get("n_trades"):
            continue
        fee = s["fee_bps"] if np.isfinite(s["fee_bps"]) else 0.0
        tot_bps = s["delay_bps"] + s["exec_bps"] + fee
        tot_usd = s["delay_usd"] + s["exec_usd"] + s["fee_usd"]
        eq = float(r["equity"])
        print(f"{pd.Timestamp(t0, unit='ms'):%Y-%m-%d %H:%M}  {eq:>9.0f}{s['turnover']:>10.0f}"
              f"{s['maker_share']*100:>6.0f}%{s['delay_bps']:>8.1f}{s['exec_bps']:>9.1f}"
              f"{s['exec_maker_bps']:>8.1f}{s['exec_taker_bps']:>8.1f}{s['fee_bps']:>8.2f}"
              f"{tot_bps:>9.1f}{tot_usd:>9.0f}{tot_usd/eq*100:>7.2f}%")
        t["rebalance_ms"] = t0
        allt.append(t)
        rows.append({"ts": t0, "equity": eq, **s, "total_bps": tot_bps, "total_usd": tot_usd})

    if not allt:
        print("Không lấy được giao dịch nào (khoá API? cửa sổ thời gian?).")
        return 1
    T = pd.concat(allt)
    S = summarize(T)
    tot = S["delay_bps"] + S["exec_bps"] + (S["fee_bps"] if np.isfinite(S["fee_bps"]) else 0)
    df = pd.DataFrame(rows)
    per_year = 365.0 / 3.0                                        # chu kỳ 72h
    turn_eq = float((df["turnover"] / df["equity"]).median())     # turnover/vốn mỗi lượt
    print("-" * 112)
    print(f"{'GỘP (theo notional)':<18}{'':>9}{S['turnover']:>10.0f}{S['maker_share']*100:>6.0f}%"
          f"{S['delay_bps']:>8.1f}{S['exec_bps']:>9.1f}{S['exec_maker_bps']:>8.1f}"
          f"{S['exec_taker_bps']:>8.1f}{S['fee_bps']:>8.2f}{tot:>9.1f}")
    print(f"\n  Chi phí thật một chiều: {tot:.1f}bp so với {BACKTEST_ASSUMED_BPS}bp backtest "
          f"giả định -> gấp {tot/BACKTEST_ASSUMED_BPS:.1f} lần.")
    extra = (tot - BACKTEST_ASSUMED_BPS) / 1e4 * turn_eq * per_year
    print(f"  Turnover trung vị mỗi lượt = {turn_eq:.1f}x vốn. Ở {per_year:.0f} lượt/năm, "
          f"phần VƯỢT giả định ăn {extra*100:+.0f}% vốn/năm.")
    print("\n  ⚠️ Một lượt có ~50-70 lệnh: TRỄ và KHỚP của MỘT lượt còn nhiễu vì giá cả thị trường")
    print("  cùng trôi. Đọc con số GỘP qua nhiều lượt. Và testnet có sổ lệnh mỏng hơn mainnet —")
    print("  con số testnet là chặn TRÊN hợp lý cho chi phí khớp, chưa phải con số mainnet.")
    out = "artifacts/shortfall_report.csv"
    df.to_csv(out, index=False)
    print(f"\n  đã ghi {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
