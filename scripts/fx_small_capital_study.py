#!/usr/bin/env python3
"""
FOREX Ở VỐN VÀI TRIỆU VND: chiến lược nào có bằng chứng, và thực sự lời được bao nhiêu.

Dữ liệu: tỷ giá ngày của Fed (H.10, qua kho `datasets/exchange-rates`), 1971 -> nay,
9 đồng chính so với USD: EUR (từ 1999), JPY, GBP, CHF, AUD, CAD, NZD, SEK, NOK.

    git clone --depth 1 https://github.com/datasets/exchange-rates ../datasets/exchange-rates
    python scripts/fx_small_capital_study.py
    python scripts/fx_small_capital_study.py --capital-vnd 5000000 --swap-markup 0.02

KHÔNG DÒ THAM SỐ. Mỗi chiến lược lấy nguyên cấu hình từ bài báo gốc, chạy một lần:
  * TSMOM 12 tháng        — Moskowitz, Ooi & Pedersen (2012), JFE: dấu lợi suất 12 tháng,
                            cỡ theo nghịch đảo biến động, tái cân bằng tháng.
  * TREND 1/3/12 tháng    — Hurst, Ooi & Pedersen (2017), JPM: trung bình dấu 3 chân trời.
  * XS-MOM 1 tháng        — Menkhoff, Sarno, Schmeling & Schrimpf (2012), JFE: long 3 đồng
                            thắng / short 3 đồng thua tháng trước, giữ 1 tháng.
  * MA 50/200             — quy tắc kỹ thuật bán lẻ kinh điển; Neely, Weller & Ulrich (2009)
                            cho thấy lợi nhuận của nhóm quy tắc này đã biến mất từ đầu 1990s.
Carry KHÔNG kiểm ở đây vì kho dữ liệu không có lãi suất — xem tài liệu dẫn trong
`docs/forex_small_capital_plan.md`.

Lợi suất là lợi suất GIÁ GIAO NGAY (không gồm chênh lãi suất). Ở tài khoản bán lẻ, chênh
lãi suất đến qua "swap" qua đêm mà sàn cộng thêm phần chênh — mô hình hoá bằng
`--swap-markup` (%/năm trên notional đang giữ, luôn là CHI PHÍ).

Chi phí một chiều (bp) là spread + hoa hồng điển hình của tài khoản bán lẻ, KHÔNG phải
giá liên ngân hàng: đồng chính 1–1,5bp, NZD 2,5bp, SEK/NOK 5bp.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path("../datasets/exchange-rates/data/daily.csv")
VND_PER_USD = 26_300.0
DAYS = 252
COUNTRIES = {"Euro": "EUR", "Japan": "JPY", "United Kingdom": "GBP", "Switzerland": "CHF",
             "Australia": "AUD", "Canada": "CAD", "New Zealand": "NZD", "Sweden": "SEK",
             "Norway": "NOK"}
#: Chi phí một chiều của tài khoản bán lẻ (spread + hoa hồng), bp.
COST_BPS = {"EUR": 1.0, "JPY": 1.0, "GBP": 1.5, "CHF": 1.5, "AUD": 1.5, "CAD": 1.5,
            "NZD": 2.5, "SEK": 5.0, "NOK": 5.0}
ASSET_VOL = 0.10            # mỗi cặp đóng góp rủi ro ngang nhau (MOP 2012 dùng 40%/năm/tài sản)
PORT_VOL = 0.10             # báo cáo mọi chiến lược ở cùng 10%/năm để so sánh công bằng
MICRO_LOT = 1_000           # 0,01 lot = 1.000 đơn vị đồng gốc — cỡ nhỏ nhất của đa số sàn
PERIODS = {"toàn bộ 1976–nay": "1976-01-01", "1999–nay (có EUR)": "1999-01-01",
           "2010–nay": "2010-01-01", "2020–nay": "2020-01-01"}


def load_prices(path: Path) -> pd.DataFrame:
    """Giá USD của MỘT đơn vị ngoại tệ (tăng = ngoại tệ mạnh lên so với USD)."""
    d = pd.read_csv(path, parse_dates=["Date"])
    d = d[d["Country"].isin(COUNTRIES)]
    px = d.pivot_table(index="Date", columns="Country", values="Exchange rate")
    px = 1.0 / px.rename(columns=COUNTRIES)          # kho ghi ngoại tệ / USD
    return px.ffill(limit=5)[list(COUNTRIES.values())]


def vol_scale(ret: pd.DataFrame) -> pd.DataFrame:
    """Nghịch đảo biến động EWMA (com 60 ngày, như MOP 2012), chỉ dùng dữ liệu <= t."""
    sig = ret.ewm(com=60, min_periods=60).std() * np.sqrt(DAYS)
    return ASSET_VOL / sig


def month_ends(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    s = pd.Series(idx, index=idx)
    return pd.DatetimeIndex(s.groupby(idx.to_period("M")).max().values)


def hold_monthly(w: pd.DataFrame) -> pd.DataFrame:
    """Chỉ đổi vị thế ở ngày cuối tháng; giữa tháng giữ nguyên."""
    me = month_ends(w.index)
    return w.loc[w.index.isin(me)].reindex(w.index).ffill()


def strat_tsmom(px, ret):
    sig = np.sign(px / px.shift(DAYS) - 1.0)
    return hold_monthly(sig * vol_scale(ret)) / px.shape[1]


def strat_trend(px, ret):
    sig = sum(np.sign(px / px.shift(h) - 1.0) for h in (21, 63, 252)) / 3.0
    return hold_monthly(sig * vol_scale(ret)) / px.shape[1]


def strat_xsmom(px, ret):
    r1 = px / px.shift(21) - 1.0
    rk = r1.rank(axis=1)
    n = r1.notna().sum(axis=1)
    w = rk.gt(n - 3, axis=0).astype(float) - (rk <= 3).astype(float)
    w = w.where(r1.notna(), 0.0) / 6.0
    w.loc[n < 6] = 0.0
    return hold_monthly(w * vol_scale(ret) / ASSET_VOL * 0.10)


def strat_ma(px, ret):
    sig = np.sign(px.rolling(50).mean() - px.rolling(200).mean())
    return sig * vol_scale(ret) / px.shape[1]          # quy tắc kỹ thuật: đổi vị thế HẰNG NGÀY


STRATS = {"TSMOM 12 tháng (MOP 2012)": strat_tsmom, "TREND 1/3/12 (HOP 2017)": strat_trend,
          "XS-MOM 1 tháng (MSSS 2012)": strat_xsmom, "MA 50/200 (bán lẻ)": strat_ma}


def backtest(w: pd.DataFrame, ret: pd.DataFrame, swap_markup: float) -> pd.DataFrame:
    """Vị thế quyết định cuối ngày t ăn lợi suất ngày t+1. Phí trên thay đổi vị thế."""
    w = w.fillna(0.0)
    pos = w.shift(1).fillna(0.0)
    gross = (pos * ret.fillna(0.0)).sum(axis=1)
    cost_bp = pd.Series(COST_BPS)[w.columns]
    trade = (w - w.shift(1).fillna(0.0)).abs()
    cost = (trade * cost_bp / 1e4).sum(axis=1).shift(1).fillna(0.0)
    swap = pos.abs().sum(axis=1) * swap_markup / DAYS
    return pd.DataFrame({"gross": gross, "net": gross - cost - swap, "cost": cost,
                         "swap": swap, "lev": pos.abs().sum(axis=1),
                         "turn": trade.sum(axis=1)})


def stats(r: pd.Series) -> dict:
    r = r.dropna()
    vol = r.std() * np.sqrt(DAYS)
    eq = (1 + r).cumprod()
    m = (1 + r).groupby(r.index.to_period("M")).prod() - 1
    return {"ann": r.mean() * DAYS, "vol": vol, "sharpe": r.mean() * DAYS / vol if vol else np.nan,
            "maxdd": float((eq / eq.cummax() - 1).min()), "worst_m": float(m.min()),
            "skew_m": float(m.skew()), "pos_years": float(
                ((1 + r).groupby(r.index.year).prod() > 1).mean())}


def granular(w_scaled: pd.DataFrame, px: pd.DataFrame, capital_usd: float, lot_units: float
             ) -> pd.DataFrame:
    """
    Trọng số THỰC ĐẶT ĐƯỢC khi mỗi lệnh phải là bội số của `lot_units` đơn vị đồng gốc.

    Cặp XXX/USD (EUR, GBP, AUD, NZD): đồng gốc là ngoại tệ, 1 lot = lot × giá USD.
    Cặp USD/XXX (JPY, CHF, CAD, SEK, NOK): đồng gốc là USD, 1 lot = lot USD.
    Làm tròn về bội số GẦN NHẤT — tức vị thế dưới nửa lot bị bỏ, trên nửa lot bị ĐẨY lên.
    """
    base_is_fx = {"EUR", "GBP", "AUD", "NZD"}
    lot_usd = pd.DataFrame({c: (px[c] * lot_units if c in base_is_fx
                                else pd.Series(float(lot_units), index=px.index))
                            for c in px.columns})
    notional = w_scaled * capital_usd
    lots = (notional / lot_usd).round()
    return (lots * lot_usd / capital_usd).where(w_scaled.notna())


def single_lot(w: pd.DataFrame, px: pd.DataFrame, capital_usd: float, lot_units: float
               ) -> pd.DataFrame:
    """Đúng MỘT lệnh `lot_units` ở cặp có |tín hiệu| lớn nhất — cách duy nhất vốn nhỏ vào được lệnh."""
    base_is_fx = {"EUR", "GBP", "AUD", "NZD"}
    w = w.fillna(0.0)
    top = w.abs().idxmax(axis=1).where(w.abs().max(axis=1) > 0)
    out = pd.DataFrame(0.0, index=w.index, columns=w.columns)
    for c in w.columns:
        on = top == c
        lot_usd = px[c] * lot_units if c in base_is_fx else float(lot_units)
        out.loc[on, c] = np.sign(w.loc[on, c]) * (lot_usd if np.isscalar(lot_usd)
                                                  else lot_usd[on]) / capital_usd
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=DATA)
    ap.add_argument("--capital-vnd", type=float, default=3_000_000.0)
    ap.add_argument("--swap-markup", type=float, default=0.01,
                    help="phần sàn cộng vào swap qua đêm, %%/năm trên notional (0,01 = 1%%)")
    a = ap.parse_args(argv)
    if not a.data.exists():
        raise SystemExit(f"Thiếu {a.data}. Chạy: git clone --depth 1 "
                         f"https://github.com/datasets/exchange-rates ../datasets/exchange-rates")

    px = load_prices(a.data)
    ret = px.pct_change(fill_method=None)
    print(f"Dữ liệu: {px.index[0].date()} -> {px.index[-1].date()}, {px.shape[1]} cặp so với USD")

    # ---------------------------------------------------------------- 1. edge, ở 10%/năm
    print("\n" + "=" * 112)
    print(f"1. CHIẾN LƯỢC CÓ BẰNG CHỨNG, cùng biến động {PORT_VOL:.0%}/năm, sau chi phí bán lẻ + "
          f"swap cộng thêm {a.swap_markup:.1%}/năm")
    print("=" * 112)
    print(f"{'chiến lược':<28}{'giai đoạn':<20}{'lãi/năm':>8}{'Sharpe':>8}{'gộp':>7}{'maxDD':>8}"
          f"{'tháng tệ':>9}{'skew':>7}{'năm lời':>8}{'phí+swap':>9}")
    print("-" * 112)
    results = {}
    for name, fn in STRATS.items():
        w = fn(px, ret)
        bt = backtest(w, ret, a.swap_markup)
        for pname, start in PERIODS.items():
            seg = bt.loc[start:]
            k = PORT_VOL / (seg["net"].std() * np.sqrt(DAYS))       # quy về 10%/năm
            s = stats(seg["net"] * k)
            g = stats(seg["gross"] * k)
            drag = (seg["cost"] + seg["swap"]).mean() * DAYS * k
            results[(name, pname)] = {**s, "k": k, "lev": seg["lev"].mean() * k}
            print(f"{name:<28}{pname:<20}{s['ann']*100:>7.1f}%{s['sharpe']:>8.2f}{g['sharpe']:>7.2f}"
                  f"{s['maxdd']*100:>7.1f}%{s['worst_m']*100:>8.1f}%{s['skew_m']:>7.2f}"
                  f"{s['pos_years']*100:>7.0f}%{drag*100:>8.1f}%")
        print()
    print("  'gộp' = Sharpe TRƯỚC chi phí. Không tham số nào được dò: cấu hình lấy nguyên từ bài báo.")
    print("\n  Độ nhạy chi phí — TREND 1/3/12, Sharpe theo giai đoạn:")
    wt = STRATS["TREND 1/3/12 (HOP 2017)"](px, ret)
    for lab, mk, cost_on in (("trước mọi chi phí", 0.0, False), ("chỉ spread+phí", 0.0, True),
                              ("+ swap 0,5%/năm", 0.005, True), ("+ swap 1%/năm", 0.01, True)):
        bt = backtest(wt, ret, mk)
        col = bt["net"] if cost_on else bt["gross"]
        cells = "".join(f"{stats(col.loc[st:])['sharpe']:>9.2f}" for st in PERIODS.values())
        print(f"    {lab:<20}{cells}     ({' | '.join(PERIODS)})")

    # ---------------------------------------------------------------- 2. vốn nhỏ + lô tối thiểu
    cap = a.capital_vnd / VND_PER_USD
    best = "TREND 1/3/12 (HOP 2017)"
    w = STRATS[best](px, ret)
    bt = backtest(w, ret, a.swap_markup)
    seg = bt.loc["2010-01-01":]
    k = PORT_VOL / (seg["net"].std() * np.sqrt(DAYS))
    w10 = (w * k).loc["2010-01-01":]
    print("=" * 112)
    print(f"2. VỐN {a.capital_vnd:,.0f} VND (~${cap:,.0f}): {best}, mục tiêu {PORT_VOL:.0%}/năm, 2010–nay")
    print("=" * 112)
    print(f"  notional mục tiêu trung bình mỗi cặp: ${(w10.abs().mean().mean() * cap):,.2f}"
          f"   |  0,01 lot ≈ $1.000–1.300 notional")
    print(f"  => với lô 0,01 thì đòn bẩy của MỘT lệnh nhỏ nhất là {1000 / cap:.1f}x vốn\n")
    print(f"{'cỡ lệnh nhỏ nhất':<34}{'số cặp giữ':>11}{'đòn bẩy tb':>11}{'vol thật':>9}{'Sharpe':>8}"
          f"{'maxDD':>8}{'lãi/năm':>9}")
    print("-" * 90)
    for label, units in (("lý tưởng (chia nhỏ vô hạn)", None), ("1 đơn vị (sàn cho đặt theo unit)", 1),
                         ("0,001 lot = 100 đơn vị", 100), ("0,01 lot = 1.000 đơn vị (micro)", MICRO_LOT)):
        wg = w10 if units is None else granular(w10, px.loc[w10.index], cap, units)
        b = backtest(wg, ret.loc[w10.index], a.swap_markup)
        s = stats(b["net"])
        held = (wg.fillna(0).abs() > 0).sum(axis=1).mean()
        print(f"{label:<34}{held:>11.1f}{b['lev'].mean():>10.2f}x{s['vol']*100:>8.1f}%"
              f"{s['sharpe']:>8.2f}{s['maxdd']*100:>7.1f}%{s['ann']*100:>8.1f}%")

    # Thứ người vốn nhỏ THỰC SỰ làm được với lô 0,01: một lệnh duy nhất, cặp có tín hiệu
    # mạnh nhất, đòn bẩy bị ÉP bởi cỡ lô chứ không do mình chọn.
    one = single_lot(w.loc["2010-01-01":], px.loc["2010-01-01":], cap, MICRO_LOT)
    b1 = backtest(one, ret.loc[one.index], a.swap_markup)
    s1 = stats(b1["net"])
    eq = np.maximum(1 + b1["net"], 0).cumprod()
    yr = eq.resample("YE").last().pct_change().dropna()
    print(f"{'1 lệnh 0,01 lot, cặp mạnh nhất':<34}{1.0:>11.1f}{b1['lev'].mean():>10.2f}x"
          f"{s1['vol']*100:>8.1f}%{s1['sharpe']:>8.2f}{s1['maxdd']*100:>7.1f}%{s1['ann']*100:>8.1f}%")
    print(f"\n  1 lệnh micro ở vốn này: {(yr < -0.5).mean()*100:.0f}% số năm mất quá nửa vốn, "
          f"{(yr > 0).mean()*100:.0f}% số năm có lời, vốn cuối/đầu 2010–nay = {eq.iloc[-1]:.2f}")
    # Vốn tối thiểu để giữ ĐỦ danh mục: cặp nhỏ nhất (phân vị 25% |trọng số|) vẫn >= 1 lô.
    wq = float(w10.abs().replace(0, np.nan).stack().quantile(0.25))
    for label, units in (("0,01 lot (1.000 đơn vị)", MICRO_LOT), ("0,001 lot (100 đơn vị)", 100)):
        need = units * 1.15 / wq
        print(f"  vốn tối thiểu để giữ đủ {px.shape[1]} cặp ở {PORT_VOL:.0%}/năm với {label}: "
              f"~${need:,.0f} ≈ {need * VND_PER_USD / 1e6:,.0f} triệu VND")

    # ---------------------------------------------------------------- 3. ra tiền
    print("\n" + "=" * 112)
    print(f"3. RA TIỀN — vốn {a.capital_vnd:,.0f} VND, nếu Sharpe sau phí GIỮ NGUYÊN như 2010–nay")
    print("=" * 112)
    from scipy.stats import norm
    sh_meas = results[(best, "2010–nay")]["sharpe"]
    for tag, sh in ((f"ĐO ĐƯỢC 2010–nay (S = {sh_meas:.2f})", sh_meas),
                    ("LẠC QUAN theo bài báo, carry + trend gộp trước phí bán lẻ (S = 0,50)", 0.50)):
        print(f"\n  {tag}. g = S·σ − σ²/2; Kelly: σ* = S, g* = S²/2")
        if sh <= 0:
            print("    Sharpe <= 0: KHÔNG có mức đòn bẩy nào cho lời kỳ vọng. Kelly = 0 — không giao dịch.")
            continue
        print(f"    {'biến động':<14}{'lãi kỳ vọng/năm':>16}{'tăng trưởng gộp':>17}{'VND/tuần':>11}"
              f"{'P(lỗ sau 1 năm)':>17}")
        for vol in (0.05, 0.10, 0.20, sh):
            mu = sh * vol
            g = mu - vol ** 2 / 2
            wk = (np.exp(g / 52) - 1) * a.capital_vnd
            lab = f"{vol:.0%}" + (" (Kelly)" if vol == sh else "")
            print(f"    {lab:<14}{mu*100:>15.1f}%{g*100:>16.1f}%{wk:>11,.0f}"
                  f"{norm.cdf(-g / vol)*100:>16.0f}%")
    print("\n  ⚠️ Kelly đầy đủ là TRẦN lý thuyết với Sharpe biết chắc; Sharpe thật không biết chắc,")
    print("     nên thực tế nên chạy ≤ 1/2 Kelly.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
