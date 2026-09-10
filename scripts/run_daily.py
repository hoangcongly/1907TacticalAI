#!/usr/bin/env python3
"""
Vòng lặp vận hành hằng ngày — chiến lược cross-sectional market-neutral.

    python scripts/run_daily.py                  # DRY RUN (mặc định, không đặt lệnh)
    python scripts/run_daily.py --live           # đặt lệnh THẬT trên testnet
    python scripts/run_daily.py --live --mainnet # TIỀN THẬT (phải xác nhận thủ công)
    python scripts/run_daily.py --kill           # dừng khẩn cấp, đóng sạch vị thế
    python scripts/run_daily.py --status         # xem trạng thái, không hành động

MẶC ĐỊNH AN TOÀN: dry-run + testnet. Muốn chạm tiền thật phải nêu tường minh cả
`--live` lẫn `--mainnet`, và gõ xác nhận.
"""
import argparse
import json
import logging
import sys

from aegis.core.state_store import StateStore
from aegis.data.ingestion.binance_rest import BinanceFuturesREST
from aegis.oms.order_router import BinanceOrderRouter
from aegis.pipelines.xs_live_pipeline import CrossSectionalLivePipeline, LiveConfig


def show_status(pipe: CrossSectionalLivePipeline) -> int:
    state = pipe.store.load()
    bal = pipe.client.balance_usdt()
    equity = bal["wallet_balance"] + bal["unrealized_pnl"]
    pos = [p for p in pipe.client.position_risk() if abs(float(p.get("positionAmt", 0))) > 0]

    print("=" * 62)
    print(f"  Ví            : ${bal['wallet_balance']:.2f}")
    print(f"  uPnL          : ${bal['unrealized_pnl']:+.2f}")
    print(f"  Equity        : ${equity:.2f}")
    print(f"  Đỉnh vốn      : ${state.peak_equity:.2f}")
    print(f"  Drawdown      : {state.drawdown(equity)*100:.1f}%")
    print(f"  Kill switch   : {'🔴 ĐÃ KÍCH HOẠT' if state.is_dead else '🟢 bình thường'}")
    print(f"  Số lần cân    : {state.rebalance_count}")
    print(f"  Vị thế mở     : {len(pos)}")
    for p in pos:
        amt = float(p["positionAmt"])
        print(f"     {p['symbol']:<14}{amt:>12.4f}  uPnL ${float(p['unRealizedProfit']):+8.2f}")
    if state.last_error:
        print(f"  Lỗi gần nhất  : {state.last_error[:120]}")
    print("=" * 62)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Vòng lặp vận hành Aegis cross-sectional")
    ap.add_argument("--live", action="store_true", help="Đặt lệnh thật (mặc định là dry-run)")
    ap.add_argument("--mainnet", action="store_true", help="Dùng TIỀN THẬT thay vì testnet")
    ap.add_argument("--kill", action="store_true", help="Dừng khẩn cấp, đóng sạch vị thế")
    ap.add_argument("--status", action="store_true", help="Chỉ xem trạng thái")
    ap.add_argument("--costs", action="store_true", help="Báo cáo chi phí thực thi thực tế")
    ap.add_argument("--leverage", type=float, default=2.0)
    ap.add_argument("--skip-refresh", action="store_true", help="Bỏ qua tải dữ liệu mới")
    ap.add_argument("--state", default="artifacts/live_state.json")
    ap.add_argument("--config", default=None,
                    help="File cấu hình chiến lược. Mặc định: artifacts/strategy_v3.json "
                         "nếu có, nếu không thì artifacts/strategy_validated.json (v1).")
    a = ap.parse_args(argv)

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")

    if a.mainnet and a.live:
        print("\n⚠️  SẮP GIAO DỊCH BẰNG TIỀN THẬT TRÊN MAINNET ⚠️")
        if input("   Gõ đúng chữ  TOI DONG Y  để tiếp tục: ").strip() != "TOI DONG Y":
            print("   Đã huỷ."); return 1

    import pathlib as _pl
    cfg_path = a.config
    if cfg_path is None:
        v3 = _pl.Path("artifacts/strategy_v3.json")
        cfg_path = str(v3) if v3.is_file() else "artifacts/strategy_validated.json"
    cfg = LiveConfig.from_artifacts(cfg_path)
    cfg.leverage = a.leverage
    client = BinanceFuturesREST(testnet=not a.mainnet)
    dry = not a.live
    pipe = CrossSectionalLivePipeline(
        config=cfg, client=client,
        router=BinanceOrderRouter(client=client, dry_run=dry),
        state_store=StateStore(a.state), dry_run=dry,
    )

    if a.costs:
        from aegis.core.execution_log import ExecutionLog
        s = ExecutionLog().summary()
        if not s.get("n_rebalances"):
            print("Chưa có lượt tái cân bằng nào được ghi nhận."); return 0
        print("=" * 58)
        print(f"  Số lượt cân        : {s['n_rebalances']}")
        print(f"  Tỷ lệ khớp maker   : {s['maker_ratio']*100:.1f}%")
        print(f"  Tỷ lệ khớp/dự tính : {s['fill_ratio']*100:.1f}%")
        print(f"  Chi phí THỰC TẾ    : {s['realized_cost_bps']:.2f} bp")
        print(f"  Backtest giả định  : {s['backtest_assumed_bps']:.2f} bp")
        print(f"  Chênh lệch         : {s['realized_cost_bps']/max(s['backtest_assumed_bps'],1e-9):.1f}x")
        print(f"  Lệch hướng TB      : {s['avg_net_exposure_pct']:.1f}%  (0% = trung lập hoàn hảo)")
        print(f"  Tổng đã giao dịch  : ${s['total_traded_usd']:,.0f}")
        print(f"  Tổng phí đã trả    : ${s['total_cost_usd']:,.2f}")
        print("=" * 58)
        return 0

    if a.status:
        return show_status(pipe)

    if a.kill:
        # [FIX KILL] Kill switch KHÔNG BAO GIỜ được chạy dry-run. Một nút dừng khẩn
        # cấp chỉ giả vờ đóng vị thế còn nguy hiểm hơn là không có nút nào, vì
        # người vận hành sẽ tin là mình đã an toàn.
        pipe.dry_run = False
        pipe.router.dry_run = False
        print("🛑 DỪNG KHẨN CẤP — đóng toàn bộ vị thế (THẬT, không dry-run)...")
        print(json.dumps(pipe.kill(), indent=2, ensure_ascii=False))
        return 0

    mode = "DRY RUN" if dry else ("MAINNET — TIỀN THẬT" if a.mainnet else "TESTNET")
    desc = (f"engine=v3 ({cfg.n_positions} vị thế, {cfg.rebalance_hours:.0f}h/lượt)"
            if cfg.engine == "v3" else f"tín hiệu={','.join(cfg.signals)}")
    print(f"[{mode}] {desc} | đòn bẩy={cfg.leverage}x "
          f"| top_frac={cfg.top_frac}\n")

    res = pipe.run_once(skip_data_refresh=a.skip_refresh)

    print(f"Hành động   : {res.get('action')}")
    if res.get("reason"):
        print(f"Lý do       : {res['reason']} — {res.get('detail','')}")
    print(f"Equity      : ${res.get('equity',0):.2f}")
    if "data_age_hours" in res:
        print(f"Tuổi dữ liệu: {res['data_age_hours']}h")
    if "n_targets" in res:
        print(f"Mục tiêu    : {res['n_targets']} vị thế (sức chứa vốn: {res['capacity']})")
    if "n_orders" in res:
        print(f"Lệnh        : {res['n_orders']} | turnover ${res.get('turnover',0):.2f} "
              f"| gross ${res.get('gross_notional',0):.2f} | bỏ qua {res.get('skipped',0)}")
    for o in res.get("orders", [])[:40]:
        print(f"   {o['side']:<5}{o['symbol']:<14}{o['qty']:>12}  @ ${o['price']:<12} "
              f"= ${o['notional']:>8.2f}  ({o['reason']})")
    if "submitted" in res:
        print(f"Đã gửi      : {res['submitted']} | bị từ chối: {res['rejected']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
