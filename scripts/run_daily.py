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
from datetime import datetime
import json
import logging
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parent.parent
src_path = str(ROOT / "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

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
    pos_sorted = sorted(pos, key=lambda x: float(x.get("unRealizedProfit", 0)), reverse=True)
    for p in pos_sorted:
        amt = float(p["positionAmt"])
        side = "LONG" if amt > 0 else "SHORT"
        entry = float(p.get("entryPrice", 0))
        mark = float(p.get("markPrice", 0))
        pnl = float(p.get("unRealizedProfit", 0))
        lev = float(p.get("leverage", 2.0))
        roi = 0.0
        if entry > 0:
            raw_diff = (mark - entry) / entry if side == "LONG" else (entry - mark) / entry
            roi = raw_diff * lev * 100.0
        print(f"     {p['symbol']:<13} {side:<5} {amt:>11.4f}  vào ${entry:<9.4f} mark ${mark:<9.4f} uPnL ${pnl:>+7.2f} ({roi:>+6.2f}%)")
    if state.last_error:
        print(f"  Lỗi gần nhất  : {state.last_error[:120]}")
    print("=" * 62)
    return 0


def sync_state(pipe: CrossSectionalLivePipeline) -> int:
    """Đồng bộ vị thế thực tế từ sàn vào state store để xoá sai lệch sổ sách."""
    state = pipe.store.load()
    positions = pipe.client.position_risk()
    real_pos = {}
    for p in positions:
        amt = float(p.get("positionAmt", 0))
        if abs(amt) > 0:
            real_pos[p["symbol"]] = amt

    bal = pipe.client.balance_usdt()
    equity = bal["wallet_balance"] + bal["unrealized_pnl"]

    state.positions = real_pos
    state.last_equity = equity
    state.peak_equity = max(state.peak_equity, equity)
    state.last_error = None
    state.updated_ms = int(time.time() * 1000)
    pipe.store.save(state)
    print("=" * 62)
    print(f"  ✅ ĐÃ ĐỒNG BỘ THÀNH CÔNG {len(real_pos)} VỊ THẾ TỪ SÀN")
    print("=" * 62)
    for sym, amt in real_pos.items():
        print(f"     {sym:<14} : {amt:>12.4f}")
    print("=" * 62)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Vòng lặp vận hành Aegis cross-sectional")
    ap.add_argument("--live", action="store_true", help="Đặt lệnh thật (mặc định là dry-run)")
    ap.add_argument("--mainnet", action="store_true", help="Dùng TIỀN THẬT thay vì testnet")
    ap.add_argument("--kill", action="store_true", help="Dừng khẩn cấp, đóng sạch vị thế")
    ap.add_argument("--status", action="store_true", help="Chỉ xem trạng thái")
    ap.add_argument("--sync", action="store_true", help="Đồng bộ vị thế thực tế từ sàn vào state store")
    ap.add_argument("--costs", action="store_true", help="Báo cáo chi phí thực thi thực tế")
    ap.add_argument("--leverage", type=float, default=2.0)
    ap.add_argument("--skip-refresh", action="store_true", help="Bỏ qua tải dữ liệu mới")
    ap.add_argument("--force", action="store_true",
                    help="Tái cân bằng NGAY dù chưa đến hạn (chỉ dùng khi can thiệp thủ công)")
    ap.add_argument("--state", default="artifacts/live_state.json")
    ap.add_argument("--config", default=None,
                    help="File cấu hình chiến lược. Mặc định: artifacts/strategy_v3.json "
                         "nếu có, nếu không thì artifacts/strategy_validated.json (v1).")
    ap.add_argument("--max-order-notional", type=float, default=None,
                    help="Trần notional tối đa cho 1 lệnh (USD). Mặc định tự tính theo quy mô vốn.")
    ap.add_argument("--loop", action="store_true",
                    help="Chạy liên tục dạng daemon trong nền, tự động theo dõi và tái cân bằng")
    ap.add_argument("--loop-interval-mins", type=float, default=60.0,
                    help="Chu kỳ kiểm tra trạng thái trong chế độ loop (phút). Mặc định: 60 phút.")
    a = ap.parse_args(argv)

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")

    # [FIX F28] Khoá chống chạy chồng — chi tiết ở `aegis.core.process_lock`.
    # Thao tác chỉ đọc (--status, --costs) không cần khoá.
    from aegis.core.process_lock import ProcessLockBusy, acquire_lock
    read_only = a.status or a.costs
    if not read_only:
        try:
            _lock = acquire_lock()
        except ProcessLockBusy as exc:
            print(f"🔒 {exc} — thoát để tránh đặt lệnh hai lần.")
            return 0

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

    max_notional = a.max_order_notional
    if max_notional is None:
        try:
            bal = client.balance_usdt()
            eq = bal.get("wallet_balance", 1000.0) + bal.get("unrealized_pnl", 0.0)
            max_notional = max(1000.0, eq * cfg.leverage * 0.35)
        except Exception:
            max_notional = 1000.0

    pipe = CrossSectionalLivePipeline(
        config=cfg, client=client,
        router=BinanceOrderRouter(client=client, max_order_notional=max_notional, dry_run=dry),
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

    if a.sync:
        return sync_state(pipe)

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

    from datetime import datetime

    def _execute_cycle():
        res = pipe.run_once(skip_data_refresh=a.skip_refresh, force=a.force)

        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{now_str}] Hành động   : {res.get('action')}")
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
            print(f"Đã gửi      : {res.get('submitted', 0)} | cắn giá: "
                  f"{res.get('taker_fallback', 0)} | chưa khớp: {res.get('unfilled', 0)}")
        if res.get("plan_failures"):
            print(f"⚠️  {len(res['plan_failures'])} lệnh KHÔNG đặt được: "
                  f"{', '.join(f['symbol'] for f in res['plan_failures'])}")
        if res.get("uncancelled"):
            print(f"🔴 CÒN LỆNH SỐNG không xác nhận huỷ: {', '.join(res['uncancelled'])}")
        if res.get("action") == "HEARTBEAT":
            print(f"Đã qua      : {res['hours_since_rebalance']:.1f}h / "
                  f"{res['rebalance_hours']:.0f}h — còn {res['hours_remaining']:.1f}h")
            print(f"Vị thế      : {res['n_positions']} | "
                  f"drawdown {res['drawdown']*100:.2f}%")
            print("Muốn cân ngay: thêm cờ --force")
        nt = res.get("neutrality")
        if nt:
            mark = "✅" if nt["ok"] else "🔴"
            lev = f" | đòn bẩy {nt['leverage']:.2f}x" if nt.get("leverage") else ""
            print(f"Sổ          : {mark} net {nt['net_ratio']*100:+.2f}% gross "
                  f"(trần ±{nt['tolerance']*100:.0f}%) | gross ${nt['gross']:,.0f}{lev}")

        # Tiến độ tới cổng cho phép bơm tiền thật — in ở MỌI lượt để không ai
        # phải tự nhớ. Cổng chỉ nói đường ống đã ổn, không nói chiến lược có lãi.
        try:
            sys.path.insert(0, "scripts")
            from readiness_gate import load_records
            rs = [r for r in load_records("artifacts/execution_log.jsonl") if r.n_orders > 0]
            streak = 0
            for r in reversed(rs):
                if r.clean:
                    streak += 1
                else:
                    break
            need = max(0, 3 - streak)
            tag = "✅ ĐẠT" if need == 0 else f"còn {need} lượt (~{need*3} ngày)"
            print(f"Cổng tiền thật: {streak}/3 lượt sạch liên tiếp — {tag}")
        except Exception:
            pass

        # Gửi thông báo Telegram nếu đã cấu hình
        try:
            from aegis.monitoring.alerts import TelegramNotifier
            notifier = TelegramNotifier()
            if notifier.is_configured:
                if res.get("action") == "HALT":
                    notifier.send_circuit_breaker_alert(
                        reason=res.get("reason", "HALT"),
                        detail=res.get("detail", ""),
                        drawdown=res.get("drawdown", 0.0),
                        equity=res.get("equity"),
                    )
                elif res.get("orders"):
                    notifier.send_order_alert(
                        mode=mode,
                        equity=res.get("equity", 0.0),
                        orders=res.get("orders", []),
                        turnover=res.get("turnover", 0.0),
                        capacity=res.get("capacity"),
                        data_age_hours=res.get("data_age_hours"),
                    )
        except Exception as exc:
            logging.warning("Không thể gửi thông báo Telegram: %s", exc)

        return res

    if not a.loop:
        _execute_cycle()
        return 0

    # Chế độ LOOP chạy ngầm:
    print(f"🚀 Bắt đầu chế độ Daemon Loop ({mode}) — Kiểm tra mỗi {a.loop_interval_mins:.0f} phút, tái cân bằng mỗi {cfg.rebalance_hours:.0f}h...")
    try:
        from aegis.monitoring.alerts import TelegramNotifier
        notifier = TelegramNotifier()
        if notifier.is_configured:
            notifier.send_message(
                f"🚀 <b>Aegis Trading Bot đã bắt đầu chạy ngầm ({mode})!</b>\n"
                f"⏰ <i>Kiểm tra mỗi {a.loop_interval_mins:.0f} phút, tái cân bằng mỗi {cfg.rebalance_hours:.0f}h.</i>\n"
                f"💰 <b>Vốn hiện tại:</b> ${pipe.client.balance_usdt().get('wallet_balance', 0):,.2f} USDT"
            )
    except Exception:
        pass

    while True:
        try:
            state = pipe.store.load()
            now_ms = int(time.time() * 1000)
            hours_since_last = (now_ms - state.last_rebalance_ms) / 3_600_000 if state.last_rebalance_ms else 999.0

            if state.rebalance_count == 0 or hours_since_last >= cfg.rebalance_hours:
                print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Đến hạn tái cân bằng ({hours_since_last:.1f}h >= {cfg.rebalance_hours:.0f}h)...")
                _execute_cycle()
            else:
                bal = pipe.client.balance_usdt()
                equity = bal["wallet_balance"] + bal["unrealized_pnl"]
                dd = state.drawdown(equity)
                print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Heartbeat: Equity ${equity:,.2f} | uPnL ${bal['unrealized_pnl']:+,.2f} | Đã qua {hours_since_last:.1f}/{cfg.rebalance_hours:.0f}h | DD: {dd*100:.2f}%")
                blocked = pipe._check_circuit_breaker(state, equity, now_ms)
                if blocked:
                    print(f"⚠️ Circuit breaker kích hoạt: {blocked}")
                    from aegis.monitoring.alerts import TelegramNotifier
                    notifier = TelegramNotifier()
                    if notifier.is_configured:
                        notifier.send_circuit_breaker_alert(
                            reason="CIRCUIT_BREAKER", detail=str(blocked), drawdown=dd, equity=equity
                        )
        except KeyboardInterrupt:
            print("\nĐã nhận tín hiệu dừng bot.")
            break
        except Exception as exc:
            logging.error("Lỗi trong chu trình daemon: %s", exc)
            try:
                from aegis.monitoring.alerts import TelegramNotifier
                notifier = TelegramNotifier()
                if notifier.is_configured:
                    notifier.send_anomaly_alert(title="Lỗi tiến trình ngầm", message=str(exc), level="ERROR")
            except Exception:
                pass

        time.sleep(a.loop_interval_mins * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())

