#!/usr/bin/env python3
"""
KIỂM TRA TRƯỚC KHI BAY — mọi thứ phải đúng TRƯỚC lượt tái cân bằng, không phải sau.

`readiness_gate.py` chấm SAU khi lượt đã chạy: nó nói lượt vừa rồi có sạch không. Hữu
ích để đếm, vô dụng để phòng. Script này chạy TRƯỚC và trả lời câu khác: "nếu tái cân
bằng ngay bây giờ, có gì đang hỏng sẵn không?"

VÌ SAO ĐIỀU NÀY NẰM TRÊN ĐƯỜNG GĂNG: cổng cần 3 lượt sạch LIÊN TIẾP. Một lượt hỏng
không chỉ mất lượt đó — nó reset chuỗi về 0 và đẩy lùi 9 ngày. Với chu kỳ 72h, mỗi
lượt là một cơ hội đắt đỏ, và gần như mọi nguyên nhân làm hỏng một lượt đều PHÁT HIỆN
ĐƯỢC TRƯỚC: lệch đồng hồ, lệnh mồ côi, sổ không khớp, dữ liệu cũ, universe không dựng
được, vốn không đủ min notional.

Mọi kiểm tra ở đây đều CHỈ ĐỌC. Script này không đặt lệnh, không sửa trạng thái, không
huỷ gì cả — nên chạy nó bao nhiêu lần cũng an toàn.

    python scripts/preflight.py
    python scripts/preflight.py --mainnet
    python scripts/preflight.py --quiet     # chỉ in các mục KHÔNG đạt
"""
import argparse
import json
import pathlib
import sys
import time
import warnings
from typing import Any, Dict, List, Optional, Tuple

warnings.filterwarnings("ignore")

PASS, WARN, FAIL = "✅", "⚠️ ", "❌"


class Check:
    """Một mục kiểm tra: tên, kết quả, chi tiết, và có CHẶN hay không."""

    def __init__(self, name: str, blocking: bool = True):
        self.name, self.blocking = name, blocking
        self.status, self.detail = FAIL, "chưa chạy"

    def ok(self, detail: str = ""):
        self.status, self.detail = PASS, detail
        return self

    def warn(self, detail: str):
        self.status, self.detail = WARN, detail
        return self

    def fail(self, detail: str):
        self.status, self.detail = FAIL, detail
        return self

    @property
    def blocked(self) -> bool:
        return self.blocking and self.status == FAIL


def _run(name: str, fn, blocking: bool = True) -> Check:
    """
    Chạy một kiểm tra, BẮT MỌI ngoại lệ và biến nó thành kết quả FAIL.

    Một script kiểm tra mà tự nó sập giữa chừng còn tệ hơn không có script: nó dừng ở
    mục thứ ba và không ai biết sáu mục còn lại thế nào.
    """
    c = Check(name, blocking)
    try:
        fn(c)
    except Exception as exc:
        c.fail(f"{type(exc).__name__}: {exc}")
    return c


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mainnet", action="store_true")
    ap.add_argument("--quiet", action="store_true", help="chỉ in mục không đạt")
    ap.add_argument("--config", default="artifacts/strategy_v3.json")
    #: Đòn bẩy thật KHÔNG nằm trong file cấu hình — `run_daily.py` nhận nó qua cờ
    #: `--leverage` rồi gán đè (`cfg.leverage = a.leverage`). Nghĩa là preflight đọc
    #: cùng file cấu hình vẫn thấy giá trị MẶC ĐỊNH 2.0x, trong khi daemon có thể
    #: đang chạy 5.0x. Ba phép kiểm dùng `cfg.leverage` — vốn mỗi vị thế, ký quỹ,
    #: và dòng hiển thị — vì thế đo sai đại lượng mà không báo gì.
    #:
    #: ĐÃ QUAN SÁT 21/09/2026: daemon chạy 50 vị thế ở 5.0x, preflight báo "2.0x,
    #: $229.99/vị thế" trong khi thực tế là $574.97/vị thế. Lần đó cả hai đều vượt
    #: ngưỡng $6 nên phán quyết không đổi — nhưng đó là may, không phải thiết kế.
    ap.add_argument("--leverage", type=float, default=None,
                    help="Đòn bẩy daemon đang chạy (khớp cờ --leverage của run_daily.py). "
                         "Bỏ trống = dùng giá trị trong file cấu hình.")
    a = ap.parse_args(argv)

    from aegis.core.state_store import StateStore
    from aegis.data.ingestion.binance_rest import BinanceFuturesREST
    from aegis.pipelines.xs_live_pipeline import CrossSectionalLivePipeline, LiveConfig

    checks: List[Check] = []
    ctx: Dict[str, Any] = {}

    # ---- 1. Cấu hình & khoá ------------------------------------------------
    def _cfg(c: Check):
        cfg = LiveConfig.from_artifacts(a.config)
        if a.leverage is not None:
            cfg.leverage = a.leverage
        ctx["cfg"] = cfg
        src = "cờ --leverage" if a.leverage is not None else "mặc định cấu hình — CÓ THỂ SAI"
        c.ok(f"engine={cfg.engine}, {cfg.n_positions} vị thế, {cfg.leverage:.1f}x "
             f"({src}), chu kỳ {cfg.rebalance_hours:.0f}h, "
             f"chờ maker {cfg.passive_wait_s:.0f}s")
    checks.append(_run("cấu hình nạp được", _cfg))
    if checks[-1].blocked:
        _report(checks, a.quiet)
        return 1

    cfg: LiveConfig = ctx["cfg"]

    def _client(c: Check):
        cl = BinanceFuturesREST(testnet=not a.mainnet)
        ctx["client"] = cl
        if not cl.ping():
            raise RuntimeError("ping thất bại")
        c.ok(f"{'MAINNET' if a.mainnet else 'testnet'} phản hồi")
    checks.append(_run("sàn kết nối được", _client))
    if checks[-1].blocked:
        _report(checks, a.quiet)
        return 1

    client = ctx["client"]

    # ---- 2. Đồng hồ --------------------------------------------------------
    def _clock(c: Check):
        """
        Lệch đồng hồ là nguyên nhân IM LẶNG kinh điển làm hỏng lệnh: Binance từ chối
        mọi request có `timestamp` lệch quá `recvWindow` (5000ms), và lỗi trả về nói
        về chữ ký chứ không nói về đồng hồ. Kiểm trước thì mất 1 giây; phát hiện sau
        thì mất một lượt tái cân bằng.
        """
        # `sync_time()` trả về ĐỘ LỆCH (server - local), KHÔNG phải giờ server.
        # Đọc nhầm ngữ nghĩa này cho ra con số vô nghĩa cỡ một epoch — đã dính.
        skew = float(client.sync_time())
        ctx["skew_ms"] = skew
        if abs(skew) > 3000:
            c.fail(f"lệch {skew:+.0f}ms — vượt recvWindow 5000ms, lệnh sẽ bị từ chối")
        elif abs(skew) > 1000:
            c.warn(f"lệch {skew:+.0f}ms — còn chạy được nhưng nên đồng bộ NTP")
        else:
            c.ok(f"lệch {skew:+.0f}ms")
    checks.append(_run("đồng hồ đồng bộ", _clock))

    # ---- 3. Số dư & sức chứa vốn ------------------------------------------
    def _equity(c: Check):
        bal = client.balance_usdt()
        eq = bal["wallet_balance"] + bal["unrealized_pnl"]
        ctx["equity"] = eq
        per = eq * cfg.leverage / cfg.n_positions
        need = 5.0 * cfg.min_notional_safety
        if per < need:
            c.fail(f"${eq:.2f} x {cfg.leverage:.1f}x / {cfg.n_positions} = ${per:.2f}/vị thế "
                   f"< ${need:.2f} cần thiết — lệnh sẽ bị bỏ, danh mục không trung lập")
        else:
            c.ok(f"${eq:,.2f} -> ${per:.2f}/vị thế (cần >= ${need:.2f})")
    checks.append(_run("vốn đủ cho số vị thế", _equity))

    def _margin(c: Check):
        bal = client.balance_usdt()
        avail, eq = bal["available_balance"], ctx.get("equity", 0.0)
        need = eq * cfg.leverage / 20.0     # ký quỹ ban đầu ~5% ở đòn bẩy 20x mỗi cặp
        if avail <= 0:
            c.fail("số dư khả dụng = 0 — không đặt được lệnh mới")
        elif avail < need:
            c.warn(f"khả dụng ${avail:,.2f}, ước cần ~${need:,.2f} cho gross mục tiêu")
        else:
            c.ok(f"khả dụng ${avail:,.2f}")
    checks.append(_run("ký quỹ khả dụng", _margin, blocking=False))

    # ---- 4. Lệnh mồ côi (F21) ---------------------------------------------
    def _orphans(c: Check):
        """
        Lệnh sống sót từ lượt trước là nguyên nhân trực tiếp của sự cố 10/09: một lệnh
        post-only COTIUSDT nằm trên sàn 3h39 và làm danh mục lệch +35% suốt 27 giờ.
        """
        oo = client.open_orders()
        ctx["open_orders"] = oo
        if oo:
            syms = sorted({o["symbol"] for o in oo})
            c.fail(f"{len(oo)} lệnh đang sống trên sàn ({', '.join(syms[:5])}"
                   f"{'...' if len(syms) > 5 else ''}) — huỷ trước khi tái cân bằng")
        else:
            c.ok("không có lệnh treo")
    checks.append(_run("không có lệnh mồ côi", _orphans))

    # ---- 5. Đối chiếu sổ sách ---------------------------------------------
    def _recon(c: Check):
        store = StateStore()
        st = store.load()
        ctx["state"] = st
        ex = {p["symbol"]: float(p["positionAmt"]) for p in client.position_risk()
              if abs(float(p.get("positionAmt", 0))) > 0}
        internal = {k: v for k, v in st.positions.items() if abs(v) > 0}
        only_ex = set(ex) - set(internal)
        only_in = set(internal) - set(ex)
        qty_off = {s for s in set(ex) & set(internal)
                   if abs(ex[s] - internal[s]) > 1e-9 * max(1.0, abs(ex[s]))}
        if only_ex or only_in or qty_off:
            parts = []
            if only_ex:
                parts.append(f"chỉ có trên sàn: {sorted(only_ex)[:4]}")
            if only_in:
                parts.append(f"chỉ có trong sổ: {sorted(only_in)[:4]}")
            if qty_off:
                parts.append(f"lệch khối lượng: {sorted(qty_off)[:4]}")
            c.fail("; ".join(parts))
        else:
            c.ok(f"{len(ex)} vị thế khớp sổ")
    checks.append(_run("sổ sách khớp sàn", _recon))

    # ---- 6. Trạng thái nội bộ ---------------------------------------------
    def _state(c: Check):
        st = ctx.get("state")
        if st is None:
            raise RuntimeError("chưa đọc được trạng thái")
        now = int(time.time() * 1000)
        if st.is_dead:
            c.fail("kill switch ĐANG BẬT — hệ thống sẽ không giao dịch")
        elif st.frozen_until_ms > now:
            c.fail(f"đang đóng băng thêm {(st.frozen_until_ms-now)/3.6e6:.1f}h")
        elif st.last_error:
            c.fail(f"lỗi treo chưa xử lý: {str(st.last_error)[:70]}")
        else:
            dd = st.drawdown(ctx.get("equity", st.last_equity))
            h = (now - st.last_rebalance_ms) / 3.6e6
            c.ok(f"sạch | drawdown {dd*100:.1f}% | {h:.1f}/{cfg.rebalance_hours:.0f}h "
                 f"-> lượt tới sau {max(0.0, cfg.rebalance_hours-h):.1f}h")
    checks.append(_run("trạng thái nội bộ sạch", _state))

    # ---- 7. Khoá tiến trình ------------------------------------------------
    def _lock(c: Check):
        """
        Daemon còn sống không — đo bằng NHỊP TIM, không đo bằng mtime của file khoá.

        File khoá được tạo một lần rồi giữ nguyên suốt phiên, nên mtime của nó già đi
        dù tiến trình hoàn toàn khoẻ. Dùng nó làm tín hiệu sống cho ra cảnh báo giả
        (đã dính: "khoá cũ 22.3h" trong khi daemon vừa gửi nhịp tim 30 phút trước).
        Nhịp tim mới là thứ chứng minh tiến trình còn chạy.
        """
        log = pathlib.Path("logs/aegis_daemon.log")
        if not log.is_file():
            return c.warn("không thấy nhật ký daemon")
        beats = [ln for ln in log.read_text(errors="ignore").splitlines()
                 if "Heartbeat" in ln]
        if not beats:
            return c.warn("nhật ký chưa có nhịp tim nào")
        import re
        m = re.search(r"\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\]", beats[-1])
        if not m:
            return c.warn("không đọc được mốc nhịp tim cuối")
        import datetime
        last = datetime.datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S")
        age_min = (datetime.datetime.now() - last).total_seconds() / 60.0
        if age_min > 90:
            c.fail(f"nhịp tim cuối {age_min/60:.1f}h trước — daemon có thể đã chết")
        elif age_min > 45:
            c.warn(f"nhịp tim cuối {age_min:.0f} phút trước (chu kỳ 30 phút)")
        else:
            c.ok(f"nhịp tim {age_min:.0f} phút trước")
    checks.append(_run("daemon còn sống", _lock, blocking=False))

    # ---- 8. Dữ liệu & universe --------------------------------------------
    def _data(c: Check):
        pipe = CrossSectionalLivePipeline(config=cfg, client=client, dry_run=True)
        ctx["pipe"] = pipe
        keep = pipe.resolve_universe()
        ctx["universe"] = keep
        if len(keep) < 10:
            c.fail(f"universe sau lọc còn {len(keep)} cặp — quá ít để market-neutral")
        else:
            c.ok(f"{len(keep)} cặp qua bộ lọc")
    checks.append(_run("universe dựng được", _data))

    def _fresh(c: Check):
        """
        Kiểm ĐƯỜNG LÀM MỚI, không kiểm dữ liệu trên đĩa.

        Dữ liệu trên đĩa CỐ Ý cũ giữa hai lượt: `run_once` gọi `refresh_data` ngay
        trước khi tính tín hiệu, nên ở giữa chu kỳ 72h mọi file đều ~35h tuổi và đó là
        bình thường. Báo động vì điều đó là báo động giả — và một script preflight hay
        báo động giả sẽ bị người ta bỏ qua, rồi bỏ qua luôn cả báo động thật.

        Thứ THỰC SỰ hỏng được là đường lấy dữ liệu. Nên kiểm bằng cách gọi thật một
        nến từ sàn và xem nó có mới không.
        """
        import glob

        import pandas as pd

        sym = (ctx.get("universe") or ["BTCUSDT"])[0]
        kl = client.klines(sym, "1h", limit=2)
        if not kl:
            raise RuntimeError(f"sàn không trả nến nào cho {sym}")
        live_ms = int(kl[-1][0])
        live_age = (time.time() * 1000 - live_ms) / 3.6e6

        files = glob.glob("data/binance/*_1h.parquet")
        disk_age = float("inf")
        if files:
            newest = max(pd.read_parquet(f, columns=["timestamp_ms"])["timestamp_ms"].max()
                         for f in files[:30])
            disk_age = (time.time() * 1000 - newest) / 3.6e6

        if live_age > 3.0:
            c.fail(f"nến MỚI NHẤT TRÊN SÀN đã {live_age:.1f}h tuổi — nguồn dữ liệu hỏng")
        else:
            c.ok(f"sàn trả nến {live_age:.1f}h tuổi (đĩa {disk_age:.1f}h — sẽ được làm "
                 f"mới ngay trước lượt tái cân bằng, cũ ở giữa chu kỳ là BÌNH THƯỜNG)")
    checks.append(_run("nguồn dữ liệu sống", _fresh))

    # ---- 9. Bộ lọc sàn cho các cặp đang nắm -------------------------------
    def _filters(c: Check):
        st = ctx.get("state")
        syms = sorted(set((st.positions if st else {}).keys()))[:8]
        if not syms:
            return c.ok("chưa có vị thế nào để kiểm")
        bad = []
        for s in syms:
            try:
                f = client.symbol_filters(s)
                if not f or f.get("tick_size", 0) <= 0:
                    bad.append(s)
            except Exception:
                bad.append(s)
        c.fail(f"không lấy được bộ lọc: {bad}") if bad else c.ok(f"{len(syms)} cặp OK")
    checks.append(_run("bộ lọc sàn đọc được", _filters))

    # ---- 10. Chính sách mục tiêu (nếu bật) --------------------------------
    def _overlay(c: Check):
        oc = cfg.goal_overlay
        if oc is None or not oc.enabled:
            return c.ok("tầng phủ mục tiêu: TẮT")
        p = pathlib.Path(oc.policy_path)
        if not p.is_file() or not p.with_suffix(".meta.json").is_file():
            c.fail(f"bật nhưng thiếu {p} — chạy scripts/build_goal_policy.py")
        elif oc.start_equity <= 0 or oc.deadline_ms <= 0:
            c.warn("bật nhưng chưa mở ván (start_equity/deadline chưa đặt) — sẽ không làm gì")
        else:
            c.ok(f"bật, derisk_only={oc.derisk_only}")
    checks.append(_run("tầng phủ mục tiêu", _overlay, blocking=False))

    return _report(checks, a.quiet)


def _report(checks: List[Check], quiet: bool) -> int:
    print("=" * 92)
    print("KIỂM TRA TRƯỚC KHI BAY — nếu tái cân bằng NGAY BÂY GIỜ thì có gì hỏng sẵn không")
    print("=" * 92)
    shown = [c for c in checks if not quiet or c.status != PASS]
    for c in shown:
        print(f"{c.status} {c.name:<28} {c.detail}")
    if quiet and not shown:
        print(f"{PASS} tất cả {len(checks)} mục đạt")

    blocked = [c for c in checks if c.blocked]
    warns = [c for c in checks if c.status == WARN]
    # Mục KHÔNG chặn nhưng vẫn ❌ — phải hiện ra trong kết luận.
    # Bản đầu in "✅ SẴN SÀNG — 13 mục đạt" ngay dưới một dòng ❌ 'daemon có thể đã
    # chết'. Một bản tóm tắt tự mâu thuẫn với chính bảng ngay trên nó sẽ bị bỏ qua,
    # và lần sau người ta bỏ qua luôn cả cảnh báo thật.
    soft_fail = [c for c in checks if c.status == FAIL and not c.blocking]
    print("-" * 92)
    if blocked:
        print(f"⛔ {len(blocked)} MỤC CHẶN — sửa trước khi để lượt tái cân bằng chạy:")
        for c in blocked:
            print(f"   • {c.name}: {c.detail}")
        print("\n   Một lượt hỏng không chỉ mất lượt đó — nó reset chuỗi sạch về 0")
        print("   và đẩy cổng chất lượng lùi 9 ngày.")
    elif soft_fail:
        print(f"⚠️  CHẠY ĐƯỢC NHƯNG CÓ {len(soft_fail)} MỤC HỎNG (không chặn lượt này):")
        for c in soft_fail:
            print(f"   • {c.name}: {c.detail}")
    else:
        n_ok = len(checks) - len(warns) - len(soft_fail)
        print(f"✅ SẴN SÀNG — {n_ok} mục đạt"
              + (f", {len(warns)} cảnh báo" if warns else ""))
    print("=" * 92)
    return 1 if blocked else (2 if soft_fail else 0)


if __name__ == "__main__":
    sys.exit(main())
