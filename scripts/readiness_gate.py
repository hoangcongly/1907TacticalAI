#!/usr/bin/env python3
"""
CỔNG CHẤT LƯỢNG — trả lời đúng một câu: đã được phép bơm tiền thật chưa?

Tiêu chí do con người quyết, nhưng việc CHẤM phải do máy làm. Nếu để mắt người
nhìn nhật ký rồi kết luận "trông ổn", ta sẽ luôn kết luận là ổn — nhất là khi đang
sốt ruột muốn vào tiền. Vì vậy tiêu chí ở đây là nhị phân và kiểm chứng được.

ĐIỀU KIỆN: `N` lượt tái cân bằng LIÊN TIẾP gần nhất đều SẠCH, trong đó sạch nghĩa là
  1. mọi lệnh trong kế hoạch đều đặt được          (F22)
  2. không còn lệnh sống ngoài tầm kiểm soát       (F21)
  3. thực thi không ném lỗi                        (F24)
  4. danh mục trung lập trong trần                 (F25)
  5. đòn bẩy gộp đúng mục tiêu                     (F29)
  6. không ai phải vào sửa tay

Bối cảnh vì sao cần cổng này: tính tới 11/09/2026, số lượt chạy đúng mà không cần
can thiệp tay là **0 trên 4**, và tốc độ phát hiện lỗi T0 vẫn là 10 lỗi trong 2 ngày
(F21–F30) — đường cong chưa đi ngang. Cổng này chỉ mở khi đường cong đó thực sự phẳng.

CỔNG NÀY KHÔNG NÓI GÌ VỀ LỢI NHUẬN. Nó chỉ nói đường ống đã thôi vỡ. Lợi nhuận cần
~891 ngày mới có ý nghĩa thống kê (xem `docs/upgrade_v3_report.md` §7).

    python scripts/readiness_gate.py            # chấm 3 lượt gần nhất
    python scripts/readiness_gate.py --n 5      # khắt khe hơn
"""
import argparse
import json
import pathlib
import sys
from typing import List

import pandas as pd

from aegis.core.execution_log import ExecutionLog, ExecutionRecord

STATE_PATH = "artifacts/live_state.json"


def load_records(path: str) -> List[ExecutionRecord]:
    p = pathlib.Path(path)
    if not p.is_file():
        return []
    out = []
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        # Bản ghi cũ thiếu trường mới -> lọc về đúng chữ ký dataclass.
        allowed = ExecutionRecord.__dataclass_fields__.keys()
        out.append(ExecutionRecord(**{k: v for k, v in d.items() if k in allowed}))
    return out


def reasons(r: ExecutionRecord) -> List[str]:
    out = []
    if r.plan_failures:
        out.append(f"{r.plan_failures} lệnh không đặt được")
    if r.uncancelled:
        out.append(f"{r.uncancelled} lệnh sống không huỷ được")
    if r.exec_error:
        out.append(f"thực thi lỗi: {r.exec_error[:40]}")
    if r.manual_intervention:
        out.append("có can thiệp tay")
    if r.net_ok is False:
        out.append(f"lệch hướng {r.net_exposure*100:+.1f}%")
    if r.gross_ok is False:
        lev = f"{r.leverage:.2f}x" if r.leverage else "?"
        out.append(f"đòn bẩy sai {lev}")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3, help="Số lượt sạch liên tiếp cần có")
    ap.add_argument("--log", default="artifacts/execution_log.jsonl")
    a = ap.parse_args(argv)

    recs = load_records(a.log)
    # Chỉ tính các lượt CÓ GIAO DỊCH; lượt giám sát (HEARTBEAT) không ghi nhật ký.
    recs = [r for r in recs if r.n_orders > 0]

    print("=" * 88)
    print(f"CỔNG CHẤT LƯỢNG — cần {a.n} lượt tái cân bằng SẠCH liên tiếp")
    print("=" * 88)

    if not recs:
        print("Chưa có lượt tái cân bằng nào được ghi nhận.")
        return 1

    print(f"{'thời điểm (UTC)':<21}{'môi trường':<11}{'lệnh':>5}{'maker':>8}{'chi phí':>9}"
          f"{'net':>8}{'đòn bẩy':>9}{'sạch':>7}  ghi chú")
    print("-" * 88)
    for r in recs[-max(a.n * 2, 6):]:
        lev = f"{r.leverage:.2f}x" if r.leverage else "  n/a"
        why = "; ".join(reasons(r))
        print(f"{pd.to_datetime(r.timestamp_ms, unit='ms').strftime('%Y-%m-%d %H:%M:%S'):<21}"
              f"{'testnet' if r.testnet else 'MAINNET':<11}{r.n_orders:>5}"
              f"{r.maker_ratio*100:>7.0f}%{r.realized_cost_bps:>8.2f}bp"
              f"{r.net_exposure*100:>+7.2f}%{lev:>9}"
              f"{'  ✅' if r.clean else '  ❌':>7}  {why}")

    streak = 0
    for r in reversed(recs):
        if r.clean:
            streak += 1
        else:
            break

    print("-" * 88)
    print(f"Chuỗi lượt sạch liên tiếp: {streak} / {a.n}")

    # Trạng thái sổ hiện tại phải sạch nữa, không chỉ lịch sử.
    state_ok, state_msg = True, "ổn"
    sp = pathlib.Path(STATE_PATH)
    if sp.is_file():
        st = json.loads(sp.read_text())
        if st.get("is_dead"):
            state_ok, state_msg = False, "kill switch đang bật"
        elif st.get("last_error"):
            state_ok, state_msg = False, f"lỗi treo: {str(st['last_error'])[:60]}"
    print(f"Trạng thái sổ hiện tại   : {'✅' if state_ok else '❌'} {state_msg}")

    passed = streak >= a.n and state_ok
    print("=" * 88)
    if passed:
        print("✅ ĐẠT CỔNG — đường ống đã ổn định. Được phép cân nhắc bơm tiền thật.")
        print()
        print("Nhắc lại cho rõ: cổng này chỉ nói ĐƯỜNG ỐNG THÔI VỠ.")
        print("Nó KHÔNG nói chiến lược có lãi. Lợi nhuận cần ~891 ngày mới có ý nghĩa.")
        print()
        print("Trước khi chuyển mainnet còn phải làm:")
        print("  1. Tạo API key mainnet — BẬT futures, TẮT rút tiền, khoá theo IP")
        print("  2. Thêm BINANCE_API_KEY / BINANCE_API_SECRET vào .env")
        print("  3. Lọc lại universe theo min notional mainnet (BTCUSDT cần $100)")
        print("  4. Đặt marginType=ISOLATED và leverage cho từng cặp")
        print("  5. Chạy `--live --mainnet` một lượt và soi kỹ lượt đầu")
    else:
        need = a.n - streak
        print(f"⛔ CHƯA ĐẠT — còn thiếu {need} lượt sạch liên tiếp"
              + ("" if state_ok else " và phải xử lý lỗi đang treo trước"))
        print(f"   Ở chu kỳ 72h, {need} lượt ≈ {need*3} ngày nữa.")
    print("=" * 88)
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
