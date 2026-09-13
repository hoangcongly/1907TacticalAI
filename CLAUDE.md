# Aegis Trading System

Hệ thống giao dịch **perpetual futures USDⓈ-M** (KHÔNG phải margin spot).
Python 3.10+, `src/aegis/`, ~12k dòng, 371 test.

## 🆕 Nâng cấp v3 (2026-09-10) — ĐỌC `docs/upgrade_v3_report.md` TRƯỚC KHI SỬA CHIẾN LƯỢC
Ba điều bắt buộc biết trước khi chạm vào code chiến lược:
1. **Chi phí thật là 4-5bp/chiều, không phải 1bp.** Nghiên cứu cũ giả định 1bp và vì
   thế cao hơn thực tế 0.1-0.35 Sharpe. Dùng `backtest_v2.estimate_cost_bps`.
2. **IC và chênh lệch decile có thể NGƯỢC DẤU** khi quan hệ không đơn điệu — đã xảy ra
   thật ở đây. Chọn tín hiệu bằng `ic_analysis.decile_spread`, không bằng IC.
3a. **Basis trade** (`research/basis_trade.py`): edge ĐÃ CHẾT từ 2025 — funding đổi
   dấu (2024: +12,7% -> 2025: −2,2%). Trên 119 cặp, Sharpe holdout trung vị −0,54,
   chỉ 17% cấu hình dương. ĐỪNG bật lại nếu chưa phân rã theo năm. §12.
3b. **ĐÃ THỬ VÀ THUA, đừng làm lại**: hồi quy mặt cắt ngang kiểu CTREND
   (`research/xs_regression.py`) và LambdaRank (`research/rank_model.py`). Cả hai
   thua tầng gộp hiện có vì universe chỉ có ~41 tài sản/kỳ — ràng buộc là ĐỘ RỘNG,
   không phải thuật toán. Chi tiết `docs/upgrade_v3_report.md` §11.
3. **Kỳ vọng hợp lý là Sharpe 1.0-1.3 / 55-75%/năm ở gross 1.0x**, không phải 1.99 của
   train. Holdout v3: Sharpe **1.28**, ann **70.8%**, maxDD 36.9%, giữ 64% Sharpe train.
   Qua 32 cấu hình siêu tham số, holdout **100% dương**, trung vị 1.24.
4. **Đánh giá holdout phải tính NHÂN QUẢ trên toàn dòng thời gian rồi CẮT SAU.** Cắt
   lưới về holdout trước sẽ khởi động lại tầng gộp thích ứng — handicap mà live không
   gặp. Sai lệch đo được: Sharpe 0.71 (sai) so với 1.28 (đúng).

## 🧭 Điều hướng — ĐỌC TRƯỚC KHI TÌM CODE
Repo này có skill điều hướng riêng. **Đừng grep dò dẫm** — nạp nó rồi tra bảng:

```
.claude/skills/aegis-nav/SKILL.md
```

Nó chứa router "triệu chứng → file:line" cho mọi vùng của hệ thống, cộng 5 file
reference nạp theo nhu cầu: `defects.md` (lỗi F1–F16), `money-path.md` (đường tiền),
`invariants.md` (bất biến + lệnh kiểm chứng), `codemap.md` (bản đồ tự sinh),
`plan.md` (kế hoạch P0/P1/P2).

## ⚠️ Trạng thái thật của hệ thống
- **341 test xanh KHÔNG có nghĩa chiến lược sinh lời.** Test khoá tính ĐÚNG (nhân quả,
  kế toán, bất biến danh mục), không khoá được EDGE. Edge chỉ đo được ngoài mẫu.
- F1–F13 và **F21–F31** đã vá; F14–F16 còn lại (`references/defects.md`).
- **Sự cố thực thi 10/09/2026** sinh ra F21–F26: một ngoại lệ không bắt trong
  `submit_plan` làm hỏng cả lượt, để lại lệnh post-only sống 3h39 trên sàn và danh
  mục lệch **+35%** khỏi trung lập suốt 27 giờ.
- **Sự cố nhân đôi 11/09/2026** sinh ra F27–F28: báo giá lại đếm trùng phần đã khớp
  -> danh mục chạy **3,69x** thay vì 2,0x (F29: cổng chặn chỉ kiểm net nên không thấy). Đọc hai mục cuối `references/defects.md`
  trước khi sửa bất cứ thứ gì trong `oms/` hoặc `xs_live_pipeline.py`.
- **~20 file vẫn là stub** chỉ có docstring (`governance/l2_depth.py`,
  `validation/flat_plateau.py`, ...). Danh sách ở đầu `references/codemap.md`.
- **CHƯA ĐƯỢC BƠM TIỀN THẬT.** `scripts/readiness_gate.py` đang báo 0/3 lượt sạch.
  Chỉ mở cổng khi 3 lượt tái cân bằng liên tiếp đều sạch (~9 ngày ở chu kỳ 72h).
- **Holdout đã dùng 2 lần** (lần 2 chạy lại sau khi sửa lỗi đo lường, không đổi tham
  số nào). Bước tiếp theo bắt buộc: giao dịch giấy tiến về phía trước 4–8 tuần
  (`python scripts/run_daily.py --live` trên testnet), rồi `--costs` để đo tỷ lệ maker thật.

## Đường đi chiến lược v3 (một đường DUY NHẤT cho research và live)
```
dữ liệu -> research/signal_library.py     (26 tín hiệu / 5 họ, mỗi tín hiệu 1 giả thuyết)
        -> research/adaptive_combiner.py  (học dấu + trọng số từ QUÁ KHỨ, không đảo dấu tay)
        -> risk/portfolio.py              (chọn theo hạng -> trọng số -> trung lập -> trần)
        -> research/leverage.py           (mục tiêu biến động + van drawdown)
```
Backtest gọi `research/strategy_v2.py`; live gọi `xs_live_pipeline._compute_target_weights_v3`
— cả hai dùng **chung các hàm trên**. `tests/pipelines/test_research_live_parity_v3.py` khoá
bất biến này (sai số < 1e-9). Viết lại công thức ở tầng live = tái lập lỗi F3.

## Lệnh
```bash
python -m pytest -q                    # 371 test, ~135s
python scripts/attribution.py          # quy kết từng nâng cấp (train)
python scripts/stability.py            # chọn cấu hình theo ĐỘ ỔN ĐỊNH, không theo đỉnh
python scripts/diagnose_oos.py         # chẩn đoán suy giảm ngoài mẫu
python scripts/refresh_spreads.py      # đo lại spread sổ lệnh thật
python scripts/readiness_gate.py       # ĐÃ ĐƯỢC PHÉP BƠM TIỀN THẬT CHƯA?
python scripts/gen_codemap.py          # sinh lại bản đồ code sau refactor
python scripts/check_docs.py           # kiểm tra tham chiếu file:line trong docs còn đúng
make agent-sync                        # chạy cả hai (làm sau mỗi lần refactor)
grep -rn "\[FIX F" src/                # xem các lỗ hổng đã vá và vá ở đâu
```

## Quy ước
- Tham số chiến lược **chỉ** đến từ `config/aegis_canonical_parameters.yaml` qua
  `load_canonical_config()`. Không hardcode hằng số trong code.
- Mọi tính tiền đi qua **đúng một** engine: `src/aegis/execution/pnl.py:compute_realized_pnl`.
- Chi phí futures phải đủ 4 khoản: **phí + funding + thanh lý + trượt giá**.
- Feature phải nhân quả (chỉ dùng dữ liệu `<= t`) và dùng **chung một hàm** cho
  train lẫn inference.
- Bình luận/tài liệu viết bằng tiếng Việt, khớp với code hiện có.
- Vá lỗi thì đánh dấu `[FIX Fxx]` ngay tại chỗ sửa.
- **Số vị thế là ràng buộc CỨNG** (`PortfolioSpec.n_positions`), không phải `top_frac`.
  Vốn $38 x min notional $5 => tối đa ~12 vị thế ở đòn bẩy 2x. Mọi so sánh backtest phải
  cố định số vị thế, nếu không ta chỉ đang đo tác dụng của đa dạng hoá.
- **Không tinh chỉnh thêm trên holdout hiện tại** — đã dùng 2 lần cho v3. Kiểm định sạch
  còn lại duy nhất là giao dịch giấy tiến về phía trước.
