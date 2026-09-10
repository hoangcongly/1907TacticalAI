# Đường tiền (money path)

Mọi thay đổi chạm vào một trong các bước dưới đây đều ẢNH HƯỞNG TRỰC TIẾP tới tiền thật.
Bắt buộc chạy `pytest tests/execution tests/risk tests/pipelines` sau khi sửa.

## Sản phẩm là PERPETUAL FUTURES (USDⓈ-M), không phải margin spot
"Isolated Margin" trong code = **chế độ ký quỹ cô lập của Binance Futures**, không phải
giao dịch ký quỹ spot. Bằng chứng: `liquidation_layer.py` (giá thanh lý/MMR),
`funding_accrual.py` (funding 8h — chỉ perp mới có), `pnl.py` (nhánh `LIQUIDATION`, cờ Coin-M).

Hệ quả: mọi thay đổi PnL PHẢI tính đủ 4 khoản — **phí + funding + thanh lý + trượt giá**.

---

## A. Luồng RESEARCH (dựng model + bảng Kelly)
`src/aegis/pipelines/research_pipeline.py` — `ResearchPipeline.run()`

```
signal_bars (SignalBarSchema, 19 cột)
  -> _ensure_candidate_features()      : thêm hl_spread/ret_1/ret_5/vol_ratio   ⚠️ F3
  -> compute_dynamic_cusum_thresholds  : labeling/cusum_events.py
  -> filter_cusum_events_dynamic       : -> event_indices
  -> generate_meta_labels_triple_barrier : labeling/triple_barrier.py -> y, t1
  -> compute_sample_weights            : labeling/sample_weights.py
  -> [feature selection]               : ⚠️ chạy trên TOÀN BỘ data -> rò rỉ lựa chọn
                                         ⚠️ fast_mode=True mặc định -> BỎ QUA hẳn
  -> WeightedBootstrapForestClassifier.fit()
  -> build_calibrated_classifier       : isotonic + PurgedKFold
  -> CPCVPipeline.run()                : -> bảng Kelly + metrics   ⚠️ F5
  -> xuất artifacts/: model.pkl, selected_features.json, kelly_tables.json, metadata.json
```

## B. Luồng CPCV (dựng bảng Kelly + đo OOS)
`src/aegis/pipelines/cpcv_pipeline.py` — `CPCVPipeline.run()`

| dòng | bước | ghi chú |
|---|---|---|
| 149 | `t1 = t0 + 10` | ⚠️ **F16** — lệnh giữ tới 120 nến |
| 166 | vòng lặp fold | ⚠️ **F5** — `train_idx` không dùng, KHÔNG fit gì |
| 188 | `run_trailing_exit_for_oos_event` | mô phỏng thoát lệnh |
| 192 | `p_i=pt` (= `p_trend`) | ⚠️ **F4** — sai đại lượng index bảng Kelly |
| 217 | `finalize_trade_record` | ⚠️ không truyền funding |
| 232 | `build_empirical_kelly_table_v2` | -> `kelly_follow` / `kelly_fade` |
| 242 | `sharpe * sqrt(252)` | ⚠️ **F15** |
| 250 | `variance_of_srs=np.var(returns)` | ⚠️ **F15** |

## C. Luồng LIVE (nơi tiền thật chảy)
`src/aegis/pipelines/live_pipeline.py` — `AegisLivePipeline.on_bar()` :262

| dòng | bước | trạng thái |
|---|---|---|
| 285 | **BƯỚC 0** mark-to-market | ✅ F7 |
| 291 | cộng dồn funding perp | ✅ FUTURES-1 |
| 310 | uPnL = PnL giá − funding | ✅ |
| 316 | circuit breaker trên `margin_balance` | ✅ F7 |
| 330 | **BƯỚC 1** quản lý vị thế | |
| 340 | **kiểm tra chạm giá thanh lý** (ưu tiên tuyệt đối) | ✅ F8 |
| 355 | SL/trailing long — trượt giá | ✅ F12 |
| 364 | `high_watermark` cập nhật nhưng KHÔNG DÙNG | ⚠️ **F14** |
| 400 | circuit breaker Tier≥2 buộc đóng lệnh | ✅ F7 |
| 410 | quyết toán qua `compute_realized_pnl` dùng chung | ✅ F10 |
| 459 | **BƯỚC 2** tìm cơ hội mở lệnh | |
| 478 | CUSUM trigger | ⚠️ `cusum_trig` tính rồi vứt; hướng lấy từ `trend_score` |
| 497 | `bar.get(col, 0.0)` | ⚠️ **F3** — zero-hoá feature |
| 504 | `p_i = predict_proba[0,1]` | ⚠️ **F4** |
| 507 | `_lookup_kelly` chọn đúng bảng follow/fade | ✅ F13 |
| 519 | `compute_position_size` | |
| 560 | pre-flight thanh lý (`validate_leverage_against_sl`) | ✅ F8 |
| 600 | kiểm tra đủ ký quỹ | ✅ |

## D. Engine PnL — DUY NHẤT một nơi
`src/aegis/execution/pnl.py` :12 `compute_realized_pnl()`

Mọi tính toán tiền PHẢI đi qua đây. Hai nhánh:
- `LIQUIDATION` :64 -> lỗ = `-(margin)`, chỉ trừ phí vào lệnh (chống đếm kép funding)
- thường :93 -> `gross - fee_entry - fee_exit - funding`

`exit_notional = size * (exit/entry)` ✅ F11 (bản cũ dùng `size + gross` — sai cho Short).

## E. Tầng bảo vệ vốn (theo thứ tự ưu tiên khi kích hoạt)
1. **Thanh lý** `liquidation_layer.py` — sàn cưỡng chế, không thể tránh
2. **Circuit breaker** `risk/circuit_breaker.py` — 5% giảm nửa / 10% đóng băng 24h / 15% kill
3. **Kelly + Half-Kelly** `sizing/kelly_empirical.py` — λ=0.5
4. **`max_safe_leverage`** — chặn cứng ở `compute_position_size`
5. **SL / trailing** `labeling/trailing_exit.py`
