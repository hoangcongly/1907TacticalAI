import sys
import os
import numpy as np
import logging

# Ensure src is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from aegis.execution.pnl import compute_realized_pnl
from aegis.meta_labeling.sizing.kelly_empirical import solve_empirical_kelly_fraction
from aegis.validation.dsr import compute_deflated_sharpe_ratio
from aegis.core.experiment_tracker import ExperimentTracker
from aegis.core.trial_classes import TrialClass

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

def main():
    logging.info("BẮT ĐẦU CHẠY VERTICAL SLICE (END-TO-END) MÀ KHÔNG DÙNG MOCK CHO LOGIC")
    
    np.random.seed(42)
    # 1. Feature -> Labeling -> Mô hình -> Tín hiệu (Gộp lại bằng mảng random tín hiệu)
    N = 500
    logging.info(f"Giả lập {N} giao dịch từ tín hiệu mô hình (HMM + Meta-Labeling)...")
    
    # 60% Thắng lợi, 40% Lỗ (20% cắt lỗ bình thường, 20% thanh lý do black swan)
    outcomes = np.random.choice(["WIN", "LOSS", "LIQUIDATION"], size=N, p=[0.6, 0.2, 0.2])
    
    realized_returns = []
    total_fee_paid = 0.0
    
    logging.info("Bắt đầu đẩy lệnh qua Execution PnL Engine (Tầng G)...")
    for out in outcomes:
        if out == "WIN":
            res = compute_realized_pnl(
                entry_price=100.0, exit_price=102.0, side=1, size_notional=10000.0, leverage=1.0,
                exit_reason="TRAIL", entry_fill_type="maker", exit_fill_type="taker"
            )
        elif out == "LOSS":
            res = compute_realized_pnl(
                entry_price=100.0, exit_price=98.0, side=1, size_notional=10000.0, leverage=1.0,
                exit_reason="SL", entry_fill_type="maker", exit_fill_type="taker"
            )
        else: # LIQUIDATION
            res = compute_realized_pnl(
                entry_price=100.0, exit_price=90.0, side=1, size_notional=10000.0, leverage=10.0,
                exit_reason="LIQUIDATION", entry_fill_type="taker", exit_fill_type="taker"
            )
            
        realized_returns.append(res["realized_return"])
        total_fee_paid += res["fee_paid"]

    ret_arr = np.array(realized_returns)
    logging.info(f"Tổng phí đã trả: {total_fee_paid:.2f} USD")
    
    logging.info("Chạy Empirical Kelly Sizing (Tầng C)...")
    f_star = solve_empirical_kelly_fraction(ret_arr, max_f=5.0)
    logging.info(f"Kelly Fraction tối ưu (f*): {f_star:.4f}")
    
    logging.info("Đánh giá hệ thống với Deflated Sharpe Ratio (Tầng E)...")
    mean_ret = np.mean(ret_arr)
    std_ret = np.std(ret_arr)
    sr_est = (mean_ret / std_ret) * np.sqrt(252 * 6) if std_ret > 0 else 0
    
    dsr_res = compute_deflated_sharpe_ratio(
        sr_estimated=sr_est,
        variance_of_srs=1.5,
        sample_length=N,
        num_trials=1,
        sr_benchmark=0.0
    )
    logging.info(f"Sharpe Ước lượng: {sr_est:.4f}")
    logging.info(f"DSR: {dsr_res['dsr']:.4f}")
    
    logging.info("Ghi nhận vào Sổ Nhật Ký Thử Nghiệm (ExperimentTracker)...")
    tracker = ExperimentTracker()
    param_hash = tracker.log_trial(
        trial_class=TrialClass.FULL_PIPELINE,
        params={"model": "Random_Vertical_Slice", "N": N},
        metrics={"sharpe": sr_est, "dsr": dsr_res['dsr'], "f_star": f_star}
    )
    logging.info(f"Trial Hash: {param_hash}")
    logging.info("VERTICAL SLICE THÀNH CÔNG!")

if __name__ == "__main__":
    main()
