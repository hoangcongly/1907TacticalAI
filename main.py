#!/usr/bin/env python3
"""
main.py — Institutional CLI Dispatcher & System Entrypoint.

Hỗ trợ 4 chế độ vận hành:
- research: Huấn luyện full-fit toàn trình và xuất artifacts.
- cpcv: Đánh giá Cross-Validation Purged K-Fold, DSR và ma trận Kelly 2D.
- live: Streaming giao dịch trực tiếp với kết nối API và sàn.
- paper: Mô phỏng streaming giao dịch bằng vốn ảo và dữ liệu thời gian thực.
"""

import os
import sys
import signal
import argparse
from typing import Optional, List, Dict, Any
import numpy as np
import pandas as pd
import yaml

from aegis.core.config_loader import load_canonical_config, deep_merge
from aegis.core.schemas import SignalBarSchema
from aegis.pipelines.research_pipeline import ResearchPipeline
from aegis.pipelines.cpcv_pipeline import CPCVPipeline
from aegis.pipelines.live_pipeline import AegisLivePipeline


def generate_synthetic_signal_bars(n_bars: int = 120, symbol: str = "BTCUSDT") -> pd.DataFrame:
    """Tạo dữ liệu nến giả lập chuẩn 100% SignalBarSchema (19 cột)."""
    np.random.seed(42)
    timestamps = 1700000000000 + np.arange(n_bars) * 60000
    prices = 100.0 + np.cumsum(np.random.normal(0.05, 0.5, n_bars))
    highs = prices + np.abs(np.random.normal(0.5, 0.2, n_bars))
    lows = prices - np.abs(np.random.normal(0.5, 0.2, n_bars))
    volumes = np.random.uniform(10.0, 100.0, n_bars)

    df = pd.DataFrame({
        "bar_idx": np.arange(n_bars, dtype=np.int64),
        "symbol": symbol,
        "timestamp_ms": timestamps,
        "open": prices,
        "high": highs,
        "low": lows,
        "close": prices,
        "volume": volumes,
        "ofi": np.clip(np.random.normal(0.0, 0.3, n_bars), -1.0, 1.0),
        "tick_count": np.random.randint(10, 50, n_bars, dtype=np.int64),
        "is_toxic_flag": np.zeros(n_bars, dtype=bool),
        "is_tail_event": np.zeros(n_bars, dtype=bool),
        "insufficient_history": np.array([True] * 20 + [False] * (n_bars - 20), dtype=bool),
        "trend_score": [np.nan] * 20 + list(np.random.normal(0.2, 0.5, n_bars - 20)),
        "p_trend": [np.nan] * 20 + list(np.random.uniform(0.4, 0.8, n_bars - 20)),
        "p_chop": [np.nan] * 20 + list(np.random.uniform(0.2, 0.6, n_bars - 20)),
        "atr_14": [np.nan] * 20 + list(np.random.uniform(0.5, 1.5, n_bars - 20)),
        "hurst_value": [np.nan] * 20 + list(np.random.uniform(0.4, 0.6, n_bars - 20)),
        "d_star_used": [0.35] * n_bars,
    })

    SignalBarSchema.validate(df)
    return df


def load_data_or_synthetic(
    data_path: Optional[str],
    n_bars: int = 120,
    symbol: str = "BTCUSDT",
    allow_synthetic: bool = True,
) -> pd.DataFrame:
    """
    Tải dữ liệu nến từ đường dẫn, hoặc sinh dữ liệu mẫu.

    [FIX F2] `allow_synthetic=False` CHẶN CỨNG dữ liệu giả. Chế độ live/paper bắt
    buộc dùng cờ này: bản cũ lặng lẽ rơi về `np.random.seed(42)` random walk, và
    tệ hơn — nếu thiếu artifacts thì tự huấn luyện model production trên chính
    nhiễu đó rồi đem đi giao dịch.
    """
    if data_path and os.path.exists(data_path):
        if data_path.endswith(".parquet"):
            df = pd.read_parquet(data_path)
        else:
            df = pd.read_csv(data_path)
        return df

    if not allow_synthetic:
        raise SystemExit(
            "\n[TỪ CHỐI CHẠY] Chế độ live/paper không được phép dùng dữ liệu giả.\n"
            f"  Đường dẫn dữ liệu: {data_path or '(không cung cấp)'}\n\n"
            "  Tải dữ liệu THẬT trước:\n"
            f"      python scripts/download_data.py --symbol {symbol} --interval 1h --days 400\n"
            f"      python main.py --mode paper --data data/binance/{symbol}_1h_signal.parquet\n"
        )

    return generate_synthetic_signal_bars(n_bars=n_bars, symbol=symbol)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aegis Trading System — Institutional Core Entrypoint")
    parser.add_argument(
        "--mode",
        choices=["research", "cpcv", "live", "paper"],
        default="paper",
        help="Lựa chọn chế độ vận hành (research, cpcv, live, paper)",
    )
    parser.add_argument("--config", type=str, default=None, help="Đường dẫn file cấu hình YAML ghi đè")
    parser.add_argument("--data", type=str, default=None, help="Đường dẫn file dữ liệu nến (CSV / Parquet)")
    parser.add_argument("--symbol", type=str, default="BTCUSDT", help="Cặp giao dịch (mặc định: BTCUSDT)")
    parser.add_argument("--artifacts-dir", type=str, default="artifacts/", help="Thư mục chứa artifacts")
    parser.add_argument("--bars-count", type=int, default=120, help="Số lượng nến tạo mẫu nếu không có data")
    parser.add_argument("--initial-capital", type=float, default=10000.0, help="Vốn ban đầu USD")

    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    # 1. Tải cấu hình hợp nhất
    config = load_canonical_config()
    if args.config and os.path.exists(args.config):
        with open(args.config, "r", encoding="utf-8") as f:
            override_cfg = yaml.safe_load(f) or {}
            config = deep_merge(config, override_cfg)

    config["fast_mode"] = config.get("fast_mode", True)

    print("=" * 60)
    print(f"AEGIS TRADING SYSTEM — MODE: {args.mode.upper()}")
    print(f"Symbol: {args.symbol} | Artifacts Dir: {args.artifacts_dir}")
    print("=" * 60)

    # 2. Phân luồng thực thi
    if args.mode == "research":
        bars = load_data_or_synthetic(args.data, n_bars=args.bars_count, symbol=args.symbol)
        pipeline = ResearchPipeline(config=config)
        res = pipeline.run(signal_bars=bars, artifacts_dir=args.artifacts_dir)
        print(f"[RESEARCH COMPLETE] Status: {res['status']}")
        print(f"Manifest Hash: {res['manifest_hash']}")
        print(f"Selected Features: {res['selected_features']}")
        print(f"OOS Sharpe: {res['metadata']['oos_sharpe']:.2f}")
        print(f"Artifacts exported to: {res['artifacts_dir']}")
        return 0

    elif args.mode == "cpcv":
        bars = load_data_or_synthetic(args.data, n_bars=args.bars_count, symbol=args.symbol)
        pipeline = CPCVPipeline(config=config)
        res = pipeline.run(bars)
        m = res["metrics"]
        print(f"[CPCV COMPLETE] Trades: {m['n_total_trades']} | Clean: {m['n_clean_trades']}")
        print(f"OOS Sharpe: {m['sharpe_oos']:.2f} | DSR: {m['dsr']:.4f} | Truncation Rate: {m['truncation_rate']:.1%}")
        return 0

    elif args.mode in ["live", "paper"]:
        # Tự động huấn luyện research nếu thư mục artifacts chưa tồn tại
        model_path = os.path.join(args.artifacts_dir, "model.pkl")
        if not os.path.exists(model_path):
            # [FIX F2] TUYỆT ĐỐI KHÔNG tự huấn luyện tại đây. Bản cũ sinh dữ liệu
            # ngẫu nhiên rồi fit model production lên nhiễu, sau đó đem đi giao dịch.
            raise SystemExit(
                f"\n[TỪ CHỐI CHẠY] Không tìm thấy artifacts tại {args.artifacts_dir}\n\n"
                "  Phải huấn luyện trên dữ liệu THẬT trước khi giao dịch:\n"
                f"      python scripts/download_data.py --symbol {args.symbol} --interval 1h --days 400\n"
                f"      python main.py --mode research --data data/binance/{args.symbol}_1h_signal.parquet\n"
            )

        live_pipe = AegisLivePipeline.from_artifacts(
            artifacts_dir=args.artifacts_dir,
            initial_capital=args.initial_capital,
            config=config,
        )

        bars = load_data_or_synthetic(
            args.data, n_bars=args.bars_count, symbol=args.symbol, allow_synthetic=False
        )
        stop_requested = False

        def sig_handler(sig, frame):
            nonlocal stop_requested
            print("\n[STOP] Nhận tín hiệu ngắt (SIGINT/SIGTERM). Đang tắt an toàn...")
            stop_requested = True

        try:
            signal.signal(signal.SIGINT, sig_handler)
            signal.signal(signal.SIGTERM, sig_handler)
        except Exception:
            pass

        print(f"[STREAMING START] Bắt đầu nạp {len(bars)} nến...")
        for _, row in bars.iterrows():
            if stop_requested:
                break
            bar_dict = row.to_dict()
            res = live_pipe.on_bar(bar_dict)
            action = res.get("action", "HOLD")
            if action in ["OPEN", "CLOSE"]:
                print(f"Bar {bar_dict.get('bar_idx')}: [{action}] {res}")

        print("=" * 60)
        print(f"Final Balance: ${live_pipe.account_tracker.wallet_balance:.2f}")
        print(f"Total Closed Trades: {len(live_pipe.closed_trades)}")
        print("=" * 60)
        return 0

    return 0


if __name__ == "__main__":
    sys.exit(main())
