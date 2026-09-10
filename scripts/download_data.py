#!/usr/bin/env python3
"""
Tải dữ liệu thị trường THẬT từ Binance mainnet (endpoint công khai, không cần khoá).

    python scripts/download_data.py --symbol BTCUSDT --interval 1m --days 30

Dữ liệu nghiên cứu LUÔN lấy từ mainnet: sổ lệnh testnet là thanh khoản giả
(43% số nến có OFI bão hoà ±1) nên backtest trên đó không có giá trị.
"""
import argparse
import logging
import sys

from aegis.data.ingestion.binance_history import (
    download_funding_rates,
    download_klines,
    save_klines,
)
from aegis.data.ingestion.binance_rest import BinanceFuturesREST


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Tải nến + funding thật từ Binance mainnet")
    parser.add_argument("--symbol", default="BTCUSDT", help="Cặp giao dịch")
    parser.add_argument("--interval", default="1m", help="Khung thời gian (1m, 5m, 1h, ...)")
    parser.add_argument("--days", type=float, default=30.0, help="Số ngày lịch sử")
    parser.add_argument("--root", default="data/binance", help="Thư mục lưu parquet")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    client = BinanceFuturesREST.public_mainnet()
    filters = client.symbol_filters(args.symbol)
    print(f"[{args.symbol}] min_notional=${filters['min_notional']} "
          f"step={filters['step_size']} tick={filters['tick_size']}")

    bars = download_klines(args.symbol, args.interval, days=args.days, client=client)
    path = save_klines(bars, args.symbol, args.interval, root=args.root)

    funding = download_funding_rates(args.symbol, days=args.days, client=client)
    rates = list(funding.values())
    if rates:
        print(f"Funding: {len(rates)} mốc | trung bình {sum(rates)/len(rates)*100:.4f}% mỗi 8h "
              f"| min {min(rates)*100:.4f}% max {max(rates)*100:.4f}%")

    span_days = (bars['timestamp_ms'].iloc[-1] - bars['timestamp_ms'].iloc[0]) / 86_400_000
    print(f"✅ {len(bars)} nến trải {span_days:.1f} ngày -> {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
