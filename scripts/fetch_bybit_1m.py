"""
DỮ LIỆU THẬT CHO NGHIÊN CỨU KHI KHÔNG CÓ `data/` (container cloud chặn Binance).

    python scripts/fetch_bybit_1m.py ../bybit_data     # ~2,6GB tải, ~190MB sau khi gộp
    python scripts/all_levers_study.py --data-root ../bybit_data/layout

Tải release v1 của meldar1986/bybit-1m-data, kiểm sha256, gộp 1m -> 1h theo ĐÚNG định dạng
`data/binance/{SYM}_1h.parquet` của repo, rồi xoá 1m.

OFI: dữ liệu Bybit không có khối lượng taker mua/bán, nên ước lượng bằng Bulk Volume
Classification (Easley, López de Prado & O'Hara 2012): mỗi nến 1 phút, phần khối lượng mua
= Φ(Δlog p / σ), σ = độ lệch chuẩn Δlog p 1 ngày TRƯỚC đó (nhân quả). ofi_1m = 2Φ − 1, rồi
gộp lên 1h theo trọng số khối lượng — giống cách repo gộp ofi thật.
"""
import hashlib
import io
import json
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import ndtr

BASE = "https://github.com/meldar1986/bybit-1m-data/releases/download/v1/"
MANIFEST = "https://raw.githubusercontent.com/meldar1986/bybit-1m-data/main/manifest.json"
ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else "../bybit_data")
OUT = ROOT / "layout"
OUT.mkdir(parents=True, exist_ok=True)
(ROOT / ".dl").mkdir(exist_ok=True)
if not (ROOT / "manifest.json").exists():
    subprocess.run(["curl", "-fsSL", "-o", str(ROOT / "manifest.json"), MANIFEST], check=True)
man = json.load(open(ROOT / "manifest.json"))


def to_1h(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("ts").drop_duplicates("ts")
    c = df["close"].astype("float64")
    r = np.log(c).diff()
    sig = r.rolling(1440, min_periods=60).std().shift(1)
    ofi = (2.0 * ndtr((r / sig).to_numpy()) - 1.0)
    vol = df["volume"].astype("float64").to_numpy()
    b = (df["ts"].to_numpy() // 3_600_000) * 3_600_000
    d = pd.DataFrame({"b": b, "open": df["open"].astype("float64").to_numpy(),
                      "high": df["high"].astype("float64").to_numpy(),
                      "low": df["low"].astype("float64").to_numpy(), "close": c.to_numpy(),
                      "volume": vol, "ov": np.nan_to_num(ofi) * vol})
    g = d.groupby("b", sort=True)
    out = pd.DataFrame({"timestamp_ms": g.size().index.astype("int64"),
                        "open": g["open"].first().to_numpy(), "high": g["high"].max().to_numpy(),
                        "low": g["low"].min().to_numpy(), "close": g["close"].last().to_numpy(),
                        "volume": g["volume"].sum().to_numpy()})
    den = g["volume"].sum().replace(0.0, np.nan)
    out["ofi"] = np.clip((g["ov"].sum() / den).fillna(0.0).to_numpy(), -1.0, 1.0)
    return out


for name, v in man["assets"].items():
    done = ROOT / ".dl" / (name + ".done")
    if done.exists():
        continue
    z = ROOT / ".dl" / name
    subprocess.run(["curl", "-fsSL", "--retry", "5", "-o", str(z), BASE + name], check=True)
    if hashlib.sha256(z.read_bytes()).hexdigest() != v["sha256"]:
        sys.exit(f"sha256 sai: {name}")
    with zipfile.ZipFile(z) as zf:
        for f in zf.namelist():
            if not f.endswith(".parquet"):
                continue
            sym = Path(f).stem
            df = pd.read_parquet(io.BytesIO(zf.read(f)))
            if "funding" in f:
                df.rename(columns={"ts": "funding_time"}).to_parquet(OUT / f"{sym}_funding.parquet")
            else:
                to_1h(df).to_parquet(OUT / f"{sym}_1h.parquet")
    z.unlink()
    done.touch()
    print("xong", name, flush=True)
print("TẤT CẢ XONG", len(list(OUT.glob("*_1h.parquet"))), "ohlcv,",
      len(list(OUT.glob("*_funding.parquet"))), "funding", flush=True)
