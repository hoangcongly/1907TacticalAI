"""Kho dump data.binance.vision: liệt kê, lọc cặp và đọc file zip."""

import io
import zipfile

import pytest

from aegis.data.ingestion import binance_vision as vision


def _zip(text: str) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("data.csv", text)
    return buffer.getvalue()


ROW = "1577836800000,10,12,9,11,100,1577840399999,1100,7,60,660,0"


@pytest.mark.parametrize(
    "symbol, expected",
    [
        ("BTCUSDT", True),
        ("1000PEPEUSDT", True),
        ("BTCUSDT_240329", False),
        ("BTCUSDC", False),
        ("ETHBTC", False),
    ],
)
def test_only_usdt_perpetuals_are_kept(symbol, expected):
    assert vision.is_perp_usdt(symbol) is expected


def test_klines_are_read_with_or_without_a_header():
    header = "open_time,open,high,low,close,volume,close_time,quote_volume,count,taker_buy_volume,taker_buy_quote_volume,ignore\n"
    for text in (ROW, header + ROW):
        frame = vision.klines_from_zip(_zip(text))
        assert frame.timestamp_ms.tolist() == [1577836800000]
        assert frame.close.tolist() == [11.0] and frame.tick_count.tolist() == [7]
        assert frame.ofi.tolist() == [pytest.approx(0.2)]  # (60 - 40) / 100


def test_funding_uses_the_rate_column():
    frame = vision.funding_from_zip(
        _zip(
            "calc_time,funding_interval_hours,last_funding_rate\n1577836800003,8,0.0001\n"
        )
    )
    assert frame.funding_time.tolist() == [1577836800003] and frame.rate.tolist() == [
        0.0001
    ]


def _listing(prefixes=(), keys=(), marker=None) -> bytes:
    ns = "http://s3.amazonaws.com/doc/2006-03-01/"
    body = "".join(
        f"<CommonPrefixes><Prefix>{p}</Prefix></CommonPrefixes>" for p in prefixes
    )
    body += "".join(f"<Contents><Key>{k}</Key></Contents>" for k in keys)
    tail = (
        f"<IsTruncated>true</IsTruncated><NextMarker>{marker}</NextMarker>"
        if marker
        else "<IsTruncated>false</IsTruncated>"
    )
    return f'<ListBucketResult xmlns="{ns}">{tail}{body}</ListBucketResult>'.encode()


def test_symbol_listing_follows_pagination():
    root = "data/futures/um/monthly/klines/"
    pages = iter(
        [
            _listing([root + "AAAUSDT/", root + "BBBUSDC/"], marker=root + "BBBUSDC/"),
            _listing([root + "CCCUSDT/"]),
        ]
    )
    seen = []

    def get(url):
        seen.append(url)
        return next(pages)

    assert vision.list_symbols(get=get) == ["AAAUSDT", "CCCUSDT"]
    assert "marker=data%2Ffutures%2Fum%2Fmonthly%2Fklines%2FBBBUSDC%2F" in seen[1]


def test_daily_files_start_after_the_last_monthly_file():
    monthly = "data/futures/um/monthly/klines/XUSDT/1h/"
    daily = "data/futures/um/daily/klines/XUSDT/1h/"
    urls = []

    def get(url):
        urls.append(url)
        if "monthly" in url:
            return _listing(
                keys=[
                    monthly + "XUSDT-1h-2024-01.zip",
                    monthly + "XUSDT-1h-2024-01.zip.CHECKSUM",
                ]
            )
        return _listing(keys=[daily + "XUSDT-1h-2024-02-01.zip"])

    keys = vision.kline_keys("XUSDT", get=get)
    assert keys == [monthly + "XUSDT-1h-2024-01.zip", daily + "XUSDT-1h-2024-02-01.zip"]
    assert "XUSDT-1h-2024-01-99" in urls[1]


def test_download_merges_and_sorts_files():
    monthly = "data/futures/um/monthly/klines/XUSDT/1h/"
    later = ROW.replace("1577836800000", "1577840400000")

    def get(url):
        if "s3-ap-northeast" in url:
            if "monthly%2Fklines" in url:
                return _listing(
                    keys=[
                        monthly + "XUSDT-1h-2020-02.zip",
                        monthly + "XUSDT-1h-2020-01.zip",
                    ]
                )
            return _listing()
        return _zip(later if url.endswith("2020-02.zip") else ROW)

    bars, funding = vision.download_symbol("XUSDT", get=get)
    assert (
        bars.timestamp_ms.tolist() == [1577836800000, 1577840400000] and funding.empty
    )
