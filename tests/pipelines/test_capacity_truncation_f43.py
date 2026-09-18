"""
[FIX F43] Cắt theo SỨC CHỨA VỐN không được phá trung lập.

Bối cảnh: khi vốn không đủ cho đủ `n_positions` vị thế (mỗi vị thế phải vượt
min_notional $5), `xs_live_pipeline` cắt bớt danh mục. Bản cũ lấy top-N theo
|trọng số| BẤT KỂ DẤU — chú thích ghi "giữ mạnh nhất ở cả hai chiều" nhưng code
không hề làm thế.

Vì sao chưa từng cắn: testnet có $6.226 nên cap luôn >> 12. Nó cắn đúng lúc chạm
tiền thật. Vốn $38 ở 2x cho cap = 12, vừa khít; sụt 8% xuống $35 là cap = 11.

Mô phỏng 20.000 lượt với độ mạnh tín hiệu ngẫu nhiên (trần trung lập ±2%):
    $35 (cap 11): 100% lượt vi phạm, net 9,1%
    $28 (cap  9): 100% lượt vi phạm, xấu nhất 33,3%
    $20 (cap  6):  57% lượt vi phạm, xấu nhất 100%  <- danh mục MỘT CHIỀU ở 2x

DD kỳ vọng 24,7%/năm nên $38 chạm $35 gần như chắc chắn xảy ra. Đây không phải
biên hiếm; đây là đường đi mặc định của tài khoản thật.
"""
import pytest

from aegis.execution.portfolio_rebalancer import max_positions_for_capital


def _cut_balanced(weights: dict, cap: int):
    """Bản sao logic cắt trong `xs_live_pipeline` [FIX F43] để test ở dạng thuần."""
    if len(weights) <= cap:
        return weights, True
    longs = sorted(((s, w) for s, w in weights.items() if w > 0), key=lambda kv: -kv[1])
    shorts = sorted(((s, w) for s, w in weights.items() if w < 0), key=lambda kv: kv[1])
    per_side = min(cap // 2, len(longs), len(shorts))
    if per_side <= 0:
        return {}, False          # phải HALT, không được giao dịch một chiều
    kl, ks = longs[:per_side], shorts[:per_side]
    sum_l = sum(w for _, w in kl)
    sum_s = -sum(w for _, w in ks)
    if sum_l <= 0 or sum_s <= 0:
        return {}, False
    half = (sum_l + sum_s) / 2.0
    out = {s: w * (half / sum_l) for s, w in kl}
    out.update({s: w * (half / sum_s) for s, w in ks})
    return out, True


def _net_ratio(w: dict) -> float:
    tot = sum(abs(v) for v in w.values())
    return sum(w.values()) / tot if tot else 0.0


@pytest.mark.parametrize("equity", [38, 37, 36, 35, 34, 32, 30, 28, 25, 22, 20, 18, 15, 12])
def test_moi_muc_von_deu_giu_trung_lap(equity):
    """Quét toàn dải vốn thật — không mức nào được phép sinh sổ lệch."""
    # Độ mạnh KHÁC NHAU giữa các cặp (để phép cắt theo |trọng số| có ý nghĩa),
    # nhưng mỗi chân chuẩn hoá về đúng 0,5 -> đầu vào trung lập TUYỆT ĐỐI.
    lm = [1 + 0.10 * i for i in range(6)]
    sm = [1 + 0.07 * i for i in range(6)]
    weights = {f"L{i}": 0.5 * lm[i] / sum(lm) for i in range(6)}
    weights.update({f"S{i}": -0.5 * sm[i] / sum(sm) for i in range(6)})
    assert abs(_net_ratio(weights)) < 1e-12, "đầu vào phải trung lập tuyệt đối"

    cap = max_positions_for_capital(equity, 2.0, min_notional=5.0, safety=1.2)
    cut, ok = _cut_balanced(weights, cap)
    if not ok:
        assert cut == {}, "không đủ vốn thì phải HALT, không được trả về danh mục què"
        return
    r = _net_ratio(cut)
    # Siết hơn trần vận hành ±2%: phép cắt là thao tác TOÁN HỌC thuần, không có
    # nguồn sai số nào ngoài dấu phẩy động, nên nó phải ra 0 tuyệt đối. Để ngưỡng
    # lỏng ở đây là tự cho phép một chỗ rò mới núp dưới trần.
    assert abs(r) <= 1e-12, (
        f"vốn ${equity} (cap {cap}) -> {len(cut)} vị thế, net {r*100:+.4f}% gross "
        f"— phép cắt phải trung lập tuyệt đối"
    )
    nL = sum(1 for v in cut.values() if v > 0)
    nS = sum(1 for v in cut.values() if v < 0)
    assert nL == nS, f"vốn ${equity}: {nL} long vs {nS} short — hai chân phải bằng nhau"


def test_cat_le_khong_de_lai_chan_thua():
    """cap lẻ (11) phải bỏ 1 suất để giữ cân, chứ không giữ 6 long / 5 short."""
    weights = {f"L{i}": 0.5 / 6 for i in range(6)}
    weights.update({f"S{i}": -0.5 / 6 for i in range(6)})
    cut, ok = _cut_balanced(weights, 11)
    assert ok and len(cut) == 10, f"cap lẻ 11 phải cho 10 vị thế cân, nhận {len(cut)}"
    assert abs(_net_ratio(cut)) <= 1e-12


def test_von_qua_nho_thi_halt_chu_khong_cuoc_mot_chieu():
    """cap < 2 nghĩa là không đủ 1 cặp mỗi chân — phải dừng, không được giao dịch."""
    weights = {"L0": 0.5, "S0": -0.5}
    cut, ok = _cut_balanced(weights, 1)
    assert not ok and cut == {}, (
        "vốn không đủ 1 cặp mỗi chân mà vẫn trả danh mục = biến quỹ trung lập thành "
        "cược hướng có đòn bẩy"
    )


def test_giu_dung_tin_hieu_manh_nhat_moi_chan():
    """Cắt phải giữ tín hiệu MẠNH NHẤT mỗi chân, không cắt bừa."""
    weights = {"L_manh": 0.30, "L_yeu": 0.05, "L_vua": 0.15,
               "S_manh": -0.30, "S_yeu": -0.05, "S_vua": -0.15}
    cut, ok = _cut_balanced(weights, 4)
    assert ok and set(cut) == {"L_manh", "L_vua", "S_manh", "S_vua"}, (
        f"phải giữ 2 mạnh nhất mỗi chân, nhận {sorted(cut)}"
    )


def test_do_lon_lech_cuc_doan_van_trung_lap():
    """Hai chân bằng SỐ CẶP nhưng lệch hẳn về ĐỘ LỚN — vẫn phải ra net = 0."""
    weights = {"L0": 0.40, "L1": 0.08, "L2": 0.02,
               "S0": -0.05, "S1": -0.20, "S2": -0.25}
    cut, ok = _cut_balanced(weights, 4)
    assert ok
    assert abs(_net_ratio(cut)) <= 1e-12, (
        f"net {_net_ratio(cut)*100:+.4f}% — cân số cặp mà không cân notional thì "
        f"sổ vẫn lệch"
    )
    assert all(weights[k] * v > 0 for k, v in cut.items()), (
        "phép cân không được ĐẢO DẤU bất kỳ cặp nào — long phải còn là long"
    )
