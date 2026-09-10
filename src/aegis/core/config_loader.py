"""
Tiện ích tải YAML configuration an toàn và quản lý cấu hình tập trung.
Đảm bảo tính nhất quán giữa Canonical Parameters Registry và Strategy Configs.
"""
import os
from typing import Optional, Dict, Any
import yaml


def get_config_dir() -> str:
    """Tìm thư mục config/ chuẩn của dự án."""
    # Thử tìm theo cấu trúc src/aegis/core/ -> ../../../config
    base_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(base_dir, "..", "..", "..", "config"),
        os.path.join(os.getcwd(), "config"),
        os.path.join(os.getcwd(), "aegis-trading-system", "config"),
    ]
    for c in candidates:
        if os.path.exists(os.path.join(c, "aegis_canonical_parameters.yaml")):
            return os.path.abspath(c)
    raise FileNotFoundError("Không tìm thấy thư mục config/ chứa aegis_canonical_parameters.yaml!")


def load_canonical_parameters(config_dir: Optional[str] = None) -> Dict[str, Any]:
    """Tải Sổ Cân Bằng Hằng Số (aegis_canonical_parameters.yaml)."""
    cfg_dir = config_dir or get_config_dir()
    path = os.path.join(cfg_dir, "aegis_canonical_parameters.yaml")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Không tìm thấy file: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def deep_merge(dict1: Dict[str, Any], dict2: Dict[str, Any]) -> Dict[str, Any]:
    """Hợp nhất đè 2 dictionary lồng nhau (deep merge)."""
    result = dict1.copy()
    for k, v in dict2.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def load_unified_config(
    strategy_name: str = "trend_following_v1",
    env: str = "paper",
    config_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Tải cấu hình hợp nhất 3 tầng:
    Tầng 1: base.yaml
    Tầng 2: aegis_canonical_parameters.yaml (Hằng số bất biến & Defaults)
    Tầng 3: environments/{env}.yaml
    Tầng 4: strategies/{strategy_name}.yaml
    """
    cfg_dir = config_dir or get_config_dir()
    unified: Dict[str, Any] = {}

    # 1. Base
    base_path = os.path.join(cfg_dir, "base.yaml")
    if os.path.exists(base_path):
        with open(base_path, "r", encoding="utf-8") as f:
            unified = deep_merge(unified, yaml.safe_load(f) or {})

    # 2. Canonical
    canonical = load_canonical_parameters(cfg_dir)
    unified = deep_merge(unified, canonical)

    # 3. Environment
    env_path = os.path.join(cfg_dir, "environments", f"{env}.yaml")
    if os.path.exists(env_path):
        with open(env_path, "r", encoding="utf-8") as f:
            unified = deep_merge(unified, yaml.safe_load(f) or {})

    # 4. Strategy
    strat_path = os.path.join(cfg_dir, "strategies", f"{strategy_name}.yaml")
    if os.path.exists(strat_path):
        with open(strat_path, "r", encoding="utf-8") as f:
            unified = deep_merge(unified, yaml.safe_load(f) or {})

    return unified


# Canonical alias for pipeline usage.
# [FIX F9] Định nghĩa thật nằm ở cuối file (sau flatten_runtime_config): các pipeline
# đọc khóa PHẲNG, nên alias này phải trả về cấu hình ĐÃ LÀM PHẲNG, không phải cây lồng nhau.


# ============================================================================
# [FIX F9] RUNTIME FLATTENING — NỐI LẠI CANONICAL REGISTRY VỚI RUNTIME
# ============================================================================
# Các pipeline (live/research/cpcv) đọc cấu hình bằng KHÓA PHẲNG:
#     config.get("c_trade"), config.get("m_sl"), config.get("t_max_live"), ...
# Trong khi load_unified_config() trả về cây LỒNG NHAU:
#     {"execution_costs": {"taker_fee_rate": {"value": 0.0004, ...}}, ...}
# Hệ quả: MỌI lookup phẳng đều trượt về hằng số mặc định hardcode trong code,
# và "Single Source of Truth" không điều khiển bất cứ thứ gì.
# Hàm dưới đây chiếu registry lồng nhau xuống đúng bộ khóa phẳng mà runtime đọc.

# Các section theo dạng {name: {value: X, unit:..., origin:..., rationale:...}}
_REGISTRY_SECTIONS = (
    "strategic_parameters",
    "architectural_constants",
    "microstructure_guards",
    "execution_costs",
)

# Bí danh: khóa runtime  ->  khóa canonical đã làm phẳng
_RUNTIME_ALIASES = {
    "c_trade": "taker_fee_rate",
    "m_sl": "m_sl_follow",
    "t_max_live": "t_max_live_follow",
}


def _unwrap_registry_entry(entry: Any) -> Any:
    """Lấy `value` khỏi entry registry {value, unit, origin, rationale}."""
    if isinstance(entry, dict) and "value" in entry:
        return entry["value"]
    return entry


def flatten_runtime_config(nested: Dict[str, Any]) -> Dict[str, Any]:
    """
    Chiếu cây cấu hình lồng nhau xuống bộ khóa phẳng mà các pipeline thực sự đọc.

    Thứ tự ưu tiên (sau thắng trước):
      1. Các section registry (`strategic_parameters`, `architectural_constants`,
         `microstructure_guards`, `execution_costs`) — đã bóc `.value`.
      2. `strategy_selection` (config chiến lược ghi đè registry).
      3. Bí danh runtime (`c_trade`, `m_sl`, `t_max_live`).
      4. Khóa phẳng người dùng đặt sẵn ở cấp gốc — luôn thắng tuyệt đối.

    Trả về dict MỚI gồm cả nhánh lồng nhau (giữ tương thích ngược) lẫn khóa phẳng.
    """
    flat: Dict[str, Any] = {}

    for section in _REGISTRY_SECTIONS:
        block = nested.get(section)
        if isinstance(block, dict):
            for key, entry in block.items():
                flat[key] = _unwrap_registry_entry(entry)

    strategy_block = nested.get("strategy_selection")
    if isinstance(strategy_block, dict):
        for key, entry in strategy_block.items():
            flat[key] = _unwrap_registry_entry(entry)

    for runtime_key, canonical_key in _RUNTIME_ALIASES.items():
        if canonical_key in flat:
            flat[runtime_key] = flat[canonical_key]

    # Khóa phẳng đặt tường minh ở cấp gốc có quyền cao nhất — không được ghi đè.
    out = dict(nested)
    for key, value in flat.items():
        if key not in out:
            out[key] = value
    return out


def load_runtime_config(
    strategy_name: str = "trend_following_v1",
    env: str = "paper",
    config_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Tải cấu hình hợp nhất VÀ làm phẳng sang bộ khóa runtime (dùng cho pipeline)."""
    return flatten_runtime_config(
        load_unified_config(strategy_name=strategy_name, env=env, config_dir=config_dir)
    )


# [FIX F9] Alias runtime — trả về cấu hình đã làm phẳng để các lookup phẳng trong
# live/research/cpcv pipeline thực sự đọc được Canonical Registry thay vì rơi hết
# về hằng số hardcode. Nhánh lồng nhau vẫn được giữ nguyên (tương thích ngược).
load_canonical_config = load_runtime_config
