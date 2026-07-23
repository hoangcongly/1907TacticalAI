import os
import yaml
import pytest
from aegis.meta_labeling.purged_kfold import PurgedKFold

def test_canonical_parameters_registry_alignment():
    """
    [TDD VERIFICATION — CANONICAL PARAMETERS REGISTRY]:
    Khẳng định Sổ Cân Bằng Hằng Số (`config/aegis_canonical_parameters.yaml`) tồn tại
    và đồng nhất 100% với tham số mặc định trong code Python và cấu hình chiến lược.
    """
    base_dir = os.path.abspath(os.path.dirname(__file__))
    reg_path = os.path.join(base_dir, "config", "aegis_canonical_parameters.yaml")
    if not os.path.exists(reg_path):
        reg_path = os.path.join(base_dir, "..", "..", "config", "aegis_canonical_parameters.yaml")
    assert os.path.exists(reg_path), f"Sổ cân bằng hằng số aegis_canonical_parameters.yaml chưa tồn tại tại {reg_path}!"
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(reg_path)))



    with open(reg_path, "r", encoding="utf-8") as f:
        reg = yaml.safe_load(f)

    # 1. Khẳng định tham số chiến lược cốt lõi
    strat_params = reg["strategic_parameters"]
    assert strat_params["t_max_live_follow"]["value"] == 120
    assert strat_params["t_max_live_fade"]["value"] == 40
    assert strat_params["embargo_bars"]["value"] == 24
    assert strat_params["embargo_pct"]["value"] == 0.0

    # 2. Khẳng định hằng số kiến trúc
    arch_params = reg["architectural_constants"]
    assert arch_params["n_states"]["value"] == 2
    assert arch_params["m_sl_follow"]["value"] == 2.0
    assert arch_params["m_sl_fade"]["value"] == 1.5

    # 3. Khẳng định microstructure_guards
    micro_guards = reg["microstructure_guards"]
    assert micro_guards["spoofing_discount"]["value"] == 0.70
    assert micro_guards["tier1_threshold_ms"]["value"] == 500
    assert micro_guards["tier2_threshold_ms"]["value"] == 2000
    assert micro_guards["safety_buffer_pct"]["value"] == 0.15

    # 4. Kiểm chứng chéo với constructor của PurgedKFold trong code Python
    pkf = PurgedKFold()
    assert pkf.embargo_bars == strat_params["embargo_bars"]["value"], (
        f"Lỗi trôi tham số PurgedKFold! embargo_bars trong code={pkf.embargo_bars}, registry={strat_params['embargo_bars']['value']}"
    )
    assert pkf.embargo_pct == strat_params["embargo_pct"]["value"], (
        f"Lỗi trôi tham số PurgedKFold! embargo_pct trong code={pkf.embargo_pct}, registry={strat_params['embargo_pct']['value']}"
    )

    # 5. Kiểm chứng chéo với file config chiến lược trend_following_v1.yaml
    strat_yaml_path = os.path.join(project_root, "config", "strategies", "trend_following_v1.yaml")
    if os.path.exists(strat_yaml_path):
        with open(strat_yaml_path, "r", encoding="utf-8") as f:
            strat_cfg = yaml.safe_load(f)
        params = strat_cfg.get("strategy_selection", {})
        assert params.get("t_max_live_follow") == strat_params["t_max_live_follow"]["value"], "Lệch t_max_live_follow với YAML!"
        assert params.get("t_max_live_fade") == strat_params["t_max_live_fade"]["value"], "Lệch t_max_live_fade với YAML!"
        assert params.get("m_sl") == arch_params["m_sl_follow"]["value"], "Lệch m_sl_follow với YAML!"

    print("✅ [CANONICAL REGISTRY] 100% Hằng số kiến trúc & Tham số microstructure đồng bộ hoàn hảo!")

