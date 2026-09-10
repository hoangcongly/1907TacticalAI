"""Unit tests for config_loader.py."""
import os
import pytest
from aegis.core.config_loader import (
    load_canonical_parameters,
    load_unified_config,
    get_config_dir,
)


def test_get_config_dir():
    cfg_dir = get_config_dir()
    assert os.path.isdir(cfg_dir)
    assert os.path.exists(os.path.join(cfg_dir, "aegis_canonical_parameters.yaml"))


def test_load_canonical_parameters():
    params = load_canonical_parameters()
    assert "strategic_parameters" in params
    assert "architectural_constants" in params
    assert params["strategic_parameters"]["embargo_bars"]["value"] == 24
    assert params["architectural_constants"]["n_states"]["value"] == 2


def test_load_unified_config_merging():
    cfg = load_unified_config(strategy_name="trend_following_v1", env="paper")
    assert "system" in cfg
    assert "strategic_parameters" in cfg
    assert "strategy" in cfg or "strategy_selection" in cfg
    
    # Invariant checks from Constitution
    strat = cfg.get("strategy_selection", {}) or cfg.get("strategy", {})
    assert strat.get("t_max_live_follow") == 120
    assert strat.get("t_max_live_fade") == 40
