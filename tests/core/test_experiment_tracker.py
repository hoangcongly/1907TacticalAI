import os
import tempfile
import json
import pytest

from aegis.core.trial_classes import TrialClass
from aegis.core.experiment_tracker import ExperimentTracker

def test_singleton_identity():
    tracker1 = ExperimentTracker(log_dir="test_logs")
    tracker2 = ExperimentTracker(log_dir="test_logs")
    assert id(tracker1) == id(tracker2)

def test_hash_consistency():
    tracker = ExperimentTracker(log_dir="test_logs")
    
    params1 = {"learning_rate": 0.01, "max_depth": 5, "features": ["a", "b", "c"]}
    params2 = {"features": ["a", "b", "c"], "max_depth": 5, "learning_rate": 0.01}
    
    hash1 = tracker.hash_params(params1)
    hash2 = tracker.hash_params(params2)
    
    # Hash of identical dicts with different insertion order must match
    assert hash1 == hash2

def test_log_trial_jsonl_io():
    with tempfile.TemporaryDirectory() as tmpdir:
        # Reset singleton instance for testing purposes
        ExperimentTracker._instance = None
        
        tracker = ExperimentTracker(log_dir=tmpdir)
        
        params = {"epochs": 100}
        metrics = {"loss": 0.5, "accuracy": 0.9}
        
        # Log first trial
        hash_result = tracker.log_trial(TrialClass.MODEL_FITTING, params, metrics)
        
        # Log second trial
        tracker.log_trial(TrialClass.STRATEGY_SELECTION, {"epochs": 200}, {"loss": 0.4})
        
        # Verify file exists
        assert os.path.exists(tracker.log_file)
        
        # Read JSONL
        with open(tracker.log_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
            
        assert len(lines) == 2
        
        # Parse first line
        record1 = json.loads(lines[0])
        assert record1["trial_class"] == "model_fitting"
        assert record1["param_hash"] == hash_result
        assert "git_commit" in record1
        assert "env_versions" in record1
        assert isinstance(record1["env_versions"], dict)
        assert record1["params"] == params
        assert record1["metrics"] == metrics
