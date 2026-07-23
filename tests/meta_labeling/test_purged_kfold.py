"""Unit tests for PurgedKFold (Task B-1-14)."""

import numpy as np
import pandas as pd
import pytest
from aegis.meta_labeling.purged_kfold import (
    PurgedKFold,
    test_purged_kfold_no_overlap_simple,
    test_purged_kfold_toy_overlap_mathematical_proof,
    test_purged_kfold_override_priority,
    test_purged_kfold_input_validation_guards,
)


def test_purged_kfold_all():
    """Chạy toàn bộ unit test cho PurgedKFold."""
    test_purged_kfold_no_overlap_simple()
    test_purged_kfold_toy_overlap_mathematical_proof()
    test_purged_kfold_override_priority()
    test_purged_kfold_input_validation_guards()
