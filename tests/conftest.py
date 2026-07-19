import pytest
import numpy as np


@pytest.fixture
def dummy_ticks():
    np.random.seed(42)
    t = np.arange(1000)
    p = 100.0 + np.cumsum(np.random.normal(0, 0.1, 1000))
    v = np.random.uniform(1.0, 10.0, 1000)
    return np.column_stack([t, p, v])
