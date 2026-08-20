import pytest
import numpy as np
from aegis.core.experiment_tracker import ExperimentTracker

@pytest.fixture(autouse=True)
def reset_experiment_tracker():
    yield
    ExperimentTracker.reset_instance()


@pytest.fixture
def dummy_ticks():
    np.random.seed(42)
    t = np.arange(1000)
    p = 100.0 + np.cumsum(np.random.normal(0, 0.1, 1000))
    v = np.random.uniform(1.0, 10.0, 1000)
    return np.column_stack([t, p, v])

@pytest.fixture(autouse=True)
def reset_tracker():
    """G3-M2: Reset Singleton ExperimentTracker before every test."""
    from aegis.core.experiment_tracker import ExperimentTracker
    ExperimentTracker.reset_instance()
    yield
    ExperimentTracker.reset_instance()

@pytest.fixture(autouse=True)
def fix_random_seed():
    """G3-M3: Force deterministic seed before every test."""
    import numpy as np
    import random
    np.random.seed(42)
    random.seed(42)
    yield
