import numpy as np
import pandas as pd
import pytest

from data.quality import check_data_quality


@pytest.fixture
def valid_df():
    """200-row DataFrame that satisfies every quality rule."""
    rng = np.random.default_rng(42)
    n, n_fraud = 200, 20
    return pd.DataFrame(
        {
            "Time": np.linspace(0, 172_800, n),
            "Amount": rng.uniform(1.0, 500.0, n),
            "Class": np.array([1] * n_fraud + [0] * (n - n_fraud), dtype=np.int64),
            **{f"V{i}": rng.standard_normal(n) for i in range(1, 29)},
        }
    )


def test_quality_gate_passes_on_valid_data(valid_df):
    result = check_data_quality(valid_df)
    assert result["success"] is True, f"Unexpected failures: {result['failures']}"


def test_quality_gate_catches_broken_dataset():
    # Two rows is below MIN_ROWS_CRITICAL (100); Amount column is absent.
    broken = pd.DataFrame(
        {
            "Time": [0.0, 1_000.0],
            "Class": np.array([0, 1], dtype=np.int64),
        }
    )
    result = check_data_quality(broken)
    assert result["success"] is False
    assert len(result["failures"]) > 0
