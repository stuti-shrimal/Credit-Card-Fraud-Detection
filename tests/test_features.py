import numpy as np
import pandas as pd
import pytest

from features.engineering import ENGINEERED_COLS, create_features


@pytest.fixture
def base_df():
    """Minimal DataFrame with all columns required by create_features."""
    rng = np.random.default_rng(0)
    n = 100
    return pd.DataFrame(
        {
            "Time": np.linspace(0, 172_800, n),
            "Amount": rng.uniform(0.5, 1_000.0, n),
            "Class": np.array([0] * 95 + [1] * 5, dtype=np.int64),
            **{f"V{i}": rng.standard_normal(n) for i in range(1, 29)},
        }
    )


def test_feature_count(base_df):
    df_out = create_features(base_df)
    added = [c for c in df_out.columns if c not in base_df.columns]
    assert len(added) == len(ENGINEERED_COLS)
    assert set(added) == set(ENGINEERED_COLS)


def test_no_nan_in_numeric_output(base_df):
    df_out = create_features(base_df)
    numeric = df_out.select_dtypes(include="number")
    assert numeric.isnull().sum().sum() == 0


def test_feature_ranges(base_df):
    df_out = create_features(base_df)

    # log1p(Amount) is non-negative for Amount >= 0
    assert (df_out["log_amount"] >= 0).all()

    # binary flag
    assert df_out["is_round_amount"].isin([0, 1]).all()

    # (Time % 86400) / 3600 is always in [0, 24)
    assert (df_out["hour_of_day"] >= 0).all()
    assert (df_out["hour_of_day"] < 24).all()

    # count of extreme V components is always non-negative
    assert (df_out["n_extreme_v"] >= 0).all()
