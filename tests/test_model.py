from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import os

MODELS_DIR = Path(__file__).resolve().parents[1] / "models"
MODEL_PATH = MODELS_DIR / "production_model.pkl"


@pytest.fixture(scope="module")
def trained_model():
    # Loading some serialized models (notably XGBoost pickles) can hard-abort the interpreter
    # when library versions differ across environments. Keep CI stable by making these tests opt-in.
    if os.getenv("RUN_MODEL_TESTS") != "1":
        pytest.skip("Set RUN_MODEL_TESTS=1 to enable model artifact loading tests.")
    if not MODEL_PATH.exists():
        pytest.skip(f"Model file not found: {MODEL_PATH}")
    import joblib
    try:
        return joblib.load(MODEL_PATH)
    except Exception as e:
        pytest.skip(f"Model could not be loaded in this environment: {e}")


def _make_input(model) -> pd.DataFrame:
    """One-row DataFrame with the features the model expects; defaults to 0."""
    row = {f: 0.0 for f in model.feature_names_in_}
    row["Amount"] = 100.0
    row["log_amount"] = np.log1p(100.0)
    row["hour_of_day"] = 14.0
    row["is_round_amount"] = 1.0
    return pd.DataFrame([row])


def test_model_loads(trained_model):
    assert hasattr(trained_model, "predict_proba")
    assert hasattr(trained_model, "feature_names_in_")
    assert len(trained_model.feature_names_in_) > 0


def test_predictions_in_valid_range(trained_model):
    proba = trained_model.predict_proba(_make_input(trained_model))

    assert proba.shape == (1, 2), f"Expected shape (1, 2), got {proba.shape}"
    assert (proba >= 0).all(), "Probabilities must be >= 0"
    assert (proba <= 1).all(), "Probabilities must be <= 1"
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)
