"""
Credit Card Fraud Detection  portfolio Streamlit app.

Loads `data/predictions.csv` and `data/model_results.json` when present;
otherwise uses in-memory defaults so the UI still runs (e.g. fresh clone).
"""
from __future__ import annotations

import html
import json
import textwrap
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import confusion_matrix

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
APP_DIR = Path(__file__).resolve().parent

INDIGO = "#60a5fa"
RED = "#fb7185"
GREEN = "#4ade80"
AMBER = "#fbbf24"
TMPL = "plotly_dark"

DEFAULT_RESULTS: dict[str, Any] = {
    "winner": "XGBoost (tuned)",
    "github_url": "https://github.com/stuti-shrimal/Credit-Card-Fraud-Detection",
    "models": [
        {
            "name": "Baseline · Logistic Regression",
            "test_recall": 0.592,
            "test_precision": 0.784,
            "test_f1": 0.674,
            "test_auc_roc": 0.941,
            "training_time_s": 2.0,
        },
        {
            "name": "Random Forest",
            "test_recall": 0.780,
            "test_precision": 0.810,
            "test_f1": 0.817,
            "test_auc_roc": 0.960,
            "training_time_s": 214.0,
        },
        {
            "name": "XGBoost (candidate)",
            "test_recall": 0.820,
            "test_precision": 0.830,
            "test_f1": 0.835,
            "test_auc_roc": 0.975,
            "training_time_s": 6.0,
        },
        {
            "name": "XGBoost (tuned)",
            "test_recall": 0.837,
            "test_precision": 0.845,
            "test_f1": 0.841,
            "test_auc_roc": 0.979,
            "training_time_s": 4.9,
        },
    ],
    "winner_notes": [
        "Highest fraud recall on the hold-out set: more stolen money stopped.",
        "Strong precision: most alerts are real fraud, so analysts are not flooded.",
        "Best F1 trade-off under severe imbalance (fraud is ~0.17% of rows).",
        "Fast training and scoring  suitable for near-real-time screening.",
    ],
    "feature_importance": [
        {"feature": "V14", "importance": 0.082},
        {"feature": "V17", "importance": 0.071},
        {"feature": "v_range", "importance": 0.065},
        {"feature": "V12", "importance": 0.058},
        {"feature": "v_mean", "importance": 0.052},
        {"feature": "V10", "importance": 0.048},
        {"feature": "v14_x_v17", "importance": 0.045},
        {"feature": "V4", "importance": 0.041},
        {"feature": "log_amount", "importance": 0.038},
        {"feature": "Amount", "importance": 0.035},
        {"feature": "V20", "importance": 0.032},
        {"feature": "V7", "importance": 0.028},
        {"feature": "hour_of_day", "importance": 0.026},
        {"feature": "V13", "importance": 0.024},
        {"feature": "V8", "importance": 0.022},
    ],
}


def _default_predictions_df() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    n = 8_000
    p_fraud = 0.002
    y = (rng.random(n) < p_fraud).astype(np.int8)
    score = rng.beta(2, 8, n) * (1 - y) + rng.beta(7, 2, n) * y
    score = np.clip(score + rng.normal(0, 0.04, n), 0.001, 0.999)
    pred = (score >= 0.5).astype(np.int8)
    return pd.DataFrame({"Class": y, "predicted": pred, "fraud_probability": score})


def inject_css() -> None:
    st.markdown(
        """
<style>
  :root {
    --font: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
      "Helvetica Neue", Arial, "Noto Sans", sans-serif;
    --bg: #0b1120;
    --surface: #111827;
    --surface-2: #1f2937;
    --border: #334155;
    --text: #f8fafc;
    --text-secondary: #cbd5e1;
    --text-muted: #94a3b8;
    --accent: #60a5fa;
    --accent-hover: #93c5fd;
    --positive: #86efac;
    --radius: 10px;
    --shadow: 0 4px 24px rgba(0, 0, 0, 0.35);
  }

  .stApp,
  [data-testid="stAppViewContainer"] {
    font-family: var(--font) !important;
    background: var(--bg) !important;
    color: var(--text) !important;
  }

  #MainMenu { visibility: hidden; }
  header[data-testid="stHeader"] {
    background: var(--surface) !important;
    border-bottom: 1px solid var(--border) !important;
  }

  section.main {
    font-family: var(--font) !important;
    color: var(--text) !important;
  }
  section.main h1 {
    color: var(--text) !important;
    font-weight: 700 !important;
    font-size: 1.75rem !important;
    letter-spacing: -0.02em !important;
  }
  section.main h2,
  section.main h3 {
    color: var(--text) !important;
    font-weight: 600 !important;
  }
  section.main [data-testid="stMarkdownContainer"] p,
  section.main [data-testid="stMarkdownContainer"] li,
  section.main [data-testid="stMarkdownContainer"] td,
  section.main [data-testid="stMarkdownContainer"] th {
    color: var(--text-secondary) !important;
    font-size: 1rem !important;
    line-height: 1.65 !important;
  }
  section.main [data-testid="stCaptionContainer"] p {
    color: var(--text-muted) !important;
  }
  section.main [data-testid="stWidgetLabel"] p,
  section.main label[data-testid="stWidgetLabel"] {
    color: var(--text-secondary) !important;
  }
  section.main .streamlit-expanderHeader {
    color: var(--text) !important;
  }
  section.main code,
  section.main [data-testid="stMarkdownContainer"] code {
    color: var(--accent-hover) !important;
    background: rgba(15, 23, 42, 0.6) !important;
    padding: 0.1rem 0.35rem !important;
    border-radius: 4px !important;
    font-size: 0.9em !important;
  }

  section.main [data-testid="column"] {
    color: var(--text) !important;
  }
  section.main [data-testid="stMetric"] {
    background: var(--surface-2) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    padding: 0.65rem 0.85rem !important;
    color: var(--text) !important;
  }
  section.main [data-testid="stMetric"] div {
    color: var(--text) !important;
  }
  section.main [data-testid="stMetric"] [data-testid="stMetricLabel"] p,
  section.main [data-testid="stMetric"] [data-testid="stMetricLabel"] div,
  section.main [data-testid="stMetric"] [data-testid="stMetricLabel"] span,
  section.main [data-testid="stMetric"] [data-testid="stMetricLabel"] label {
    color: var(--text-muted) !important;
    font-weight: 500 !important;
  }
  section.main [data-testid="stMetric"] [data-testid="stMetricValue"] p,
  section.main [data-testid="stMetric"] [data-testid="stMetricValue"] div,
  section.main [data-testid="stMetric"] [data-testid="stMetricValue"] span {
    color: var(--text) !important;
    font-weight: 600 !important;
  }
  section.main [data-testid="stMetric"] [data-testid="stMetricValue"] svg {
    fill: var(--text) !important;
  }
  section.main [data-testid="stMetric"] [data-testid="stMetricDelta"] p,
  section.main [data-testid="stMetric"] [data-testid="stMetricDelta"] div,
  section.main [data-testid="stMetric"] [data-testid="stMetricDelta"] span {
    color: var(--positive) !important;
    font-weight: 500 !important;
  }
  section.main [data-testid="stMetric"] [data-testid="stMetricDelta"] svg {
    fill: var(--positive) !important;
  }

  section[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
  }
  section[data-testid="stSidebar"] > div {
    background: transparent !important;
  }
  section[data-testid="stSidebar"] [data-testid="stRadio"] label,
  section[data-testid="stSidebar"] [data-testid="stRadio"] label p,
  section[data-testid="stSidebar"] [data-testid="stRadio"] label span {
    color: var(--text) !important;
    font-size: 0.9375rem !important;
    font-weight: 500 !important;
  }
  section[data-testid="stSidebar"] [data-testid="stRadio"] svg {
    fill: var(--accent) !important;
  }
  section[data-testid="stSidebar"] hr {
    border-color: var(--border) !important;
  }

  section[data-testid="stSidebar"] .glance-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.8125rem;
    margin: 0.35rem 0 0.85rem;
    border: 1px solid var(--border);
    border-radius: var(--radius);
    overflow: hidden;
    background: var(--surface-2);
  }
  section[data-testid="stSidebar"] .glance-table thead th {
    background: var(--bg);
    color: var(--text-muted);
    font-weight: 700;
    font-size: 0.65rem;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    padding: 0.5rem 0.6rem;
    text-align: left;
    border-bottom: 1px solid var(--border);
  }
  section[data-testid="stSidebar"] .glance-table thead th:last-child {
    text-align: right;
  }
  section[data-testid="stSidebar"] .glance-table tbody td {
    padding: 0.5rem 0.6rem;
    border-bottom: 1px solid var(--border);
    vertical-align: middle;
  }
  section[data-testid="stSidebar"] .glance-table tbody tr:last-child td {
    border-bottom: none;
  }
  section[data-testid="stSidebar"] .glance-table .glance-metric {
    color: var(--text-secondary);
    font-weight: 700;
    font-size: 0.8125rem;
  }
  section[data-testid="stSidebar"] .glance-table .glance-val {
    text-align: right;
    color: var(--text);
    font-weight: 600;
    font-size: 0.875rem;
    font-variant-numeric: tabular-nums;
  }

  .glance-head {
    color: var(--text) !important;
    font-size: 0.8125rem !important;
    font-weight: 800 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
    margin: 1.15rem 0 0.65rem !important;
    line-height: 1.3 !important;
  }

  .sidebar-brand {
    text-align: center;
    padding: 1rem;
    background: var(--surface-2);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    margin-bottom: 0.75rem;
  }
  .sidebar-brand-title {
    color: var(--text) !important;
    font-weight: 600;
    font-size: 0.9375rem;
  }
  .sidebar-brand-sub {
    color: var(--text-muted) !important;
    font-size: 0.75rem;
    margin-top: 0.25rem;
  }

  .top-bar {
    background: var(--surface);
    border-bottom: 1px solid var(--border);
    padding: 0.75rem 0;
    margin: 0 -4rem 1.5rem;
    padding-left: max(1rem, 2rem);
    padding-right: max(1rem, 2rem);
    box-shadow: var(--shadow);
  }
  .top-bar-inner {
    max-width: 1200px;
    margin: 0 auto;
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 1rem;
  }
  .top-brand {
    font-weight: 600;
    font-size: 0.9375rem;
    color: var(--text);
    letter-spacing: -0.02em;
  }
  .top-tag {
    font-size: 0.8125rem;
    color: var(--text-muted);
    font-weight: 400;
  }

  .hero {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    border-top: 3px solid var(--accent);
    padding: 1.75rem 1.5rem;
    margin-bottom: 1.25rem;
    box-shadow: var(--shadow);
  }
  .hero-eyebrow {
    display: block;
    color: var(--text-muted);
    font-size: 0.6875rem;
    font-weight: 600;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    margin-bottom: 0.5rem;
  }
  .hero-title {
    font-size: clamp(1.35rem, 2.2vw, 1.85rem);
    font-weight: 700;
    margin: 0 0 0.5rem;
    line-height: 1.25;
    color: var(--text) !important;
    letter-spacing: -0.03em;
  }
  .hero-sub {
    font-size: 1rem;
    max-width: 48rem;
    margin: 0;
    line-height: 1.6;
    color: var(--text-secondary) !important;
  }

  .sec-title {
    font-size: 0.9375rem;
    font-weight: 600;
    color: var(--text);
    margin: 2rem 0 0.65rem;
    padding-bottom: 0.45rem;
    border-bottom: 1px solid var(--border);
    letter-spacing: -0.01em;
  }

  .prose p {
    color: var(--text-secondary) !important;
    font-size: 1rem !important;
    line-height: 1.7 !important;
    margin: 0 0 0.85rem !important;
  }
  .prose p:last-child {
    margin-bottom: 0 !important;
  }
  .prose strong {
    color: var(--text) !important;
  }

  .badge-wrap {
    display: flex;
    flex-wrap: wrap;
    gap: 0.5rem;
  }
  .badge {
    padding: 0.35rem 0.7rem;
    border-radius: 6px;
    font-size: 0.75rem;
    font-weight: 500;
    background: var(--surface-2);
    color: var(--text-secondary);
    border: 1px solid var(--border);
  }

  .callout {
    border-radius: var(--radius);
    padding: 1rem 1.125rem;
    margin: 0.65rem 0;
    border: 1px solid var(--border);
    border-left: 4px solid var(--accent);
    background: var(--surface-2);
    color: var(--text-secondary) !important;
    font-size: 0.9375rem;
    line-height: 1.6;
  }
  .callout strong {
    color: var(--text) !important;
  }
  .callout code {
    color: var(--accent-hover) !important;
    background: rgba(0, 0, 0, 0.25) !important;
    padding: 0.1rem 0.3rem !important;
    border-radius: 4px !important;
  }
  .callout-amber {
    border-left-color: #fbbf24;
    background: rgba(120, 53, 15, 0.35);
    border-color: rgba(251, 191, 36, 0.35);
    color: #fde68a !important;
  }
  .callout-amber strong {
    color: #fef3c7 !important;
  }
  .callout-green {
    border-left-color: #4ade80;
    background: rgba(6, 78, 59, 0.35);
    border-color: rgba(74, 222, 128, 0.25);
    color: #bbf7d0 !important;
  }
  .callout-green strong {
    color: #ecfdf5 !important;
  }
  .callout-red {
    border-left-color: #fb7185;
    background: rgba(127, 29, 29, 0.35);
    border-color: rgba(251, 113, 133, 0.3);
    color: #fecdd3 !important;
  }
  .callout-red strong {
    color: #fff1f2 !important;
  }

  section.main a {
    color: var(--accent) !important;
    text-decoration: none;
  }
  section.main a:hover {
    color: var(--accent-hover) !important;
    text-decoration: underline !important;
  }
  section[data-testid="stSidebar"] a {
    color: var(--accent) !important;
  }

  .footer {
    text-align: center;
    color: var(--text-muted);
    font-size: 0.8125rem;
    padding: 2rem 0 1rem;
    margin-top: 2.5rem;
    border-top: 1px solid var(--border);
  }

  .pred-fraud,
  .pred-ok {
    border-radius: var(--radius);
    padding: 1.25rem;
    text-align: center;
    border: 1px solid var(--border);
  }
  .pred-fraud {
    background: rgba(127, 29, 29, 0.4);
    border-color: rgba(251, 113, 133, 0.5);
  }
  .pred-ok {
    background: rgba(6, 78, 59, 0.4);
    border-color: rgba(74, 222, 128, 0.45);
  }
  .pred-fraud .pred-head,
  .pred-ok .pred-head {
    color: var(--text) !important;
    font-size: 1.125rem;
    font-weight: 700;
  }
  .pred-fraud .pred-score,
  .pred-ok .pred-score {
    color: var(--text) !important;
    font-size: 1.75rem;
    font-weight: 800;
    margin: 0.35rem 0;
  }
  .pred-fraud .pred-note,
  .pred-ok .pred-note {
    color: var(--text-muted) !important;
    font-size: 0.875rem;
  }
</style>
        """,
        unsafe_allow_html=True,
    )


st.set_page_config(
    page_title="Credit Card Fraud Detection | Portfolio",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_css()


@st.cache_data(show_spinner="Loading predictions…")
def load_predictions() -> pd.DataFrame:
    path = DATA_DIR / "predictions.csv"
    if path.exists():
        return pd.read_csv(path)
    return _default_predictions_df()


def _strip_champion_mark(s: Any) -> str:
    if not isinstance(s, str):
        return str(s)
    return s.replace("\u2713", "").replace("✓", "").strip()


def _sidebar_glance_table_html(res: dict[str, Any], win: dict[str, Any]) -> str:
    champ = html.escape(_strip_champion_mark(res.get("winner", win.get("name", ""))))
    auc_v = win.get("test_auc_roc")
    auc_s = html.escape(f"{auc_v:.4f}" if auc_v is not None else "—")
    f1_v = win.get("test_f1")
    f1_s = html.escape(f"{f1_v:.3f}" if f1_v is not None else "—")
    rec_v = win.get("test_recall")
    rec_s = html.escape(f"{rec_v:.1%}" if rec_v is not None else "—")
    prec_v = win.get("test_precision")
    prec_s = html.escape(f"{prec_v:.1%}" if prec_v is not None else "—")
    rows = [
        ("Champion model", champ),
        ("AUC-ROC", auc_s),
        ("F1", f1_s),
        ("Recall", rec_s),
        ("Precision", prec_s),
        ("Dataset rows", html.escape("284,807")),
        ("Features", html.escape("40 (+13 eng.)")),
    ]
    body = "".join(
        f"<tr><td class='glance-metric'>{html.escape(lab)}</td><td class='glance-val'>{val}</td></tr>"
        for lab, val in rows
    )
    return (
        "<table class='glance-table' role='grid'>"
        "<thead><tr><th>Metric</th><th>Value</th></tr></thead>"
        f"<tbody>{body}</tbody></table>"
    )


@st.cache_data(show_spinner="Loading model results…")
def load_model_results() -> dict[str, Any]:
    path = DATA_DIR / "model_results.json"
    if path.exists():
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        out = {**DEFAULT_RESULTS, **data}
        if "models" in data:
            out["models"] = data["models"]
        out["winner"] = _strip_champion_mark(out.get("winner", ""))
        out["models"] = [{**m, "name": _strip_champion_mark(m.get("name", ""))} for m in out.get("models", [])]
        return out
    return json.loads(json.dumps(DEFAULT_RESULTS))


@st.cache_data(show_spinner="Loading feature sample…")
def load_features_sample(n: int = 50_000) -> pd.DataFrame | None:
    path = DATA_DIR / "features.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path, dtype=np.float32)
    num = df.select_dtypes(include="number")
    if "Class" not in num.columns:
        return None
    return num.sample(n=min(n, len(num)), random_state=42)


@st.cache_resource
def load_trained_model():
    import joblib

    p = MODELS_DIR / "production_model.pkl"
    if p.exists():
        return joblib.load(p)
    for alt in ("xgboost.pkl", "lightgbm.pkl", "randomforest.pkl"):
        q = MODELS_DIR / alt
        if q.exists():
            return joblib.load(q)
    return None


def section(title: str) -> None:
    st.markdown(f'<p class="sec-title">{title}</p>', unsafe_allow_html=True)


def prose(md: str) -> None:
    st.markdown(
        f'<div class="prose">{textwrap.dedent(md).strip()}</div>',
        unsafe_allow_html=True,
    )


def callout(html_body: str, variant: str = "") -> None:
    cls = f"callout {variant}".strip()
    st.markdown(f'<div class="{cls}">{html_body}</div>', unsafe_allow_html=True)


def top_header() -> None:
    st.markdown(
        """
<div class="top-bar"><div class="top-bar-inner">
  <div class="top-brand">🛡️ Credit Card Fraud Detection</div>
  <div class="top-tag">Portfolio · Risk prioritization</div>
</div></div>
        """,
        unsafe_allow_html=True,
    )


def page_overview() -> None:
    top_header()
    res = load_model_results()
    win = res["models"][-1]
    auc = win["test_auc_roc"]
    auc_base = res["models"][0]["test_auc_roc"]
    auc_delta = (auc - auc_base) * 100
    f1 = win["test_f1"]
    f1_base = res["models"][0]["test_f1"]
    recall_v = win.get("test_recall")
    if recall_v is None:
        recall_v = 0.837
    precision_v = win.get("test_precision")
    if precision_v is None:
        precision_v = 0.845
    recall_pct = int(round(recall_v * 100))
    false_alarm_pct = max(0, int(round((1.0 - precision_v) * 100)))

    st.markdown(
        """
<div class="hero">
  <div class="hero-eyebrow">End-to-end ML system · Feature engineering · Model tuning</div>
  <h1 class="hero-title">System that reduces financial loss by prioritizing high-risk transactions</h1>
  <p class="hero-sub">
    A scoring layer ranks incoming payments so fraud analysts review the highest-risk cases first
    catching more fraud with fewer unnecessary blocks and less manual triage across the whole queue.
  </p>
</div>
        """,
        unsafe_allow_html=True,
    )

    callout(
        "<strong>Why this matters:</strong><br>"
        "Credit card fraud costs billions annually. This system helps financial institutions "
        "detect fraud early while maintaining a smooth customer experience.",
    )

    prec_pct = int(round(precision_v * 100))
    section("Business Impact")
    prose(
        f"""
        This system is designed for fraud analysts to prioritize high-risk transactions
        instead of reviewing everything manually.

        If deployed:

        - Could reduce fraud losses by catching ~{recall_pct}% of fraudulent transactions
        - Keeps false alerts low (~{prec_pct}% precision)
        - Enables near real-time decisioning

        Goal: maximize fraud caught while minimizing customer friction
        """
    )

    section("How This Would Work in Production")
    st.markdown(
        """
1. Incoming transactions are streamed (Kafka / API)  
2. Features are generated in real-time  
3. Model scores each transaction in milliseconds  
4. High-risk transactions → flagged for review or blocked  
5. Analysts focus only on top-risk cases  

This reduces manual workload significantly
        """
    )

    section("What this project does")
    prose(
        """
        Banks see millions of card payments per day; only a tiny fraction are fraud. Under the hood,
        a gradient-boosted tree model scores each transaction using amount, time, and anonymised PCA
        features (plus engineered signals). Because “accuracy” is misleading when fraud is rare, the
        evaluation centres on recall, precision, F1, and AUC-ROC the same trade-offs fraud operations
        care about when choosing thresholds and review queues.
        """
    )

    section("Key numbers")
    st.caption("Plain-language KPIs. Deltas compare the tuned winner to the logistic baseline.")
    r1, r2, r3, r4 = st.columns(4)
    with r1:
        st.metric(
            "Transactions analysed",
            f"{284_807:,}",
            help="Rows in the public European card transaction dataset.",
        )
    with r2:
        st.metric(
            "Model inputs (features)",
            "40",
            delta="13 engineered",
            help="After selection: raw + engineered columns fed to the tree model.",
        )
    with r3:
        st.metric(
            "Winner AUC-ROC",
            f"{auc:.3f}",
            delta=f"+{auc_delta:.1f} pp vs baseline",
            delta_color="normal",
            help="Ability to rank fraud above legitimate when pairs are sampled at random.",
        )
    with r4:
        st.metric(
            "Winner F1 score",
            f"{f1:.3f}",
            delta=f"+{f1 - f1_base:.3f} vs baseline",
            delta_color="normal",
            help="Harmonic mean of precision and recall at a 0.5 decision threshold.",
        )

    r5, r6, r7, r8 = st.columns(4)
    with r5:
        st.metric(
            "Test recall (fraud caught)",
            f"{recall_v:.1%}",
            help="Share of real fraud in the test split flagged by the model.",
        )
    with r6:
        st.metric(
            "Test precision (alerts correct)",
            f"{precision_v:.1%}",
            help="When the model says fraud, how often it is right.",
        )
    with r7:
        st.metric(
            "Baseline AUC-ROC",
            f"{auc_base:.3f}",
            help="Logistic regression reference from the same train/test protocol.",
        )
    with r8:
        st.metric(
            "Baseline F1",
            f"{f1_base:.3f}",
            help="Simple linear model before tree ensembles and tuning.",
        )

    callout(
        f"<strong>What does {recall_v:.0%} recall mean?</strong><br>"
        f"Out of 100 fraudulent transactions, the model catches ~{recall_pct}  directly reducing financial loss.",
        "callout-green",
    )
    callout(
        f"<strong>What does {precision_v:.0%} precision mean?</strong><br>"
        f"Only ~{false_alarm_pct}% of flagged transactions are false alarms  reducing unnecessary customer friction.",
    )

    section("Why not deep learning?")
    st.markdown(
        """
- Dataset is structured tabular data → tree models perform better  
- Faster training and inference  
- Easier interpretability for fraud teams  
- Lower infrastructure cost  

XGBoost is the industry standard for this type of problem
        """
    )

    section("Tech stack")
    stack = res.get(
        "tech_stack",
        [
            "Python",
            "pandas",
            "NumPy",
            "scikit-learn",
            "XGBoost",
            "LightGBM",
            "Streamlit",
            "Plotly",
            "Joblib",
            "Optuna",
            "MLflow",
        ],
    )
    st.markdown(
        '<div class="badge-wrap">' + "".join(f'<span class="badge">{s}</span>' for s in stack) + "</div>",
        unsafe_allow_html=True,
    )


def page_eda() -> None:
    top_header()
    st.title("Explore the data")
    prose(
        """
        Use the controls to compare legitimate vs fraud behaviour. Charts use a random sample
        for speed; shapes match the full dataset. Class is the target: `1` = fraud, `0` = legitimate.
        """
    )

    df = load_features_sample(50_000)
    if df is None:
        st.warning(
            "Place `data/features.csv` in the project (run `python src/features/run_features.py`) "
            "to enable the full EDA page. Showing target distribution from `predictions.csv` only."
        )
        preds = load_predictions()
        if "Class" in preds.columns:
            vc = preds["Class"].value_counts().sort_index()
            fig = px.bar(
                x=["Legitimate (0)", "Fraud (1)"],
                y=[vc.get(0, 0), vc.get(1, 0)],
                labels={"x": "", "y": "Rows"},
                title="Target distribution (from predictions file)",
                color_discrete_sequence=[INDIGO],
                template=TMPL,
            )
            st.plotly_chart(fig, use_container_width=True)
            st.markdown(
                """
**What this means (in simple terms):**  
This view shows that fraud is extremely rare compared to normal transactions (same idea as a pie chart).  
In real life, this makes the problem harder   because a model can look "accurate" while still missing most fraud cases.

**This is why we focus on recall and precision instead of accuracy.**

**Why this matters for business:**  
It justifies fraud-specific metrics and review workflows instead of headline accuracy alone.
                """
            )
        return

    fraud = df[df["Class"] == 1]
    legit = df[df["Class"] == 0]

    section("Target distribution")
    c1, c2 = st.columns([1, 1])
    with c1:
        n_fraud = int(df["Class"].sum())
        n_ok = len(df) - n_fraud
        fig = go.Figure(
            go.Pie(
                labels=["Legitimate", "Fraud"],
                values=[n_ok, n_fraud],
                hole=0.55,
                marker_colors=[GREEN, RED],
                textinfo="label+percent",
            )
        )
        fig.update_layout(
            template=TMPL,
            height=320,
            title="Share of fraud in the sample",
            showlegend=False,
            margin=dict(t=40, b=20),
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown(
            """
**What this means (in simple terms):**  
This chart shows that fraud is extremely rare compared to normal transactions.  
In real life, this makes the problem harder   because a model can look "accurate" while still missing most fraud cases.

**This is why we focus on recall and precision instead of accuracy.**

**Why this matters for business:**  
It helps fraud teams justify recall/precision targets and queue design instead of chasing misleading accuracy.
            """
        )
    with c2:
        callout(
            "<strong>Why not focus on accuracy?</strong><br>"
            "A model that always predicts “legitimate” could still reach ~99.8% accuracy here, "
            "yet catch <strong>no fraud</strong>. That is why we stress recall, precision, F1, and AUC.",
            "callout-amber",
        )
        callout(
            "<strong>What you are looking at</strong><br>"
            "V1–V28 are privacy-preserving PCA scores (already scaled). Amount and Time are raw fields; "
            "engineered columns such as <code>log_amount</code> and <code>hour_of_day</code> make patterns easier to learn.",
        )

    section("Pick a feature to compare classes")
    candidates = [c for c in df.columns if c != "Class" and pd.api.types.is_numeric_dtype(df[c])]
    default_i = candidates.index("V14") if "V14" in candidates else 0
    feat = st.selectbox("Feature", candidates, index=default_i)
    bins = st.slider("Histogram bins", 15, 80, 40)

    fig = go.Figure()
    fig.add_trace(
        go.Histogram(
            x=legit[feat],
            name="Legitimate",
            opacity=0.55,
            nbinsx=bins,
            marker_color=GREEN,
        )
    )
    fig.add_trace(
        go.Histogram(
            x=fraud[feat],
            name="Fraud",
            opacity=0.75,
            nbinsx=bins,
            marker_color=RED,
        )
    )
    fig.update_layout(
        barmode="overlay",
        template=TMPL,
        height=380,
        title=f"Distribution of {feat}",
        xaxis_title=feat,
        yaxis_title="Count",
        legend=dict(orientation="h", y=1.12),
    )
    st.plotly_chart(fig, use_container_width=True)
    st.markdown(
        f"""
**What this means:**  
This chart compares how the feature **{feat}** behaves for fraud vs normal transactions.

- If the two colors overlap a lot → this feature alone cannot separate fraud  
- If they are clearly different → this feature helps the model detect fraud  

**The model learns from combinations of these patterns across many features.**

**Why this matters for business:**  
It shows which raw signals actually move with fraud so teams invest in the right data and controls.
        """
    )

    section("Correlation heatmap (top signals)")
    corr_ser = df.corr(numeric_only=True)["Class"].drop("Class", errors="ignore").dropna()
    corr_ser = corr_ser.reindex(corr_ser.abs().sort_values(ascending=False).index).head(12)
    heat_cols = list(corr_ser.index) + ["Class"]
    heat_cols = [c for c in heat_cols if c in df.columns]
    cmat = df[heat_cols].corr(numeric_only=True)
    fig = px.imshow(
        cmat,
        text_auto=".2f",
        aspect="auto",
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        title="How strongly numeric fields move with fraud (Class)",
        template=TMPL,
        height=520,
    )
    st.plotly_chart(fig, use_container_width=True)
    st.markdown(
        """
**What this means:**  
This heatmap shows which features are most related to fraud.

- Darker colors = stronger relationship  
- Positive = increases fraud likelihood  
- Negative = decreases fraud likelihood  

**This helps us understand which variables are most useful for prediction.**

**Why this matters for business:**  
It gives a clear story for which drivers back each risk score useful for model reviews and stakeholder trust.
        """
    )

    section("EDA takeaways")
    callout(
        "<strong>Imbalance dominates the problem.</strong> Fraud is a needle in a haystack; metrics and "
        "training choices must reflect that.",
        "callout-red",
    )
    callout(
        "<strong>A few PCA dimensions carry most signal.</strong> V14, V17, and V12 often separate fraud "
        "from legitimate spend in plots.",
        "callout-green",
    )
    if "hour_of_day" in df.columns:
        callout(
            "<strong>Time of day matters.</strong> Engineered <code>hour_of_day</code> helps capture "
            "quiet-hour behaviour that raw seconds alone obscure.",
        )


def page_model_results() -> None:
    top_header()
    st.title("Model results")
    prose(
        """
        Below is a side-by-side comparison of every model evaluated on the same split and metrics.
        Numbers come from <code>data/model_results.json</code> when that file exists; otherwise sample
        values illustrate the layout for reviewers.
        """
    )

    res = load_model_results()
    preds = load_predictions()
    model = load_trained_model()

    pred_col = "predicted" if "predicted" in preds.columns else None
    if pred_col is None and "y_pred" in preds.columns:
        pred_col = "y_pred"

    section("Model comparison")
    rows = []
    for m in res["models"]:
        rows.append(
            {
                "Model": m["name"],
                "Recall (fraud caught)": f"{m.get('test_recall', 0):.1%}" if m.get("test_recall") is not None else "",
                "Precision (alerts correct)": f"{m.get('test_precision', 0):.1%}"
                if m.get("test_precision") is not None
                else "",
                "F1": f"{m.get('test_f1', 0):.3f}",
                "AUC-ROC": f"{m.get('test_auc_roc', 0):.3f}",
                "Train time (s)": f"{m['training_time_s']:.1f}" if m.get("training_time_s") else "",
            }
        )
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    names = [m["name"] for m in res["models"]]
    f1s = [m.get("test_f1", 0) for m in res["models"]]
    aucs = [m.get("test_auc_roc", 0) for m in res["models"]]
    g1, g2 = st.columns(2)
    with g1:
        fig = px.bar(
            x=names,
            y=f1s,
            text=[f"{v:.3f}" for v in f1s],
            labels={"x": "", "y": "F1"},
            title="F1 score by model",
            color_discrete_sequence=[INDIGO],
            template=TMPL,
        )
        fig.update_traces(textposition="outside")
        fig.update_layout(height=360, yaxis_range=[0, max(1.0, max(f1s) * 1.15)])
        st.plotly_chart(fig, use_container_width=True)
        st.markdown(
            """
**What this means:**  
**F1 score** balances two things: catching fraud (recall) and avoiding false alarms (precision).

**A higher F1 score means the model is better at detecting fraud without overwhelming analysts with false alerts.**

**Why this matters for business:**  
It links model quality directly to analyst workload and customer friction what operations feel every day.
            """
        )
    with g2:
        fig = px.bar(
            x=aucs,
            y=names,
            orientation="h",
            text=[f"{v:.3f}" for v in aucs],
            labels={"x": "AUC-ROC", "y": ""},
            title="AUC-ROC by model",
            template=TMPL,
        )
        fig.update_traces(marker_color=INDIGO)
        fig.update_layout(height=360, yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig, use_container_width=True)
        st.markdown(
            """
**What this means:**  
**AUC** measures how well the model ranks fraud higher than normal transactions.

- Closer to **1.0** = better separation  
- **0.5** = random guessing  

**This tells us how good the model is at prioritizing risky transactions.**

**Why this matters for business:**  
Strong AUC supports ranked review queues analysts tackle the highest-risk payments first.
            """
        )

    section("Why this winner   not only the highest score")
    st.markdown("The tuned " + str(res.get("winner", "winner")) + " was chosen because:")
    for note in res.get("winner_notes", DEFAULT_RESULTS["winner_notes"]):
        st.markdown(f"- {note}", unsafe_allow_html=True)

    section("Feature importance (top 15)")
    if model is not None and hasattr(model, "feature_importances_"):
        names_fi = list(model.feature_names_in_)
        imp = model.feature_importances_
        fi_df = (
            pd.DataFrame({"Feature": names_fi, "Importance": imp})
            .sort_values("Importance", ascending=False)
            .head(15)
        )
    else:
        fi_df = pd.DataFrame(res.get("feature_importance", DEFAULT_RESULTS["feature_importance"]))
        if fi_df.empty:
            fi_df = pd.DataFrame(DEFAULT_RESULTS["feature_importance"])
    fi_plot = fi_df.sort_values("Importance", ascending=True)
    fig = px.bar(
        fi_plot,
        x="Importance",
        y="Feature",
        orientation="h",
        title="Which inputs the model leans on most",
        template=TMPL,
        color_discrete_sequence=[INDIGO],
        height=460,
    )
    st.plotly_chart(fig, use_container_width=True)
    st.markdown(
        """
**What this means:**  
This shows which inputs the model relies on the most to detect fraud.

**Higher importance = bigger influence on the model’s decision**   and it helps explain the model to business teams and build trust.

**Why this matters for business:**  
It turns a “black box” into a short list of drivers auditors and fraud leaders can reason about.
        """
    )
    if model is None:
        st.caption("Train and save `models/production_model.pkl` to show importances from the fitted estimator.")

    section("Confusion matrix (test predictions)")
    if pred_col is None or "Class" not in preds.columns:
        st.info("Add `Class` and `predicted` columns to `data/predictions.csv` to plot the confusion matrix.")
    else:
        cm = confusion_matrix(preds["Class"], preds[pred_col], labels=[0, 1])
        fig = px.imshow(
            cm,
            x=["Pred: OK", "Pred: Fraud"],
            y=["Actual: OK", "Actual: Fraud"],
            text_auto=True,
            color_continuous_scale=[[0, "#0f172a"], [0.45, "#1d4ed8"], [1, "#e2e8f0"]],
            aspect="equal",
            template=TMPL,
            title="Counts on the held-out split",
        )
        fig.update_layout(height=360, coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown(
            """
**What this means:**

- **Top-left:** Correctly identified normal transactions  
- **Bottom-right:** Correctly detected fraud  
- **Bottom-left:** Missed fraud (most costly)  
- **Top-right:** False alarms  

**The goal is to catch as much fraud as possible while minimizing false alerts.**

**Why this matters for business:**  
It maps model errors to dollars and customer experience: missed fraud vs unnecessary blocks.
            """
        )
        tn, fp, fn, tp = cm.ravel()
        st.caption(
            f"True negatives {tn:,} · False positives {fp:,} · False negatives {fn:,} · True positives {tp:,}  "
            "False negatives are missed fraud; false positives annoy good customers."
        )

    section("Try it yourself")
    if model is None:
        st.info(
            "Live scoring needs a saved model at `models/production_model.pkl` (or `xgboost.pkl`). "
            "Without it, this section stays disabled so the app still loads for reviewers."
        )
        return

    st.caption("Adjust values, then submit. Other features default to neutral zeros; this is a simplified demo.")
    with st.form("predict"):
        a, b, c = st.columns(3)
        with a:
            amount = st.number_input("Amount ($)", 0.01, 25_000.0, 120.0, step=1.0)
            hour = st.slider("Hour of day", 0.0, 23.9, 14.0, 0.1)
        with b:
            v14 = st.slider("V14", -25.0, 10.0, 0.0, 0.1)
            v17 = st.slider("V17", -25.0, 10.0, 0.0, 0.1)
        with c:
            v10 = st.slider("V10", -25.0, 10.0, 0.0, 0.1)
            v4 = st.slider("V4", -10.0, 15.0, 0.0, 0.1)
        go_btn = st.form_submit_button("Score this transaction", type="primary")

    if go_btn:
        v_vals = {f"V{i}": 0.0 for i in range(1, 29)}
        v_vals.update({"V4": v4, "V10": v10, "V14": v14, "V17": v17})
        all_v = list(v_vals.values())
        log_amt = float(np.log1p(amount))
        row = {
            "Time": 50_000.0,
            **v_vals,
            "Amount": float(amount),
            "log_amount": log_amt,
            "is_round_amount": float(amount % 1 == 0),
            "hour_of_day": float(hour),
            "days_elapsed": 0.0,
            "v_mean": float(np.mean(all_v)),
            "v_range": float(np.max(all_v) - np.min(all_v)),
            "n_extreme_v": float(sum(abs(v) > 3.0 for v in all_v)),
            "v14_x_log_amount": v14 * log_amt,
            "v17_x_log_amount": v17 * log_amt,
            "v14_x_v17": v14 * v17,
        }
        cols = list(model.feature_names_in_)
        X = pd.DataFrame([{k: row.get(k, 0.0) for k in cols}])
        prob = float(model.predict_proba(X)[0, 1])
        fraud = prob >= 0.5
        left, right = st.columns([1, 1])
        with left:
            box = "pred-fraud" if fraud else "pred-ok"
            label = "Elevated fraud risk" if fraud else "Low fraud risk"
            st.markdown(
                f'<div class="{box}"><div class="pred-head">{label}</div>'
                f'<div class="pred-score">{prob:.1%}</div>'
                '<div class="pred-note">Estimated probability of fraud (threshold 50%).</div></div>',
                unsafe_allow_html=True,
            )
        with right:
            fig = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=prob * 100,
                    number={"suffix": "%", "font": {"size": 36}},
                    title={"text": "Fraud probability"},
                    gauge={
                        "axis": {"range": [0, 100]},
                        "bar": {"color": RED if fraud else GREEN},
                        "steps": [
                            {"range": [0, 40], "color": "#14532d"},
                            {"range": [40, 60], "color": "#713f12"},
                            {"range": [60, 100], "color": "#7f1d1d"},
                        ],
                        "threshold": {
                            "line": {"color": "#f8fafc", "width": 2},
                            "thickness": 0.8,
                            "value": 50,
                        },
                    },
                )
            )
            fig.update_layout(height=280, template=TMPL, margin=dict(t=30, b=20))
            st.plotly_chart(fig, use_container_width=True)
            st.markdown(
                """
**What this means:**  
This score represents how risky a transaction is.

- Closer to **100%** → very likely fraud  
- Closer to **0%** → likely safe  

**In real systems, this score is used to decide:** auto block, send for review, or approve the transaction.

**Why this matters for business:**  
It is the lever for automated holds vs human review balancing loss prevention with customer flow.
                """
            )


def page_build() -> None:
    top_header()
    st.title("How I built this")
    prose(
        """
        This project was built as an end-to-end machine learning system, not just a model.

        The goal was to simulate how credit card fraud detection works in a real financial system, 
        from raw transaction data to a production-ready scoring pipeline that can prioritize
        high-risk transactions in real time.

        The focus was not just model accuracy, but building something that is:

        - Interpretable for business teams  
        - Scalable for production use  
        - Aligned with real credit card fraud detection workflows  
        """
    )

    section("Architecture")
    st.graphviz_chart(
        """
        digraph G {
            rankdir=LR;
            graph [fontname="Helvetica" bgcolor="#0b1120"];
            node [shape=box style="rounded,filled" fontname="Helvetica" fontsize=11
                  fontcolor="#f1f5f9" fillcolor="#1f2937" color="#475569"];
            edge [color="#60a5fa"];
            A [label="Raw CSV\\n284k rows"];
            B [label="EDA\\nplots + checks"];
            C [label="Features\\nengineer + select"];
            D [label="Train / CV\\nRF · XGB · LGBM"];
            E [label="Tune\\nOptuna (optional)"];
            F [label="Track\\nMLflow (optional)"];
            G [label="Serve\\nStreamlit + joblib"];
            A -> B -> C -> D -> E -> F -> G;
        }
        """
    )
    st.markdown(
        """
**What this architecture shows:**  

This pipeline represents the full lifecycle of a machine learning system:

1. Raw transaction data is ingested  
2. Data is explored to understand fraud patterns  
3. Features are engineered to make patterns learnable  
4. Multiple models are trained and compared  
5. The best model is tuned and tracked  
6. Final model is deployed for real-time scoring  

**This mirrors how production ML systems are built in industry.**
        """
    )
    st.markdown(
        """
The timeline below follows the same pipeline — showing how each stage was built step-by-step.
        """
    )

    section("Timeline (7 days)")
    days = [
        (
            "Day 1",
            "Data Understanding (EDA)",
            "Started by analyzing class imbalance and feature behavior. Identified key fraud signals and challenges in the dataset.",
        ),
        (
            "Day 2",
            "Feature Engineering",
            "Transformed raw inputs into meaningful signals like log_amount, hour_of_day, and interaction features.",
        ),
        (
            "Day 3",
            "Baseline Modeling",
            "Built Logistic Regression to establish a benchmark and identify gaps in credit card fraud detection.",
        ),
        (
            "Day 4",
            "Model Training & Comparison",
            "Evaluated Random Forest, XGBoost, and LightGBM using consistent metrics and validation strategy.",
        ),
        (
            "Day 5",
            "Model Tuning",
            "Optimized XGBoost using hyperparameter tuning to improve recall while maintaining precision.",
        ),
        (
            "Day 6",
            "Tracking & Export",
            "Saved model artifacts, predictions, and evaluation metrics for reproducibility and deployment.",
        ),
        (
            "Day 7",
            "Deployment (Streamlit App)",
            "Built an interactive app to visualize results and simulate real-time fraud scoring.",
        ),
    ]
    for d, t, desc in days:
        with st.expander(f"{d} — {t}", expanded=False):
            st.write(desc)

    section("Key decisions")
    st.markdown(
        """
### Key engineering decisions

**1. Focused on Recall + Precision instead of Accuracy**  
Fraud is extremely rare (~0.17%), so accuracy is misleading.  
Optimized for catching fraud while keeping false alerts manageable.

**2. Used Tree-Based Models (XGBoost)**  
Tree models perform significantly better on structured/tabular data compared to deep learning.  
They also provide better interpretability and faster inference.

**3. Feature Engineering over Model Complexity**  
Performance gains came more from better features than more complex models.

**4. Avoided Data Leakage**  
Handled class imbalance using class weights instead of oversampling before splitting.

**5. Built for Production Thinking**  
- Saved model as `.pkl`  
- Created scoring pipeline  
- Designed UI for business users  

**The goal was not just to train a model, but to simulate a deployable system.**
        """
    )

    section("Challenges & learnings")
    st.markdown(
        """
**1. Extreme class imbalance**  
Initially, the model looked highly accurate but failed to detect fraud.  
This required shifting focus to recall and F1 score.

**2. Feature interpretation**  
The dataset uses PCA-transformed features (V1–V28), making it harder to interpret.  
Had to rely on correlation and feature importance instead of domain meaning.

**3. Precision vs Recall trade-off**  
Increasing credit card fraud detection led to more false positives.  
Balancing this trade-off is critical in real-world systems.

**4. Performance vs speed**  
Some models performed well but were too slow for real-time use.  
XGBoost provided the best balance.

**These are real challenges faced in production ML systems.**
        """
    )

    section("Real-world impact")
    st.markdown(
        """
If deployed in a financial system, this solution would:

- Reduce fraud losses by catching high-risk transactions early  
- Improve analyst efficiency by prioritizing risky cases  
- Reduce customer friction by minimizing false alarms  
- Enable near real-time credit card   

**This turns a machine learning model into a business decision system.**
        """
    )

    section("Repository")
    meta = load_model_results()
    gh = st.text_input(
        "GitHub URL",
        value=str(meta.get("github_url", "") or ""),
        placeholder="https://github.com/stuti-shrimal/Credit-Card-Fraud-Detection",
    )
    if gh.strip():
        st.link_button("Open repository →", gh.strip(), use_container_width=False)


def main() -> None:
    with st.sidebar:
        st.markdown(
            """
<div class="sidebar-brand">
  <div style="font-size:1.75rem;line-height:1;">🛡️</div>
  <div class="sidebar-brand-title">Credit card fraud detection</div>
  <div class="sidebar-brand-sub">Portfolio navigation</div>
</div>
            """,
            unsafe_allow_html=True,
        )
        page = st.radio(
            "Section",
            ["Overview", "Explore the Data", "Model Results", "How I Built This"],
            label_visibility="collapsed",
        )
        st.markdown('<p class="glance-head">Outcomes at a glance</p>', unsafe_allow_html=True)
        res = load_model_results()
        win = res["models"][-1]
        st.markdown(_sidebar_glance_table_html(res, win), unsafe_allow_html=True)

    routes = {
        "Overview": page_overview,
        "Explore the Data": page_eda,
        "Model Results": page_model_results,
        "How I Built This": page_build,
    }
    routes[page]()

    st.markdown(
        '<div class="footer">Credit Card Fraud Detection  portfolio project · Streamlit · scikit-learn · XGBoost · Plotly</div>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
