import numpy as np
import pandas as pd

V_COLS = [f"V{i}" for i in range(1, 29)]
EXTREME_V_THRESHOLD = 3.0   # std devs; V-features are already standardised

# Ordered list of every column this module adds  useful for downstream
# selection without hard-coding column names elsewhere.
ENGINEERED_COLS: list[str] = [
    # domain-specific
    "log_amount",
    "amount_bin",
    "is_round_amount",
    "hour_of_day",
    "days_elapsed",
    # statistical
    "v_mean",
    "v_std",
    "v_l2_norm",
    "v_range",
    "n_extreme_v",
    # interactions
    "v14_x_log_amount",
    "v17_x_log_amount",
    "v14_x_v17",
]


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    _add_domain_features(df)
    _add_statistical_features(df)
    _add_interaction_features(df)
    return df


# ── Category 1: Domain-specific ───────────────────────────────────────────────

def _add_domain_features(df: pd.DataFrame) -> None:
    # Raw Amount is right-skewed with a heavy tail of large transactions.
    # log1p compresses that tail into a roughly normal distribution, which
    # helps linear models and distance-based algorithms (KNN, SVM) equally
    # weight small and large transactions without distorting the zero case.
    df["log_amount"] = np.log1p(df["Amount"])

    # Fraud patterns differ sharply by spend tier: micro-transactions (<$1)
    # are a classic card-validity probe; large charges (>$1 000) can indicate
    # an account-takeover purchase. Binning encodes this non-linear boundary
    # as an ordinal signal a tree-based model can exploit directly.
    df["amount_bin"] = pd.cut(
        df["Amount"],
        bins=[-np.inf, 1.0, 100.0, 1_000.0, np.inf],
        labels=["micro", "small", "medium", "large"],
    ).astype(str)

    # Automated fraud scripts often probe card validity with exact whole-dollar
    # amounts (e.g. $1.00, $5.00) because they generate test charges
    # programmatically. A binary flag makes this pattern explicit.
    df["is_round_amount"] = ((df["Amount"] % 1) == 0).astype(np.int8)

    # Time is recorded as seconds elapsed since the first transaction in the
    # dataset. Modulo 86 400 maps it to within-day seconds; dividing by 3 600
    # yields a continuous 0–24 hour proxy. Fraud rates are elevated during
    # late-night / early-morning hours when human oversight is lowest.
    df["hour_of_day"] = (df["Time"] % 86_400) / 3_600

    # Integer day index within the monitoring window (~48 hours in this
    # dataset). Captures day-level temporal drift and any batch patterns
    # (e.g. fraudulent campaigns that run only on certain days).
    df["days_elapsed"] = (df["Time"] // 86_400).astype(np.int32)


# ── Category 2: Statistical aggregates over V-features ────────────────────────

def _add_statistical_features(df: pd.DataFrame) -> None:
    v = df[V_COLS]

    # Average activation across all 28 PCA components. Legitimate transactions
    # cluster tightly around zero (PCA centred the training set); fraud
    # transactions tend to have a shifted mean, making this a cheap summary
    # of overall "outlier-ness" in PCA space.
    df["v_mean"] = v.mean(axis=1)

    # Standard deviation across the 28 components for a single transaction.
    # A very narrow spread suggests the transaction activates only a small
    # subset of PCA dimensions  an unusual pattern worth flagging.
    df["v_std"] = v.std(axis=1)

    # Euclidean distance from the PCA origin. Because the legitimate
    # transactions were used to define the PCA axes, fraud observations
    # typically sit further from the origin  this scalar summarises that
    # distance without requiring the model to reason over 28 dimensions.
    df["v_l2_norm"] = np.sqrt((v ** 2).sum(axis=1))

    # Max minus min of the V values for one transaction. A very wide range
    # means some components are strongly positive while others are strongly
    # negative  a signature of transactions that sit in "unusual corners"
    # of the PCA space simultaneously.
    df["v_range"] = v.max(axis=1) - v.min(axis=1)

    # Count of V components with |value| > 3. Since the features are already
    # standardised, values beyond ±3 are statistically extreme (~0.3% under
    # normality). Multiple extreme components in a single transaction is a
    # concentrated anomaly signal that a raw correlation misses.
    df["n_extreme_v"] = (v.abs() > EXTREME_V_THRESHOLD).sum(axis=1).astype(np.int16)


# ── Category 3: Interaction features ──────────────────────────────────────────

def _add_interaction_features(df: pd.DataFrame) -> None:
    # V14 is the single feature most correlated with fraud (EDA section 5).
    # Multiplying by log_amount encodes the intuition that a large transaction
    # with an anomalous V14 is far more suspicious than an anomalous V14 alone
    # (e.g. a $0.01 probe vs. a $900 purchase).
    df["v14_x_log_amount"] = df["V14"] * df["log_amount"]

    # V17 is the second strongest fraud indicator. The same size-amplification
    # logic applies: a skewed V17 on a high-value transaction is the dangerous
    # combination a linear model cannot capture without this product term.
    df["v17_x_log_amount"] = df["V17"] * df["log_amount"]

    # Joint signal from the two strongest fraud features. When V14 and V17 are
    # simultaneously anomalous the product is large-and-negative (both skew
    # negative for fraud), creating a signal that is stronger than either
    # feature alone and lets a linear model capture the conjunction cheaply.
    df["v14_x_v17"] = df["V14"] * df["V17"]


# ── Feature selection ─────────────────────────────────────────────────────────

CORR_THRESHOLD = 0.95       # drop one of any pair with |r| above this
VARIANCE_FLOOR_RATIO = 0.01  # drop features whose variance < this × overall mean variance


def select_features(
    df: pd.DataFrame,
    corr_threshold: float = CORR_THRESHOLD,
    variance_floor_ratio: float = VARIANCE_FLOOR_RATIO,
    target: str = "Class",
) -> tuple[list[str], pd.DataFrame]:
    """Return (selected_feature_names, reduced_df) after removing near-constant
    and highly-collinear numeric features.  The target column is always kept
    and is never treated as a candidate for removal."""

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    candidates = [c for c in numeric_cols if c != target]
    dropped: list[tuple[str, str]] = []   # (column, reason)

    # ── Pass 1: variance filter ───────────────────────────────────────────────
    # A near-constant feature carries almost no information: its variance is
    # close to zero, so it cannot discriminate between classes.  We flag any
    # feature whose variance falls below `variance_floor_ratio` times the mean
    # variance across all candidates  a relative threshold that adapts to the
    # scale of the dataset rather than requiring a hand-picked absolute value.
    variances = df[candidates].var()
    # Median is used instead of mean so that one extreme-scale column (e.g.
    # raw Time in seconds, variance ~2.3 billion) cannot inflate the threshold
    # and wipe out every other feature.
    overall_median_var = variances.median()
    variance_threshold = variance_floor_ratio * overall_median_var

    low_var_cols = variances[variances < variance_threshold].index.tolist()
    for col in low_var_cols:
        dropped.append((col, f"variance {variances[col]:.6f} < threshold {variance_threshold:.6f})"))

    survivors = [c for c in candidates if c not in low_var_cols]

    # ── Pass 2: correlation filter ────────────────────────────────────────────
    # When two features share |r| > corr_threshold they are almost linearly
    # redundant: including both inflates dimensionality without adding signal
    # and can destabilise coefficient-based models.  We iterate the upper
    # triangle of the correlation matrix and, for each offending pair, drop the
    # column that appears *later* in the column order  i.e. we keep the first
    # feature encountered (original columns first, then engineered ones).
    corr_matrix = df[survivors].corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape, dtype=bool), k=1))

    corr_drop: set[str] = set()
    for col in upper.columns:
        if col in corr_drop:
            continue
        redundant = upper.index[upper[col] > corr_threshold].tolist()
        for partner in redundant:
            if partner not in corr_drop:
                r = corr_matrix.loc[col, partner]
                dropped.append((partner, f"correlation {r:.4f} > {corr_threshold} with '{col}'"))
                corr_drop.add(partner)

    survivors = [c for c in survivors if c not in corr_drop]

    # ── Build output ──────────────────────────────────────────────────────────
    _log_selection(dropped, survivors, target, overall_median_var, variance_threshold)

    keep = survivors + ([target] if target in df.columns else [])
    return survivors, df[keep]


def _log_selection(
    dropped: list[tuple[str, str]],
    survivors: list[str],
    target: str,
    overall_median_var: float,
    variance_threshold: float,
) -> None:
    low_var  = [(c, r) for c, r in dropped if r.startswith("variance")]
    high_cor = [(c, r) for c, r in dropped if r.startswith("corr")]

    print(f"\n{'─'*60}")
    print("Feature selection report")
    print(f"{'─'*60}")
    print(f"Overall median variance  : {overall_median_var:.6f}")
    print(f"Variance floor (×{VARIANCE_FLOOR_RATIO})    : {variance_threshold:.6f}")

    if low_var:
        print(f"\nDropped  low variance ({len(low_var)}):")
        for col, reason in low_var:
            print(f"  ✗  {col:<28}  {reason}")
    else:
        print("\nDropped  low variance (0): none")

    if high_cor:
        print(f"\nDropped  high correlation ({len(high_cor)}):")
        for col, reason in high_cor:
            print(f"  ✗  {col:<28}  {reason}")
    else:
        print("\nDropped  high correlation (0): none")

    print(f"\nRetained : {len(survivors)} features  (+ target '{target}')")
    print(f"{'─'*60}\n")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from data.loader import load_csv

    DATA_DIR = Path(__file__).resolve().parents[2] / "data"
    cleaned_path = DATA_DIR / "cleaned.csv"
    raw_path = DATA_DIR / "creditcard.csv"

    source = "cleaned.csv" if cleaned_path.exists() else "creditcard.csv"
    print(f"Loading '{source}' ...")
    df_raw = load_csv(source)
    print(f"Before : {df_raw.shape[0]:,} rows × {df_raw.shape[1]} columns")

    df_eng = create_features(df_raw)
    new_cols = [c for c in df_eng.columns if c not in df_raw.columns]
    print(f"After  : {df_eng.shape[0]:,} rows × {df_eng.shape[1]} columns")
    print(f"Added  : {len(new_cols)} features\n")

    print(f"{'Feature':<22}  {'dtype':<10}  {'mean':>10}  {'std':>10}  {'min':>10}  {'max':>10}")
    print("-" * 78)
    for col in new_cols:
        s = df_eng[col]
        if pd.api.types.is_numeric_dtype(s):
            print(f"{col:<22}  {str(s.dtype):<10}  {s.mean():>10.4f}  {s.std():>10.4f}  {s.min():>10.4f}  {s.max():>10.4f}")
        else:
            vc = s.value_counts()
            print(f"{col:<22}  {str(s.dtype):<10}  top={vc.index[0]!r} ({vc.iloc[0]:,})")

    selected, df_sel = select_features(df_eng)
    print(f"Selected dataframe : {df_sel.shape[0]:,} rows × {df_sel.shape[1]} columns")
    print(f"Selected features  : {selected}")
