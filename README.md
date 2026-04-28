# Credit Card Fraud Detection

**End-to-end ML system that scores transactions by fraud risk so analysts review the highest-risk cases first.** — [Live Demo](https://credit-card-fraud-detection-stuti.streamlit.app/)

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Architecture](#architecture)
3. [Results](#results)
4. [Tech Stack](#tech-stack)
5. [Setup & Installation](#setup--installation)
6. [How to Run](#how-to-run)
7. [Feature Engineering](#feature-engineering)
8. [Key Decisions & Lessons](#key-decisions--lessons)
9. [File Structure](#file-structure)

---

## Project Overview

### The problem

Credit card fraud costs the global financial industry tens of billions of dollars every year. The challenge is not simply detecting fraud — it is detecting it accurately enough that:

- **Genuine fraud is caught** before money leaves the account (maximise recall)
- **Good customers are not blocked** unnecessarily (maintain precision)
- **Analysts are not overwhelmed** with thousands of false alerts per day (keep false-positive rate low)

A naïve model that flags every transaction as legitimate looks "99.8% accurate" yet catches zero fraud. This project addresses that with a proper ML pipeline evaluated on the metrics that actually matter to a fraud operations team.

### End user

A **fraud analyst** at a card-issuing bank. The model does not make final decisions — it assigns a fraud probability to each transaction so the analyst's review queue is sorted by risk. The highest-probability cases land at the top; low-risk transactions are cleared automatically.

### Data

The [Kaggle European Credit Card dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) contains two days of real transactions from European cardholders:

| Property | Value |
|---|---|
| Total rows | 284,807 |
| Fraud cases | 492 (0.17 %) |
| Features | 30 raw + 13 engineered = 43 total |
| Target | `Class` — 1 = fraud, 0 = legitimate |
| V1–V28 | PCA-anonymised, already scaled |
| `Amount` | Raw transaction amount in EUR |
| `Time` | Seconds elapsed since first transaction in dataset |

### What the model outputs

A **fraud probability score between 0 and 1** for each incoming transaction. At a 0.5 decision threshold the tuned XGBoost model catches **83.7 % of fraud cases** with **84.5 % precision** on the held-out test set.

### Key design decision

Tree-based models — not deep learning. The dataset is structured tabular data with a handful of highly informative features. Gradient-boosted trees outperform neural networks on this class of problem, train in seconds rather than hours, and produce feature importances that a fraud team can audit and explain to regulators.

---

## Architecture

```
 Raw CSV (284k rows)
        │
        ▼
 ┌─────────────┐
 │     EDA     │  Understand imbalance, feature distributions,
 │             │  correlation with Class
 └──────┬──────┘
        │
        ▼
 ┌─────────────┐
 │   Cleaner   │  Drop high-null columns, handle duplicates,
 │             │  coerce dtypes, run quality gate
 └──────┬──────┘
        │
        ▼
 ┌─────────────────┐
 │    Features     │  Engineer 13 new columns (domain, statistical,
 │   Engineering   │  interaction); select by variance + correlation
 └──────┬──────────┘
        │
        ▼
 ┌──────────────────────────────────────────┐
 │              Model Training              │
 │                                          │
 │  Baseline         Random      XGBoost    │
 │  Logistic    →    Forest  →  (candidate) │
 │  Regression                              │
 │                                    │     │
 │                                    ▼     │
 │                             XGBoost      │
 │                             (Optuna      │
 │                              tuned)      │
 └──────────────────────────────────────────┘
        │
        ├── MLflow experiment tracking (optional)
        │
        ▼
 ┌─────────────────┐
 │  Saved artefacts│  production_model.pkl · predictions.csv
 │                 │  model_results.json · best_params.json
 └──────┬──────────┘
        │
        ▼
 ┌─────────────────┐
 │  Streamlit App  │  Live scoring · model comparison charts ·
 │  (port 8501)    │  feature importance · confusion matrix
 └─────────────────┘
```

**Data flow in production (hypothetical deployment)**

```
Kafka stream ──► Feature service ──► Model API ──► Risk score
                 (real-time feats)   (XGBoost)     │
                                                   ├─ score < 0.3  → auto-approve
                                                   ├─ 0.3–0.7      → soft review
                                                   └─ score > 0.7  → block + alert
```

---

## Results

### Model comparison

| Model | Recall | Precision | F1 | AUC-ROC | Train time |
|---|:---:|:---:|:---:|:---:|:---:|
| Baseline — Logistic Regression | 59.2 % | 78.4 % | 0.674 | 0.941 | 2 s |
| Random Forest | 78.0 % | 81.0 % | 0.817 | 0.960 | 214 s |
| XGBoost (default params) | 82.0 % | 83.0 % | 0.835 | 0.975 | 6 s |
| **XGBoost (Optuna tuned) ✓** | **83.7 %** | **84.5 %** | **0.841** | **0.979** | **4.9 s** |

**Improvement over baseline:** +24.5 pp recall · +6.1 pp precision · +0.167 F1 · +0.038 AUC-ROC

### Why the tuned XGBoost won

- **Highest recall on the hold-out set** — more stolen money stopped before it leaves the account
- **Strong precision** — most alerts are real fraud, so analysts are not flooded with false positives
- **Best F1** — optimal balance between catching fraud and avoiding customer friction under severe class imbalance
- **Fast inference** — single-transaction scoring in milliseconds, suitable for near-real-time payment decisioning
- **Random Forest was 43× slower to train** with lower recall despite its large ensemble size

### Why not accuracy?

A model that predicts *every* transaction as legitimate reaches **99.83 % accuracy** while catching **zero fraud**. Recall, precision, F1, and AUC-ROC measure what matters: how well the model separates fraud from legitimate spend.

---

## Tech Stack

| Tool | Purpose |
|---|---|
| **Python 3.9** | Core language |
| **pandas** | Data loading, cleaning, feature tables |
| **NumPy** | Vectorised feature computation |
| **scikit-learn** | Logistic Regression, Random Forest, train/test split, metrics |
| **XGBoost** | Production classifier — gradient-boosted trees |
| **LightGBM** | Candidate ensemble model during comparison |
| **Optuna** | Bayesian hyperparameter search for XGBoost tuning |
| **MLflow** | Experiment tracking — logs params, metrics, and artefacts per run |
| **Joblib** | Model serialisation (`.pkl`) and parallel CV |
| **Streamlit** | Interactive portfolio dashboard and live scoring UI |
| **Plotly** | Interactive charts (ROC, confusion matrix, feature importance) |
| **pytest** | Unit and integration tests for data quality, features, and model |
| **ruff** | Fast Python linter — replaces flake8 + isort |
| **Docker** | Reproducible container for the Streamlit app |
| **GitHub Actions** | CI — runs tests and lint on every push to `main` |

---

## Setup & Installation

### Prerequisites

- Python 3.9+
- Git
- (Optional) Docker Desktop for the containerised app

### Clone and install

```bash
git clone https://github.com/stuti-shrimal/credit-card-fraud-detection.git
cd credit-card-fraud-detection

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -r requirements.txt
pip install -e .                   # installs src/ as an editable package
```

### Download the dataset

Download `creditcard.csv` from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) and place it at:

```
data/creditcard.csv
```

The raw file is ~144 MB and is excluded from the repository via `.gitignore`.

---

## How to Run

### 1 — Streamlit app (fastest — no training required)

The app ships with pre-computed demo data and falls back to synthetic data if no model artefacts are present. Works on a fresh clone.

```bash
streamlit run app/streamlit_app.py
# → opens http://localhost:8501
```

### 2 — Full training pipeline

Run each step in order. Each script saves its output to `data/` or `models/` so the next step can pick it up.

```bash
# Step 1 — Data quality gate
python src/data/quality.py

# Step 2 — Clean the raw data
python src/data/cleaner.py

# Step 3 — Feature engineering + selection
python src/features/run_features.py

# Step 4 — Baseline model (Logistic Regression)
python src/models/baseline.py

# Step 5 — Compare Random Forest, XGBoost, LightGBM
python src/models/compare_models.py

# Step 6 — Hyperparameter tuning (Optuna, ~30 trials)
python src/models/tuning.py

# Step 7 — Full training run with MLflow tracking
python src/models/run_training.py
```

### 3 — Docker

```bash
# Build the image
docker build -t credit-card-fraud-detection .

# Run the container
docker run -p 8501:8501 credit-card-fraud-detection
# → http://localhost:8501

# Or use Compose (mounts data/ and models/ as volumes)
docker compose up
```

### 4 — Tests

```bash
pytest tests/ -v
```

```
tests/test_data_quality.py::test_quality_gate_passes_on_valid_data   PASSED
tests/test_data_quality.py::test_quality_gate_catches_broken_dataset  PASSED
tests/test_features.py::test_feature_count                            PASSED
tests/test_features.py::test_no_nan_in_numeric_output                 PASSED
tests/test_features.py::test_feature_ranges                           PASSED
tests/test_model.py::test_model_loads                                 PASSED
tests/test_model.py::test_predictions_in_valid_range                  PASSED
```

### 5 — Lint

```bash
ruff check src/ app/
```

---

## Feature Engineering

13 features were engineered on top of the 30 raw columns. Feature selection then removed near-constant and highly-collinear columns, leaving 40 features fed to the final model.

### Engineered features

| Feature | Type | Rationale |
|---|---|---|
| `log_amount` | Domain | `log1p(Amount)` compresses the right-skewed distribution; fraud and legitimate spend separate more cleanly in log space |
| `amount_bin` | Domain | Micro (<$1), small, medium, large — card-probing scripts often use round micro-amounts; binning surfaces this as a discrete signal |
| `is_round_amount` | Domain | Automated fraud scripts generate exact whole-dollar test charges; binary flag makes the pattern explicit |
| `hour_of_day` | Domain | `(Time % 86400) / 3600` — fraud rates spike in late-night hours when human oversight is lowest |
| `days_elapsed` | Domain | Day index within the monitoring window; captures day-level temporal drift and batch campaign patterns |
| `v_mean` | Statistical | Mean activation across all 28 PCA components — legitimate transactions cluster near zero; fraud shifts this mean |
| `v_std` | Statistical | Std deviation across V1–V28 — narrow spread flags transactions that activate only a small subset of PCA dimensions |
| `v_l2_norm` | Statistical | Euclidean distance from the PCA origin — fraud observations typically sit further from where legitimate transactions were modelled |
| `v_range` | Statistical | Max minus min of V values — wide range signals a transaction sitting in an "unusual corner" of PCA space |
| `n_extreme_v` | Statistical | Count of V components with \|value\| > 3σ — multiple extreme components in one transaction is a concentrated anomaly signal |
| `v14_x_log_amount` | Interaction | V14 × log_amount — an anomalous V14 on a high-value transaction is far more suspicious than on a $0.01 probe |
| `v17_x_log_amount` | Interaction | V17 × log_amount — same size-amplification logic for the second strongest fraud indicator |
| `v14_x_v17` | Interaction | Product of the two strongest fraud signals — captures the conjunction that neither feature alone can represent linearly |

### Top features by model importance (XGBoost tuned)

| Rank | Feature | Importance | Notes |
|:---:|---|:---:|---|
| 1 | V14 | 0.082 | Strongest single fraud discriminator in EDA |
| 2 | V17 | 0.071 | Second strongest; anti-correlated with V14 on fraud |
| 3 | `v_range` | 0.065 | Engineered — PCA spread; not in raw data |
| 4 | V12 | 0.058 | Negative shift on fraud transactions |
| 5 | `v_mean` | 0.052 | Engineered — aggregate PCA signal |
| 6 | V10 | 0.048 | Correlated with V14 fraud pattern |
| 7 | `v14_x_v17` | 0.045 | Engineered interaction — top-10 despite being derived |
| 8 | V4 | 0.041 | Positive shift on fraud |
| 9 | `log_amount` | 0.038 | Engineered — outranks raw Amount |
| 10 | Amount | 0.035 | Raw amount still contributes independently |

Three of the top ten features were engineered — confirming that feature work contributed meaningfully beyond model selection and tuning alone.

---

## Key Decisions & Lessons

**Optimised for recall, not accuracy.**
Accuracy is meaningless when 99.83 % of the data is one class. Every modelling decision — loss function, class weights, evaluation protocol — was anchored to recall and F1. This is the single most important framing choice in the project.

**Feature engineering outperformed model complexity.**
Moving from Logistic Regression to default XGBoost (same features) improved F1 from 0.674 to 0.835. Adding 13 engineered features and tuning pushed it further to 0.841. The lesson: on tabular data, better features beat a fancier model.

**Class weights instead of SMOTE.**
Oversampling the minority class before splitting risks data leakage — synthetic fraud points generated from the full dataset can bleed information into the test set. Using `scale_pos_weight` in XGBoost and `class_weight='balanced'` in scikit-learn achieves the same rebalancing effect without touching the train/test boundary.

**Random Forest was not worth the cost.**
It took 214 seconds to train — 43× longer than tuned XGBoost — and still delivered lower recall (78.0 % vs 83.7 %). For a problem that demands fast iteration and near-real-time inference, a deep ensemble of hundreds of trees is the wrong tool even when it performs decently.

**What failed: SMOTE applied before the train/test split.**
An early experiment applied SMOTE to the entire dataset before splitting. The model looked exceptional — F1 > 0.95 — but scores collapsed on a truly held-out sample. Synthetic fraud points had effectively leaked test-set signal into training. This was caught by maintaining a hold-out set never touched during any preprocessing step, and it reinforced that validation discipline matters more than algorithm choice.

---

## File Structure

```
credit-card-fraud-detection/
│
├── app/
│   └── streamlit_app.py          # Streamlit dashboard — overview, EDA, results, live scoring
│
├── data/
│   ├── creditcard.csv            # Raw dataset (not committed — download from Kaggle)
│   ├── cleaned.csv               # Output of cleaner.py
│   ├── features.csv              # Output of run_features.py
│   ├── predictions.csv           # Test-set predictions from training run
│   └── model_results.json        # Metrics for all models — read by Streamlit app
│
├── models/
│   ├── production_model.pkl      # Final deployed model (XGBoost tuned)
│   ├── xgboost.pkl               # XGBoost default params
│   ├── randomforest.pkl          # Random Forest
│   ├── lightgbm.pkl              # LightGBM
│   ├── baseline.pkl              # Logistic Regression baseline
│   ├── tuned_model.pkl           # Optuna best trial artefact
│   ├── best_params.json          # Optuna best hyperparameters
│   └── model_comparison.csv      # Cross-model metrics table
│
├── notebooks/
│   └── *.ipynb                   # Exploratory analysis notebooks
│
├── src/
│   ├── data/
│   │   ├── loader.py             # load_csv() with filename fuzzy-matching
│   │   ├── quality.py            # check_data_quality() — schema, nulls, ranges, imbalance
│   │   └── cleaner.py            # clean_data() — dedup, null handling, dtype coercion
│   │
│   ├── features/
│   │   ├── engineering.py        # create_features(), select_features()
│   │   └── run_features.py       # Orchestrates feature pipeline, saves features.csv
│   │
│   └── models/
│       ├── baseline.py           # Logistic Regression benchmark
│       ├── compare_models.py     # RF vs XGBoost vs LightGBM with cross-validation
│       ├── tuning.py             # Optuna hyperparameter search for XGBoost
│       └── run_training.py       # Full training run with MLflow logging
│
├── tests/
│   ├── conftest.py               # Adds src/ to sys.path
│   ├── test_data_quality.py      # Quality gate: pass on valid data, fail on broken data
│   ├── test_features.py          # Column count, no NaN, feature value ranges
│   └── test_model.py             # Model loads, predict_proba shape and range
│
├── .github/
│   └── workflows/
│       └── ci.yml                # GitHub Actions — test + lint on push/PR to main
│
├── Dockerfile                    # python:3.9-slim image for the Streamlit app
├── docker-compose.yml            # Starts app on port 8501, mounts data/ and models/
├── requirements.txt              # Python dependencies
├── setup.py                      # Editable install for src/ packages
└── README.md                     # This file
```

---

## CI Status

![CI](https://github.com/stuti-shrimal/credit-card-fraud-detection/actions/workflows/ci.yml/badge.svg)

---

*Built as a portfolio project demonstrating end-to-end ML engineering: data quality, feature engineering, model selection, hyperparameter tuning, experiment tracking, testing, containerisation, and deployment.*
