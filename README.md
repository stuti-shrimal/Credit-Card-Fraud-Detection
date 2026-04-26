# Credit Card Fraud Detection

Binary classification project to detect fraudulent credit card transactions.

## Exploratory Data Analysis

**Dataset:** 284,807 transactions × 31 columns — 28 anonymised PCA features (V1–V28), plus `Time`, `Amount`, and `Class` (target). No missing values.

**Key findings:**

- **Severe class imbalance:** Only 0.17 % of transactions are fraud (492 of 284,807). A classifier trained on raw counts will default to predicting legitimate — resampling (SMOTE) or `class_weight='balanced'` is required.
- **V-features are pre-scaled:** V1–V28 are already zero-centred with unit variance from PCA. Standard scaling is redundant for these columns.
- **Amount is right-skewed:** A long tail of large transactions makes `Amount` unsuitable for distance-based models without log or RobustScaler transformation.
- **Top discriminative features:** V14, V17, V12, and V10 show the largest median shift between fraud and legitimate classes — strong candidates for feature importance and rule-based pre-filters.

**Modeling implications:** Use `log1p(Amount)` before scaling; drop or alias one of any highly correlated V-feature pairs; evaluate on F1/AUC-PR rather than accuracy given the imbalance.
