import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.loader import load_csv
from features.engineering import create_features, select_features

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
OUTPUT_FILE = "features.csv"


def _resolve_input() -> str:
    # Prefer cleaned, then fall back to any creditcard*.csv variant (e.g. "creditcard 2.csv")
    for name in ("cleaned.csv", "creditcard.csv"):
        if (DATA_DIR / name).exists():
            return name
    candidates = sorted(DATA_DIR.glob("creditcard*.csv"))
    if candidates:
        return candidates[0].name
    raise FileNotFoundError(
        f"No input CSV found in {DATA_DIR}. "
        "Add creditcard.csv (raw) or cleaned.csv (pre-processed)."
    )


def main() -> None:
    t0 = time.time()

    # ── Load ──────────────────────────────────────────────────────────────────
    input_file = _resolve_input()
    print(f"Loading '{input_file}' ...")
    df_raw = load_csv(input_file)
    print(f"  Raw shape       : {df_raw.shape[0]:,} rows × {df_raw.shape[1]} columns")
    t1 = time.time()
    print(f"  Load time       : {t1 - t0:.2f}s\n")

    # ── Engineer features ─────────────────────────────────────────────────────
    print("Running create_features() ...")
    df_eng = create_features(df_raw)
    new_cols = [c for c in df_eng.columns if c not in df_raw.columns]
    print(f"  Engineered shape: {df_eng.shape[0]:,} rows × {df_eng.shape[1]} columns")
    print(f"  New features    : {len(new_cols)}")
    t2 = time.time()
    print(f"  Feature time    : {t2 - t1:.2f}s\n")

    # ── Select features ───────────────────────────────────────────────────────
    print("Running select_features() ...")
    selected, df_sel = select_features(df_eng)
    t3 = time.time()
    print(f"  Selected shape  : {df_sel.shape[0]:,} rows × {df_sel.shape[1]} columns")
    print(f"  Selection time  : {t3 - t2:.2f}s\n")

    # ── Save ──────────────────────────────────────────────────────────────────
    out_path = DATA_DIR / OUTPUT_FILE
    print(f"Saving to '{out_path}' ...")
    df_sel.to_csv(out_path, index=False)
    size_mb = out_path.stat().st_size / 1_048_576
    t4 = time.time()
    print(f"  File size       : {size_mb:.1f} MB")
    print(f"  Save time       : {t4 - t3:.2f}s\n")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("─" * 60)
    print(f"Before : {df_raw.shape[0]:,} rows × {df_raw.shape[1]:>2} columns")
    print(f"After  : {df_sel.shape[0]:,} rows × {df_sel.shape[1]:>2} columns")
    print(f"\nKept features ({len(selected)}):")
    for col in selected:
        print(f"  {col}")
    print(f"\nTotal elapsed : {t4 - t0:.2f}s")
    print("─" * 60)


if __name__ == "__main__":
    main()
