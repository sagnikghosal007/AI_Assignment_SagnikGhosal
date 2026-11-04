"""
preprocessing.py  (Task 1 - Data Preparation)
---------------------------------------------
Reads data/tickets.csv, inspects it, cleans it, and saves data/tickets_clean.csv.

Run:  python src/preprocessing.py
The functions here (clean_text, load_data, prepare_data) are also imported by
train.py and predict.py, so the SAME cleaning is used for training and prediction.
"""

import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW_PATH = ROOT / "data" / "tickets.csv"
CLEAN_PATH = ROOT / "data" / "tickets_clean.csv"


def clean_text(text: str) -> str:
    """Basic text cleaning for one ticket description."""
    text = str(text).lower()                    # 1. lowercase:  "Login" -> "login"
    text = re.sub(r"[^a-z0-9\s]", " ", text)    # 2. remove punctuation/special characters
    text = re.sub(r"\s+", " ", text).strip()    # 3. collapse extra spaces, trim the ends
    return text


def load_data(path: Path = RAW_PATH) -> pd.DataFrame:
    """Read the CSV file into a DataFrame."""
    return pd.read_csv(path)


def prepare_data(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Check and fix duplicates / missing values, then clean the text."""
    if verbose:
        print("\n--- Missing values (before cleaning) ---")
        print(df.isnull().sum())
        print(f"\nDuplicate rows (before cleaning): {df.duplicated().sum()}")

    # Duplicates: keep the first copy, drop the rest
    df = df.drop_duplicates().copy()

    # Missing description or category: the row is useless for learning -> drop it
    df = df.dropna(subset=["ticket_description", "category"]).copy()

    # Missing priority/status: not used by the model -> fill with "Unknown"
    df["priority"] = df["priority"].fillna("Unknown")
    df["status"] = df["status"].fillna("Unknown")

    # Text cleaning -> new column (the original text is kept for reference)
    df["clean_description"] = df["ticket_description"].apply(clean_text)

    # After cleaning, two different sentences may become identical; remove those too
    df = df.drop_duplicates(subset=["clean_description", "category"]).copy()

    return df.reset_index(drop=True)


if __name__ == "__main__":
    pd.set_option("display.max_colwidth", 70)
    pd.set_option("display.width", 150)

    raw = load_data()

    print("=" * 60)
    print("RAW DATASET")
    print("=" * 60)
    print(raw.head(10))
    print(f"\nShape (rows, columns): {raw.shape}")
    print("\n--- Dataset info ---")
    raw.info()

    clean = prepare_data(raw)

    print("\n" + "=" * 60)
    print("AFTER CLEANING")
    print("=" * 60)
    print(f"Shape (rows, columns): {clean.shape}")
    print(f"Missing values left: {int(clean.isnull().sum().sum())}")
    print(f"Duplicates left:     {int(clean.duplicated().sum())}")
    print("\nOriginal vs cleaned text (first 5 rows):")
    print(clean[["ticket_description", "clean_description"]].head(5))

    clean.to_csv(CLEAN_PATH, index=False)
    print(f"\nSaved cleaned data to {CLEAN_PATH}")
