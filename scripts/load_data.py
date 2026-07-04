"""
load_data.py
------------
Module responsible for loading the Online Retail II dataset.

Industry Standards:
- PEP8 compliant
- Full docstrings on every function
- Relative path resolution so the project is portable across machines
- Graceful error handling with informative messages

Author  : Data Engineering Team
Project : Retail Analytics - AI/ML Internship Mini Project
"""

import os
import sys
import pandas as pd


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

# Resolve the project root regardless of where the script is called from.
# scripts/ is one level below the project root.
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
DEFAULT_FILE = os.path.join(DATA_DIR, "online_retail_II.csv")


# ---------------------------------------------------------------------------
# Core loader
# ---------------------------------------------------------------------------

def load_dataset(filepath: str = DEFAULT_FILE, encoding: str = "ISO-8859-1") -> pd.DataFrame:
    """
    Load the Online Retail II CSV dataset into a pandas DataFrame.

    The Online Retail II file uses ISO-8859-1 (Latin-1) encoding, which is
    required to correctly decode special characters present in the Description
    column (e.g., accented characters in product names).

    Parameters
    ----------
    filepath : str, optional
        Absolute or relative path to the CSV file.
        Defaults to ``data/online_retail_II.csv`` under the project root.
    encoding : str, optional
        File encoding. Defaults to ``ISO-8859-1``.

    Returns
    -------
    pd.DataFrame
        Raw, unprocessed DataFrame as loaded from disk.

    Raises
    ------
    FileNotFoundError
        If the CSV file does not exist at the given path.
    ValueError
        If the loaded DataFrame is completely empty.
    """
    filepath = os.path.abspath(filepath)

    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Dataset not found at: {filepath}\n"
            "Please place 'online_retail_II.csv' inside the 'data/' directory."
        )

    print(f"[INFO] Loading dataset from: {filepath}")

    try:
        df = pd.read_csv(filepath, encoding=encoding)
    except UnicodeDecodeError:
        # Fallback to utf-8 if Latin-1 fails (some exports use utf-8-sig)
        print("[WARN] ISO-8859-1 decode failed. Retrying with utf-8-sig …")
        df = pd.read_csv(filepath, encoding="utf-8-sig")

    if df.empty:
        raise ValueError("The loaded DataFrame is empty. Please check the source file.")

    print(f"[INFO] Dataset loaded successfully. Shape: {df.shape}")
    return df


# ---------------------------------------------------------------------------
# Overview / EDA helpers
# ---------------------------------------------------------------------------

def show_shape(df: pd.DataFrame) -> None:
    """Print the number of rows and columns in the DataFrame."""
    rows, cols = df.shape
    print(f"\n{'='*50}")
    print(f"  Dataset Shape")
    print(f"{'='*50}")
    print(f"  Rows    : {rows:,}")
    print(f"  Columns : {cols}")


def show_column_names(df: pd.DataFrame) -> None:
    """Print all column names with their index positions."""
    print(f"\n{'='*50}")
    print(f"  Column Names")
    print(f"{'='*50}")
    for idx, col in enumerate(df.columns):
        print(f"  [{idx}] {col}")


def show_dtypes(df: pd.DataFrame) -> None:
    """Print the data type of every column."""
    print(f"\n{'='*50}")
    print(f"  Data Types")
    print(f"{'='*50}")
    print(df.dtypes.to_string())


def show_first_rows(df: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    """
    Display the first *n* rows of the DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
    n  : int, optional
        Number of rows to display. Defaults to 5.

    Returns
    -------
    pd.DataFrame
        Slice containing the first *n* rows.
    """
    print(f"\n{'='*50}")
    print(f"  First {n} Rows")
    print(f"{'='*50}")
    subset = df.head(n)
    print(subset.to_string())
    return subset


def show_descriptive_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Display descriptive statistics for all numeric columns.

    Returns
    -------
    pd.DataFrame
        Summary statistics (count, mean, std, min, quartiles, max).
    """
    print(f"\n{'='*50}")
    print(f"  Descriptive Statistics")
    print(f"{'='*50}")
    stats = df.describe()
    print(stats.to_string())
    return stats


def show_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute and display a missing-value summary table.

    Shows the count and percentage of nulls for every column that has at
    least one missing value, sorted by percentage descending.

    Returns
    -------
    pd.DataFrame
        Table with columns ['Missing Count', 'Missing %'] for columns
        that contain nulls.
    """
    print(f"\n{'='*50}")
    print(f"  Missing Value Summary")
    print(f"{'='*50}")

    missing_count = df.isnull().sum()
    missing_pct = (missing_count / len(df)) * 100

    missing_df = pd.DataFrame({
        "Missing Count": missing_count,
        "Missing %": missing_pct.round(2)
    })

    # Only show columns that actually have missing values
    missing_df = missing_df[missing_df["Missing Count"] > 0]
    missing_df = missing_df.sort_values("Missing %", ascending=False)

    if missing_df.empty:
        print("  No missing values found.")
    else:
        print(missing_df.to_string())

    return missing_df


def dataset_overview(df: pd.DataFrame) -> None:
    """
    Run a full exploratory overview of the raw dataset.

    Calls: show_shape, show_column_names, show_dtypes,
           show_first_rows, show_descriptive_stats, show_missing_values.

    Parameters
    ----------
    df : pd.DataFrame
        Raw DataFrame loaded from disk.
    """
    show_shape(df)
    show_column_names(df)
    show_dtypes(df)
    show_first_rows(df)
    show_descriptive_stats(df)
    show_missing_values(df)


# ---------------------------------------------------------------------------
# Entry point – run as standalone script
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    raw_df = load_dataset()
    dataset_overview(raw_df)
