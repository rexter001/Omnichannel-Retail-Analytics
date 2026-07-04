"""
clean_data.py
-------------
Module responsible for all data-cleaning and feature-engineering steps
applied to the Online Retail II dataset.

Cleaning pipeline (in order):
  1. Drop fully-duplicate rows
  2. Remove cancelled invoices  (Invoice starting with 'C')
  3. Drop rows with missing CustomerID
  4. Drop rows with missing Description
  5. Remove rows with Quantity  <= 0
  6. Remove rows with UnitPrice <= 0
  7. Parse InvoiceDate to datetime
  8. Feature engineering: TotalAmount = Quantity × UnitPrice
  9. Final type optimisation & reset index
 10. Save cleaned dataset to cleaned_data/cleaned_retail.csv

Industry Standards:
- PEP8 compliant
- Full docstrings on every function
- Relative path resolution (portable across machines)
- Informative logging to stdout at each step
- Returns a clean copy; original DataFrame is never mutated

Author  : Data Engineering Team
Project : Retail Analytics - AI/ML Internship Mini Project
"""

import os
import sys
import pandas as pd

# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CLEANED_DATA_DIR = os.path.join(PROJECT_ROOT, "cleaned_data")
OUTPUT_FILE = os.path.join(CLEANED_DATA_DIR, "cleaned_retail.csv")


# ---------------------------------------------------------------------------
# Individual cleaning steps
# ---------------------------------------------------------------------------

def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove fully-duplicate rows from the DataFrame.

    A row is considered a duplicate only if every column value matches
    an earlier row exactly.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
        DataFrame with duplicate rows removed.
    """
    before = len(df)
    df = df.drop_duplicates()
    removed = before - len(df)
    print(f"[CLEAN] Duplicates removed      : {removed:,}  (remaining: {len(df):,})")
    return df


def remove_cancelled_invoices(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop cancelled invoices.

    In the Online Retail II dataset, any Invoice number that begins with
    the letter 'C' represents a cancellation / credit note and should be
    excluded from sales analysis.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
        DataFrame with cancellation rows removed.
    """
    before = len(df)
    # Cast to string first to handle any mixed-type invoice columns safely
    cancelled_mask = df["Invoice"].astype(str).str.startswith("C")
    df = df[~cancelled_mask].copy()
    removed = before - len(df)
    print(f"[CLEAN] Cancelled invoices removed : {removed:,}  (remaining: {len(df):,})")
    return df


def remove_missing_customer_id(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop rows where CustomerID is null.

    Transactions without a CustomerID cannot be attributed to any customer
    and are therefore unusable for RFM analysis.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
        DataFrame with missing CustomerID rows removed.
    """
    before = len(df)
    df = df.dropna(subset=["Customer ID"]).copy()
    removed = before - len(df)
    print(f"[CLEAN] Missing CustomerID removed : {removed:,}  (remaining: {len(df):,})")
    return df


def remove_missing_description(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop rows where Description is null.

    While not strictly required for RFM, retaining rows without a product
    description adds no analytical value.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
        DataFrame with missing Description rows removed.
    """
    before = len(df)
    df = df.dropna(subset=["Description"]).copy()
    removed = before - len(df)
    print(f"[CLEAN] Missing Description removed: {removed:,}  (remaining: {len(df):,})")
    return df


def remove_negative_quantity(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove rows where Quantity is zero or negative.

    Zero or negative quantities indicate returns or data errors that would
    distort monetary calculations.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
        DataFrame containing only rows with positive Quantity.
    """
    before = len(df)
    df = df[df["Quantity"] > 0].copy()
    removed = before - len(df)
    print(f"[CLEAN] Non-positive Quantity removed: {removed:,}  (remaining: {len(df):,})")
    return df


def remove_negative_unit_price(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove rows where UnitPrice is zero or negative.

    A price of zero or below is invalid for revenue calculations and likely
    represents data-entry errors or test records.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
        DataFrame containing only rows with positive UnitPrice.
    """
    before = len(df)
    df = df[df["Price"] > 0].copy()
    removed = before - len(df)
    print(f"[CLEAN] Non-positive UnitPrice removed: {removed:,}  (remaining: {len(df):,})")
    return df


def parse_invoice_date(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert the InvoiceDate column from object/string to datetime64.

    Uses ``infer_datetime_format=True`` for performance, with ``errors='coerce'``
    so any unparseable strings become NaT rather than raising an exception.
    Any NaT rows introduced by coercion are then dropped.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
        DataFrame with InvoiceDate as datetime64[ns].
    """
    df = df.copy()
    before = len(df)
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], infer_datetime_format=True, errors="coerce")
    nat_count = df["InvoiceDate"].isna().sum()
    if nat_count > 0:
        df = df.dropna(subset=["InvoiceDate"])
        print(f"[CLEAN] Unparseable InvoiceDate rows dropped: {nat_count:,}")
    removed = before - len(df)
    print(f"[CLEAN] InvoiceDate parsed to datetime  (rows after: {len(df):,})")
    return df


def engineer_total_amount(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create the TotalAmount feature.

    TotalAmount = Quantity × UnitPrice

    This derived column represents the revenue contribution of each
    line item and is the primary input for Monetary value in RFM.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain 'Quantity' and 'Price' columns with positive values.

    Returns
    -------
    pd.DataFrame
        DataFrame with a new 'TotalAmount' column (float64).
    """
    df = df.copy()
    df["TotalAmount"] = df["Quantity"] * df["Price"]
    print(f"[FEAT]  TotalAmount column created  "
          f"(min={df['TotalAmount'].min():.2f}, "
          f"max={df['TotalAmount'].max():.2f}, "
          f"mean={df['TotalAmount'].mean():.2f})")
    return df


def optimise_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Optimise column data types to reduce memory usage.

    - CustomerID  → int64  (safe after dropping nulls)
    - StockCode   → str    (may contain alphanumeric codes)
    - Invoice     → str

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
        DataFrame with optimised data types.
    """
    df = df.copy()
    df["Customer ID"] = df["Customer ID"].astype("int64")
    df["Invoice"] = df["Invoice"].astype(str)
    df["StockCode"] = df["StockCode"].astype(str)
    print(f"[OPT]   Data types optimised.")
    return df


# ---------------------------------------------------------------------------
# Pipeline orchestrator
# ---------------------------------------------------------------------------

def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Execute the full data-cleaning and feature-engineering pipeline.

    Steps applied in order:
      1. remove_duplicates
      2. remove_cancelled_invoices
      3. remove_missing_customer_id
      4. remove_missing_description
      5. remove_negative_quantity
      6. remove_negative_unit_price
      7. parse_invoice_date
      8. engineer_total_amount
      9. optimise_dtypes
     10. Reset DataFrame index

    Parameters
    ----------
    df : pd.DataFrame
        Raw DataFrame as returned by ``load_data.load_dataset()``.

    Returns
    -------
    pd.DataFrame
        Fully cleaned and feature-engineered DataFrame ready for analysis.
    """
    print(f"\n{'='*55}")
    print(f"  Starting Data Cleaning Pipeline")
    print(f"  Input shape : {df.shape}")
    print(f"{'='*55}")

    df = remove_duplicates(df)
    df = remove_cancelled_invoices(df)
    df = remove_missing_customer_id(df)
    df = remove_missing_description(df)
    df = remove_negative_quantity(df)
    df = remove_negative_unit_price(df)
    df = parse_invoice_date(df)
    df = engineer_total_amount(df)
    df = optimise_dtypes(df)

    # Reset index so it is clean and sequential
    df = df.reset_index(drop=True)

    print(f"\n{'='*55}")
    print(f"  Cleaning Pipeline Complete")
    print(f"  Output shape : {df.shape}")
    print(f"{'='*55}\n")

    return df


# ---------------------------------------------------------------------------
# Save helper
# ---------------------------------------------------------------------------

def save_cleaned_data(df: pd.DataFrame, output_path: str = OUTPUT_FILE) -> None:
    """
    Persist the cleaned DataFrame to a CSV file.

    The output directory is created automatically if it does not exist.
    The index is excluded from the saved file.

    Parameters
    ----------
    df          : pd.DataFrame
        Cleaned DataFrame to save.
    output_path : str, optional
        Destination file path. Defaults to
        ``cleaned_data/cleaned_retail.csv`` under the project root.
    """
    output_path = os.path.abspath(output_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"[SAVE]  Cleaned data saved → {output_path}")
    print(f"        Rows: {len(df):,}  |  Columns: {df.shape[1]}")


# ---------------------------------------------------------------------------
# Entry point – run as standalone script
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Import sibling module using sys.path manipulation for script-level runs
    sys.path.insert(0, PROJECT_ROOT)
    from scripts.load_data import load_dataset

    raw_df = load_dataset()
    cleaned_df = clean_dataset(raw_df)
    save_cleaned_data(cleaned_df)
