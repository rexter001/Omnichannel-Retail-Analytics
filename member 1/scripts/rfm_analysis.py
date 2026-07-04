"""
rfm_analysis.py
---------------
Module responsible for Customer-Level Aggregation and RFM (Recency,
Frequency, Monetary) analysis on the cleaned Online Retail II dataset.

RFM Methodology
---------------
  Recency   – How recently did the customer purchase?
              Days between the customer's last purchase and the reference date.
              Lower is better (more recent).

  Frequency – How often do they purchase?
              Count of unique invoices (transactions) per customer.
              Higher is better.

  Monetary  – How much do they spend?
              Total amount spent across all transactions per customer.
              Higher is better.

Reference Date : max(InvoiceDate) + 1 day
  Using max + 1 day is the industry-standard convention. It ensures even
  the most-recent customer has a recency of at least 1 day (avoids zero),
  and it avoids data-leakage by not using a future date.

Industry Standards:
- PEP8 compliant
- Full docstrings on every function
- Relative path resolution (portable across machines)
- Informative logging to stdout at each step

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
CLEANED_FILE = os.path.join(CLEANED_DATA_DIR, "cleaned_retail.csv")
RFM_OUTPUT_FILE = os.path.join(CLEANED_DATA_DIR, "customer_rfm.csv")


# ---------------------------------------------------------------------------
# Data loader (for cleaned CSV)
# ---------------------------------------------------------------------------

def load_cleaned_data(filepath: str = CLEANED_FILE) -> pd.DataFrame:
    """
    Load the cleaned retail CSV produced by ``clean_data.py``.

    InvoiceDate is parsed back to datetime64 on load so it can be used
    directly in RFM calculations without any further conversion.

    Parameters
    ----------
    filepath : str, optional
        Path to the cleaned CSV. Defaults to
        ``cleaned_data/cleaned_retail.csv``.

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame with InvoiceDate as datetime64[ns].

    Raises
    ------
    FileNotFoundError
        If the cleaned CSV does not exist. Run ``clean_data.py`` first.
    """
    filepath = os.path.abspath(filepath)

    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Cleaned dataset not found at: {filepath}\n"
            "Please run clean_data.py first to generate the cleaned dataset."
        )

    print(f"[INFO] Loading cleaned dataset from: {filepath}")
    df = pd.read_csv(
        filepath,
        parse_dates=["InvoiceDate"],
        dtype={"Customer ID": "int64"}
    )
    print(f"[INFO] Cleaned dataset loaded. Shape: {df.shape}")
    return df


# ---------------------------------------------------------------------------
# RFM computation
# ---------------------------------------------------------------------------

def compute_reference_date(df: pd.DataFrame) -> pd.Timestamp:
    """
    Compute the RFM reference date as max(InvoiceDate) + 1 day.

    This is the industry-standard convention:
    - Adding 1 day ensures even the most-recent purchase has Recency >= 1.
    - It avoids data leakage (we do not use a hard-coded future date).

    Parameters
    ----------
    df : pd.DataFrame
        Must contain an 'InvoiceDate' column of dtype datetime64[ns].

    Returns
    -------
    pd.Timestamp
        The reference date used for Recency calculation.
    """
    ref_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)
    print(f"[RFM]  Reference date            : {ref_date.date()}")
    print(f"[RFM]  Max InvoiceDate in data   : {df['InvoiceDate'].max().date()}")
    return ref_date


def compute_recency(df: pd.DataFrame, ref_date: pd.Timestamp) -> pd.Series:
    """
    Compute the Recency metric per customer.

    Recency = (reference_date - last_purchase_date).days

    Groups by CustomerID and finds each customer's most-recent InvoiceDate,
    then calculates the difference in whole days from the reference date.

    Parameters
    ----------
    df       : pd.DataFrame
    ref_date : pd.Timestamp

    Returns
    -------
    pd.Series
        Index = CustomerID, values = integer days since last purchase.
        Series name = 'Recency'.
    """
    recency = (
        df.groupby("Customer ID")["InvoiceDate"]
        .max()
        .apply(lambda last: (ref_date - last).days)
        .rename("Recency")
    )
    return recency


def compute_frequency(df: pd.DataFrame) -> pd.Series:
    """
    Compute the Frequency metric per customer.

    Frequency = number of unique Invoice IDs per customer.

    Using ``nunique`` on Invoice ensures that multi-line invoices (multiple
    products bought in the same session) are counted as a single visit.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.Series
        Index = CustomerID, values = unique invoice count.
        Series name = 'Frequency'.
    """
    frequency = (
        df.groupby("Customer ID")["Invoice"]
        .nunique()
        .rename("Frequency")
    )
    return frequency


def compute_monetary(df: pd.DataFrame) -> pd.Series:
    """
    Compute the Monetary metric per customer.

    Monetary = total spending = sum of TotalAmount per customer.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain 'TotalAmount' column (Quantity × UnitPrice).

    Returns
    -------
    pd.Series
        Index = CustomerID, values = total spend (float, rounded to 2 dp).
        Series name = 'Monetary'.
    """
    monetary = (
        df.groupby("Customer ID")["TotalAmount"]
        .sum()
        .round(2)
        .rename("Monetary")
    )
    return monetary


def build_rfm_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build the complete customer-level RFM table.

    Orchestrates the three metric functions and merges them into a single
    DataFrame indexed by CustomerID with columns:
      - Recency   (int)   : days since last purchase
      - Frequency (int)   : unique invoice count
      - Monetary  (float) : total spend

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned retail DataFrame (output of ``clean_data.clean_dataset()``).

    Returns
    -------
    pd.DataFrame
        Customer-level RFM table, sorted by CustomerID ascending.
    """
    print(f"\n{'='*55}")
    print(f"  Starting RFM Analysis")
    print(f"  Unique customers : {df['Customer ID'].nunique():,}")
    print(f"  Date range       : {df['InvoiceDate'].min().date()} → "
          f"{df['InvoiceDate'].max().date()}")
    print(f"{'='*55}")

    ref_date = compute_reference_date(df)

    recency_s = compute_recency(df, ref_date)
    frequency_s = compute_frequency(df)
    monetary_s = compute_monetary(df)

    # Combine all three series into one DataFrame
    rfm_df = pd.concat([recency_s, frequency_s, monetary_s], axis=1)
    rfm_df.reset_index(inplace=True)          # CustomerID becomes a column
    rfm_df.sort_values("Customer ID", inplace=True)
    rfm_df.reset_index(drop=True, inplace=True)

    print(f"\n[RFM]  RFM table built. Shape: {rfm_df.shape}")
    print(f"[RFM]  Recency   — min: {rfm_df['Recency'].min()}, "
          f"max: {rfm_df['Recency'].max()}, "
          f"mean: {rfm_df['Recency'].mean():.1f}")
    print(f"[RFM]  Frequency — min: {rfm_df['Frequency'].min()}, "
          f"max: {rfm_df['Frequency'].max()}, "
          f"mean: {rfm_df['Frequency'].mean():.1f}")
    print(f"[RFM]  Monetary  — min: {rfm_df['Monetary'].min():.2f}, "
          f"max: {rfm_df['Monetary'].max():.2f}, "
          f"mean: {rfm_df['Monetary'].mean():.2f}")

    return rfm_df


# ---------------------------------------------------------------------------
# Statistical summary
# ---------------------------------------------------------------------------

def rfm_summary(rfm_df: pd.DataFrame) -> pd.DataFrame:
    """
    Print and return descriptive statistics for the RFM table.

    Parameters
    ----------
    rfm_df : pd.DataFrame
        Output of ``build_rfm_table()``.

    Returns
    -------
    pd.DataFrame
        Descriptive statistics table (count, mean, std, min, quartiles, max).
    """
    print(f"\n{'='*55}")
    print(f"  RFM Descriptive Statistics")
    print(f"{'='*55}")
    stats = rfm_df[["Recency", "Frequency", "Monetary"]].describe().round(2)
    print(stats.to_string())
    return stats


# ---------------------------------------------------------------------------
# Save helper
# ---------------------------------------------------------------------------

def save_rfm_table(rfm_df: pd.DataFrame, output_path: str = RFM_OUTPUT_FILE) -> None:
    """
    Persist the RFM table to a CSV file.

    The output directory is created automatically if it does not exist.
    The integer index is excluded from the saved file.

    Parameters
    ----------
    rfm_df      : pd.DataFrame
        RFM table as produced by ``build_rfm_table()``.
    output_path : str, optional
        Destination file path. Defaults to
        ``cleaned_data/customer_rfm.csv`` under the project root.
    """
    output_path = os.path.abspath(output_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    rfm_df.to_csv(output_path, index=False)
    print(f"\n[SAVE]  RFM table saved → {output_path}")
    print(f"        Rows (customers): {len(rfm_df):,}  |  Columns: {rfm_df.shape[1]}")


# ---------------------------------------------------------------------------
# Entry point – run as standalone script
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    sys.path.insert(0, PROJECT_ROOT)

    # Load the cleaned dataset
    cleaned_df = load_cleaned_data()

    # Build and inspect the RFM table
    rfm_df = build_rfm_table(cleaned_df)
    rfm_summary(rfm_df)

    # Persist to disk
    save_rfm_table(rfm_df)

    print(f"\n{'='*55}")
    print("  RFM Analysis Complete")
    print(f"{'='*55}")
