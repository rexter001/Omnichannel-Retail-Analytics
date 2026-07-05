import pandas as pd


def feature_engineering():
    """
    Perform feature engineering on the retail dataset.
    """

    # Load datasets
    retail_df = pd.read_csv("../data/cleaned_retail.csv")
    rfm_df = pd.read_csv("../data/customer_rfm.csv")

    # Convert InvoiceDate to datetime
    retail_df["InvoiceDate"] = pd.to_datetime(retail_df["InvoiceDate"])

    # Sort values
    retail_df = retail_df.sort_values(["Customer ID", "InvoiceDate"])

    # --------------------------------------------------
    # Feature 1 : Average Time Between Orders
    # --------------------------------------------------

    retail_df["DaysBetweenOrders"] = (
        retail_df.groupby("Customer ID")["InvoiceDate"]
        .diff()
        .dt.days
    )

    avg_time = (
        retail_df.groupby("Customer ID")["DaysBetweenOrders"]
        .mean()
        .reset_index()
    )

    avg_time.rename(
        columns={"DaysBetweenOrders": "AvgTimeBetweenOrders"},
        inplace=True,
    )

    # --------------------------------------------------
    # Feature 2 : Weekend Purchase Ratio
    # --------------------------------------------------

    retail_df["DayOfWeek"] = retail_df["InvoiceDate"].dt.dayofweek

    retail_df["IsWeekend"] = (
        retail_df["DayOfWeek"] >= 5
    ).astype(int)

    weekend_ratio = (
        retail_df.groupby("Customer ID")["IsWeekend"]
        .mean()
        .reset_index()
    )

    weekend_ratio.rename(
        columns={"IsWeekend": "WeekendPurchaseRatio"},
        inplace=True,
    )

    # --------------------------------------------------
    # Feature 3 : Previous Order Amount (Lag Feature)
    # --------------------------------------------------

    retail_df["PreviousOrderAmount"] = (
        retail_df.groupby("Customer ID")["TotalAmount"]
        .shift(1)
    )

    lag_feature = (
        retail_df.groupby("Customer ID")["PreviousOrderAmount"]
        .mean()
        .reset_index()
    )

    # --------------------------------------------------
    # Merge all engineered features
    # --------------------------------------------------

    final_df = rfm_df.merge(
        avg_time,
        on="Customer ID",
        how="left",
    )

    final_df = final_df.merge(
        weekend_ratio,
        on="Customer ID",
        how="left",
    )

    final_df = final_df.merge(
        lag_feature,
        on="Customer ID",
        how="left",
    )

    # Save output
    final_df.to_csv(
        "../data/feature_engineered_data.csv",
        index=False,
    )

    print("Feature engineering completed successfully!")

    return final_df


if __name__ == "__main__":
    feature_engineering()