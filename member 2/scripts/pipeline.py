from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


def create_pipeline():
    """
    Create preprocessing pipeline for feature engineered data.
    """

    numeric_features = [
        "Recency",
        "Frequency",
        "Monetary",
        "AvgTimeBetweenOrders",
        "WeekendPurchaseRatio",
        "PreviousOrderAmount",
    ]

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
        ]
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
        ]
    )

    print("Pipeline created successfully!")

    return pipeline


if __name__ == "__main__":
    create_pipeline()