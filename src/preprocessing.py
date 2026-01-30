import pandas as pd
from src.config import TARGET_COL, ID_COL


def preprocess_data(df: pd.DataFrame):
    """
    Preprocess the Telco Customer Churn dataset for modeling.

    Steps:
    - Drop ID column
    - Convert TotalCharges to numeric
    - Drop rows with missing values
    - Encode target variable (Yes/No -> 1/0)
    - One-hot encode categorical features
    - Split X and y
    """

    data = df.copy()

    # -------------------------
    # Drop ID column
    # -------------------------
    if ID_COL in data.columns:
        data.drop(columns=[ID_COL], inplace=True)

    # -------------------------
    # TotalCharges to numeric
    # -------------------------
    data["TotalCharges"] = pd.to_numeric(
        data["TotalCharges"], errors="coerce"
    )

    # -------------------------
    # Drop missing values
    # -------------------------
    data.dropna(inplace=True)

    # -------------------------
    # Encode target
    # -------------------------
    data[TARGET_COL] = data[TARGET_COL].map({"Yes": 1, "No": 0})

    # -------------------------
    # Feature / Target split
    # -------------------------
    X = data.drop(columns=[TARGET_COL])
    y = data[TARGET_COL]

    # -------------------------
    # One-hot encoding
    # -------------------------
    X = pd.get_dummies(X, drop_first=True)

    return X, y
