import pandas as pd
from src.config import DATA_PATH, TARGET_COL


def load_data() -> pd.DataFrame:
    """
    Load Telco Customer Churn dataset from CSV.

    Returns
    -------
    pd.DataFrame
        Raw dataset
    """
    df = pd.read_csv(DATA_PATH)
    return df


def basic_data_check(df: pd.DataFrame) -> None:
    """
    Perform basic sanity checks on the dataset.
    """

    print("=== DATA OVERVIEW ===")
    print(f"Shape: {df.shape}")
    print()

    print("=== COLUMNS ===")
    print(df.columns.tolist())
    print()

    if TARGET_COL not in df.columns:
        raise ValueError(
            f"Target column '{TARGET_COL}' not found in dataset!"
        )

    print("=== TARGET DISTRIBUTION ===")
    print(df[TARGET_COL].value_counts())
    print()
    print("=== TARGET RATIO ===")
    print(df[TARGET_COL].value_counts(normalize=True))
