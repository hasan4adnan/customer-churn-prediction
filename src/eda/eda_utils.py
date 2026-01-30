import pandas as pd
import numpy as np


def data_overview(df: pd.DataFrame) -> pd.DataFrame:
    """
    Basic overview of dataset structure.
    """
    overview = pd.DataFrame({
        "dtype": df.dtypes,
        "missing_count": df.isnull().sum(),
        "missing_ratio": df.isnull().mean(),
        "n_unique": df.nunique()
    })

    return overview


def numeric_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Summary statistics for numeric features.
    """
    numeric_df = df.select_dtypes(include=[np.number])
    summary = numeric_df.describe().T
    return summary


def missing_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Missing value summary (count + ratio).
    """
    missing = df.isnull().sum()
    missing_ratio = df.isnull().mean()

    summary = pd.DataFrame({
        "missing_count": missing,
        "missing_ratio": missing_ratio
    })

    return summary.sort_values(by="missing_count", ascending=False)


def churn_rate_by_categorical(
    df: pd.DataFrame,
    categorical_col: str,
    target_col: str = "Churn"
) -> pd.DataFrame:
    """
    Calculates churn rate for a categorical feature.
    """
    temp = df.copy()
    temp[target_col] = temp[target_col].map({"Yes": 1, "No": 0})

    churn_rate = (
        temp
        .groupby(categorical_col)[target_col]
        .mean()
        .sort_values(ascending=False)
    )

    return churn_rate.to_frame(name="churn_rate")


def group_stats_by_churn(
    df: pd.DataFrame,
    numeric_col: str,
    target_col: str = "Churn"
) -> pd.DataFrame:
    """
    Group statistics (mean, median) of numeric feature by churn.
    """
    temp = df.copy()
    temp[target_col] = temp[target_col].map({"Yes": 1, "No": 0})

    stats = temp.groupby(target_col)[numeric_col].agg(
        mean="mean",
        median="median",
        count="count"
    )

    return stats
