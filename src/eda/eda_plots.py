import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from src.config import FIG_DIR


# -------------------------
# Helper
# -------------------------
def _ensure_fig_dir():
    os.makedirs(FIG_DIR, exist_ok=True)


# -------------------------
# Target Distribution
# -------------------------
def plot_target_distribution(df: pd.DataFrame, target_col: str = "Churn"):
    _ensure_fig_dir()

    plt.figure()
    sns.countplot(x=target_col, data=df)
    plt.title("Target Distribution (Churn)")
    plt.xlabel("Churn")
    plt.ylabel("Count")

    plt.savefig(f"{FIG_DIR}/target_distribution.png", bbox_inches="tight")
    plt.close()


# -------------------------
# Missing Values
# -------------------------
def plot_missing_values(df: pd.DataFrame):
    _ensure_fig_dir()

    missing_ratio = df.isnull().mean()
    missing_ratio = missing_ratio[missing_ratio > 0]

    if missing_ratio.empty:
        return

    plt.figure(figsize=(8, 4))
    missing_ratio.sort_values(ascending=False).plot(kind="bar")
    plt.title("Missing Value Ratio by Feature")
    plt.ylabel("Missing Ratio")

    plt.savefig(f"{FIG_DIR}/missing_values.png", bbox_inches="tight")
    plt.close()


# -------------------------
# Numeric Distributions
# -------------------------
def plot_numeric_distributions(df: pd.DataFrame):
    _ensure_fig_dir()

    numeric_cols = df.select_dtypes(include=[np.number]).columns

    for col in numeric_cols:
        plt.figure()
        sns.histplot(df[col], kde=True)
        plt.title(f"Distribution of {col}")

        plt.savefig(f"{FIG_DIR}/numeric_distribution_{col}.png", bbox_inches="tight")
        plt.close()


# -------------------------
# Numeric vs Churn
# -------------------------
def plot_numeric_vs_churn(
    df: pd.DataFrame,
    numeric_col: str,
    target_col: str = "Churn"
):
    _ensure_fig_dir()

    plt.figure()
    sns.boxplot(x=target_col, y=numeric_col, data=df)
    plt.title(f"{numeric_col} vs Churn")

    plt.savefig(
        f"{FIG_DIR}/churn_vs_{numeric_col.lower()}.png",
        bbox_inches="tight"
    )
    plt.close()


# -------------------------
# Categorical Churn Rate
# -------------------------
def plot_churn_rate_by_categorical(
    df: pd.DataFrame,
    categorical_col: str,
    target_col: str = "Churn"
):
    _ensure_fig_dir()

    temp = df.copy()
    temp[target_col] = temp[target_col].map({"Yes": 1, "No": 0})

    churn_rate = (
        temp
        .groupby(categorical_col)[target_col]
        .mean()
        .sort_values(ascending=False)
    )

    plt.figure(figsize=(8, 4))
    churn_rate.plot(kind="bar")
    plt.title(f"Churn Rate by {categorical_col}")
    plt.ylabel("Churn Rate")

    plt.savefig(
        f"{FIG_DIR}/churn_by_{categorical_col.lower()}.png",
        bbox_inches="tight"
    )
    plt.close()


# -------------------------
# Correlation Heatmap
# -------------------------
def plot_correlation_heatmap(df: pd.DataFrame):
    _ensure_fig_dir()

    numeric_df = df.select_dtypes(include=[np.number])

    if numeric_df.shape[1] < 2:
        return

    corr = numeric_df.corr()

    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Correlation Heatmap (Numeric Features)")

    plt.savefig(f"{FIG_DIR}/correlation_heatmap.png", bbox_inches="tight")
    plt.close()
