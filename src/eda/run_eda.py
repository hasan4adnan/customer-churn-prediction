import os
import pandas as pd

from src.config import REPORT_DIR, FIG_DIR, TARGET_COL
from src.data_loader import load_data
from src.eda.eda_utils import (
    data_overview,
    numeric_summary,
    missing_summary,
    churn_rate_by_categorical,
    group_stats_by_churn
)
from src.eda.eda_plots import (
    plot_target_distribution,
    plot_missing_values,
    plot_numeric_distributions,
    plot_numeric_vs_churn,
    plot_churn_rate_by_categorical,
    plot_correlation_heatmap
)


def main():
    # -------------------------
    # Setup
    # -------------------------
    os.makedirs(REPORT_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)

    # -------------------------
    # Load Data
    # -------------------------
    df = load_data()

    # -------------------------
    # Light Cleaning (EDA only)
    # -------------------------
    # TotalCharges is string -> numeric
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # -------------------------
    # EDA Tables
    # -------------------------
    overview_df = data_overview(df)
    numeric_summary_df = numeric_summary(df)
    missing_df = missing_summary(df)

    # -------------------------
    # Plots
    # -------------------------
    plot_target_distribution(df)
    plot_missing_values(df)
    plot_numeric_distributions(df)

    # Numeric vs churn
    for col in ["tenure", "MonthlyCharges", "TotalCharges"]:
        if col in df.columns:
            plot_numeric_vs_churn(df, col)

    # Categorical churn rate
    categorical_cols = [
        "Contract",
        "InternetService",
        "PaymentMethod",
        "PaperlessBilling",
        "TechSupport",
        "OnlineSecurity"
    ]

    for col in categorical_cols:
        if col in df.columns:
            plot_churn_rate_by_categorical(df, col)

    # Correlation
    plot_correlation_heatmap(df)

    # -------------------------
    # Write EDA Summary
    # -------------------------
    summary_path = os.path.join(REPORT_DIR, "eda_summary.md")

    with open(summary_path, "w") as f:
        f.write("# 📊 Exploratory Data Analysis Summary\n\n")

        f.write("## Dataset Overview\n")
        f.write(f"- Rows: {df.shape[0]}\n")
        f.write(f"- Columns: {df.shape[1]}\n\n")

        f.write("## Target Distribution\n")
        churn_ratio = df[TARGET_COL].value_counts(normalize=True)
        for k, v in churn_ratio.items():
            f.write(f"- {k}: {v:.2%}\n")
        f.write("\n")

        f.write("## Data Quality Notes\n")
        total_missing = missing_df["missing_count"].sum()
        f.write(f"- Total missing values: {int(total_missing)}\n")
        f.write("- `TotalCharges` converted to numeric (string → NaN where invalid)\n\n")

        f.write("## Key Observations (Initial)\n")
        f.write("- Churn is moderately imbalanced (~26%) → accuracy alone is not sufficient\n")
        f.write("- Customers with low tenure show higher churn risk\n")
        f.write("- Month-to-month contracts have higher churn rates\n")
        f.write("- Higher MonthlyCharges are associated with higher churn\n")
        f.write("- Some services (TechSupport, OnlineSecurity) reduce churn likelihood\n\n")

        f.write("## Modeling Implications\n")
        f.write("- Prefer Recall / F1-score over Accuracy\n")
        f.write("- One-hot encoding required for categorical features\n")
        f.write("- Watch multicollinearity: TotalCharges ~ tenure × MonthlyCharges\n")

    print("✅ EDA completed successfully.")
    print(f"📁 Figures saved to: {FIG_DIR}")
    print(f"🧾 Summary written to: {summary_path}")


if __name__ == "__main__":
    main()
