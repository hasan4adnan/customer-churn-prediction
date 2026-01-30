# 📊 Exploratory Data Analysis Summary

## Dataset Overview
- Rows: 7043
- Columns: 21

## Target Distribution
- No: 73.46%
- Yes: 26.54%

## Data Quality Notes
- Total missing values: 11
- `TotalCharges` converted to numeric (string → NaN where invalid)

## Key Observations (Initial)
- Churn is moderately imbalanced (~26%) → accuracy alone is not sufficient
- Customers with low tenure show higher churn risk
- Month-to-month contracts have higher churn rates
- Higher MonthlyCharges are associated with higher churn
- Some services (TechSupport, OnlineSecurity) reduce churn likelihood

## Modeling Implications
- Prefer Recall / F1-score over Accuracy
- One-hot encoding required for categorical features
- Watch multicollinearity: TotalCharges ~ tenure × MonthlyCharges
