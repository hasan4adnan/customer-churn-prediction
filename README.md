# 📊 Customer Churn Prediction with Logistic Regression  
**End-to-End Machine Learning Project (Script-Based, No Notebooks)**

---

## 🔍 Project Overview

Customer churn is one of the most critical business problems for subscription-based companies such as **telecommunication providers**.  
The goal of this project is to **predict whether a customer will churn (leave the company)** using historical customer data and a **Logistic Regression** model.

This project is designed as a **production-style ML pipeline**, not a notebook experiment.  
It includes:

- Comprehensive **Exploratory Data Analysis (EDA)**  
- Clean and reusable **preprocessing pipeline**  
- Multiple **model improvement iterations**  
- **Class imbalance handling & threshold tuning**  
- Clear **business-oriented evaluation**  
- **Visual artifacts** suitable for reporting and LinkedIn sharing  

---

## 🎯 Business Problem

> *“Which customers are likely to churn, so that we can take proactive actions?”*

### Why churn prediction matters:
- Acquiring new customers is **much more expensive** than retaining existing ones
- Early churn detection allows:
  - Targeted retention campaigns
  - Personalized offers
  - Reduced revenue loss

### Problem Type:
- **Binary Classification**
  - `Churn = Yes → 1`
  - `Churn = No → 0`

---

## 📁 Dataset

**Telco Customer Churn Dataset**  
- Rows: **7,043**
- Columns: **21**
- Target variable: `Churn`

### Feature Types:
- **Demographic**: gender, SeniorCitizen, Partner, Dependents  
- **Service-related**: InternetService, TechSupport, OnlineSecurity, StreamingTV, etc.  
- **Contract & billing**: Contract, PaymentMethod, MonthlyCharges, TotalCharges  
- **Tenure information**

📌 Note:  
`TotalCharges` is provided as a string in the raw dataset and required explicit conversion during preprocessing.

---

## 🗂️ Project Structure

customer_churn_project/
│
├── data/
│ └── WA_Fn-UseC_-Telco-Customer-Churn.csv
│
├── reports/
│ ├── eda_summary.md
│ └── figures/
│ ├── target_distribution.png
│ ├── churn_by_contract.png
│ ├── churn_by_paymentmethod.png
│ ├── churn_vs_tenure.png
│ ├── correlation_heatmap.png
│ ├── roc_curve.png
│ ├── confusion_matrix_threshold_0.63.png
│ └── logistic_feature_importance.png
│
├── src/
│ ├── config.py
│ ├── data_loader.py
│ ├── preprocessing.py
│ ├── train.py
│ ├── evaluate.py
│ ├── main.py
│ │
│ └── eda/
│ ├── eda_utils.py
│ ├── eda_plots.py
│ └── run_eda.py
│
├── requirements.txt
└── README.md

---

## ⚙️ Installation & Setup

```bash
pip install -r requirements.txt
```

**Required Libraries**
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn

▶️ How to Run the Project
1️⃣ Run Exploratory Data Analysis
Generates all EDA plots and a written summary report.

```bash
python -m src.eda.run_eda
```

Outputs:
- `reports/figures/` → EDA visualizations
- `reports/eda_summary.md` → Key findings

2️⃣ Train & Evaluate the Model
Runs preprocessing, training, threshold tuning, evaluation, and model visualizations.

```bash
python -m src.main
```

Outputs:
- Confusion matrix (best threshold)
- ROC curve
- Feature importance plot
- Console evaluation metrics

---

## 📊 Exploratory Data Analysis (EDA)

EDA was conducted as a decision-driven analysis, not random plotting.

**Key EDA Findings:**
- Target imbalance: ~26% churn → accuracy alone is misleading
- Tenure: Customers with shorter tenure churn significantly more
- Contract type: Month-to-month contracts show the highest churn rates
- MonthlyCharges: Higher monthly charges correlate with increased churn
- Support services: Customers with TechSupport or OnlineSecurity churn less
- Multicollinearity: TotalCharges ≈ tenure × MonthlyCharges

📌 These insights directly influenced preprocessing and metric selection.

**Key EDA Visuals:**  
<p align="center">
  <img src="reports/figures/target_distribution.png" alt="Target distribution" width="280" />
  <img src="reports/figures/churn_by_contract.png" alt="Churn by contract" width="280" />
  <img src="reports/figures/churn_by_paymentmethod.png" alt="Churn by payment method" width="280" />
</p>

<p align="center">
  <img src="reports/figures/churn_vs_tenure.png" alt="Churn vs tenure" width="420" />
  <img src="reports/figures/correlation_heatmap.png" alt="Correlation heatmap" width="420" />
</p>

---

## 🧹 Preprocessing Pipeline

**Steps:**
- Drop `customerID` (no predictive value, potential leakage)
- Convert `TotalCharges` to numeric
- Handle missing values
- Encode target variable (`Yes`/`No` → `1`/`0`)
- One-hot encode categorical features
- Train/Test split with stratification

---

## 🤖 Model Development

**Baseline Model:**
- Logistic Regression

**Iterative Improvements:**
- Feature scaling (`StandardScaler`)
- Class imbalance handling (`class_weight="balanced"`)
- Hyperparameter tuning (`GridSearchCV`)
- Threshold tuning (optimized for F1-score)
- L1 regularization (feature selection attempt)

📌 Logistic Regression was pushed to its practical limits to understand its strengths and limitations.

---

## 📈 Model Evaluation

**Final Selected Configuration:**
- Logistic Regression (scaled)
- Class-weight balanced
- Tuned decision threshold ≈ 0.63

**Confusion Matrix (Best Threshold)**
| | Predicted No | Predicted Yes |
|---|---:|---:|
| Actual No | 832 | 201 |
| Actual Yes | 112 | 262 |

**Key Metrics (Churn = 1):**
- Precision: ~0.57
- Recall: ~0.70
- F1-score: ~0.63
- ROC-AUC: ~0.835

📌 Business interpretation:
- Model successfully identifies ~70% of churners
- False positives are reduced via threshold tuning
- Balanced trade-off between recall and operational cost

---

## 📉 Model Visualizations (For Reporting & LinkedIn)

Generated automatically:
- ✅ Confusion Matrix Heatmap
- ✅ ROC Curve
- ✅ Logistic Regression Feature Importance

**Model Visuals:**  
<p align="center">
  <img src="reports/figures/confusion_matrix_threshold_0.63.png" alt="Confusion matrix" width="420" />
  <img src="reports/figures/roc_curve.png" alt="ROC curve" width="420" />
</p>

<p align="center">
  <img src="reports/figures/logistic_feature_importance.png" alt="Feature importance" width="820" />
</p>

These visuals make the model:
- Interpretable
- Presentable
- Business-friendly

---

## 🧠 Key Learnings

- Logistic Regression is a strong baseline, but has a performance ceiling
- Threshold tuning is crucial in churn problems
- ROC-AUC can be strong even when F1 is moderate
- Business context determines the “best” model behavior
- Script-based ML pipelines are closer to real-world production systems

---

## 🚀 Next Steps

Possible extensions:
- Tree-based models (XGBoost / LightGBM)
- Cost-sensitive evaluation
- Customer lifetime value (CLV) integration
- Deployment-ready API
- Monitoring & retraining strategy

---

## 🧾 Final Notes

This project intentionally avoids notebooks to demonstrate:
- Clean architecture
- Reproducibility
- Modularity

---

## 👤 Author

Hasan Adnan
Software Engineer 

Feel free to connect and discuss improvements 🚀
