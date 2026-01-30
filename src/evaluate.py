import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve,
    precision_recall_fscore_support
)

from src.config import FIG_DIR


def _metrics_at_threshold(y_true, y_prob, threshold: float):
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, digits=4)
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    return y_pred, cm, report, p, r, f1


def find_best_threshold_for_f1(y_true, y_prob):
    thresholds = np.linspace(0.05, 0.95, 91)
    best_t, best_f1 = 0.5, -1.0

    for t in thresholds:
        _, _, _, _, _, f1 = _metrics_at_threshold(y_true, y_prob, t)
        if f1 > best_f1:
            best_f1 = f1
            best_t = t

    return best_t, best_f1


def plot_confusion_matrix(cm, threshold):
    plt.figure(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["No Churn", "Churn"],
        yticklabels=["No Churn", "Churn"]
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix (threshold = {threshold:.2f})")
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/confusion_matrix_threshold_{threshold:.2f}.png")
    plt.close()


def plot_feature_importance(model, feature_names, top_n=15):
    coef = model.named_steps["clf"].coef_[0]
    importance = pd.Series(coef, index=feature_names).sort_values()

    top_features = pd.concat([
        importance.head(top_n),
        importance.tail(top_n)
    ])

    plt.figure(figsize=(8, 6))
    top_features.plot(kind="barh")
    plt.title("Logistic Regression Feature Importance (Coefficients)")
    plt.xlabel("Coefficient Value")
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/logistic_feature_importance.png")
    plt.close()


def evaluate_model(model, X_test, y_test):
    os.makedirs(FIG_DIR, exist_ok=True)

    y_prob = model.predict_proba(X_test)[:, 1]

    # -------------------------
    # ROC-AUC + ROC Curve
    # -------------------------
    roc_auc = roc_auc_score(y_test, y_prob)
    print(f"ROC-AUC Score: {roc_auc:.4f}\n")

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.savefig(f"{FIG_DIR}/roc_curve.png")
    plt.close()

    # -------------------------
    # Threshold tuning
    # -------------------------
    best_t, best_f1 = find_best_threshold_for_f1(y_test, y_prob)
    print(f"Best threshold (F1): {best_t:.2f} | F1: {best_f1:.4f}\n")

    # -------------------------
    # Evaluation @ best threshold
    # -------------------------
    y_pred, cm, report, p, r, f1 = _metrics_at_threshold(
        y_test, y_prob, best_t
    )

    print("=== Evaluation @ Tuned Threshold ===")
    print("Confusion Matrix:\n", cm, "\n")
    print(report)
    print(f"Precision: {p:.4f} | Recall: {r:.4f} | F1: {f1:.4f}\n")

    # -------------------------
    # PLOTS FOR LINKEDIN
    # -------------------------
    plot_confusion_matrix(cm, best_t)
    plot_feature_importance(model, X_test.columns)

    print("📊 Model plots saved to reports/figures/")
    print("- confusion_matrix_*.png")
    print("- roc_curve.png")
    print("- logistic_feature_importance.png")

    return best_t
