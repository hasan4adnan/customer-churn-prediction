from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV


def train_model(X_train, y_train):
    """
    Final Logistic Regression attempt:
    - Scaling
    - L1 regularization (feature selection)
    - Class imbalance handling
    """

    pipeline = Pipeline(steps=[
        ("scaler", StandardScaler()),
        (
            "clf",
            LogisticRegression(
                penalty="l1",
                solver="saga",
                max_iter=7000
            )
        )
    ])

    param_grid = {
        "clf__C": [0.05, 0.1, 0.2, 0.5, 1],
        "clf__class_weight": [None, "balanced"]
    }

    search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring="f1",
        cv=5,
        n_jobs=-1
    )

    search.fit(X_train, y_train)

    print("✅ Best params:", search.best_params_)
    print(f"✅ Best CV F1: {search.best_score_:.4f}")

    # --- Optional but very informative ---
    best_model = search.best_estimator_
    coef = best_model.named_steps["clf"].coef_[0]
    n_selected = (coef != 0).sum()
    print(f"🧠 Selected features (non-zero coef): {n_selected} / {len(coef)}")

    return best_model
