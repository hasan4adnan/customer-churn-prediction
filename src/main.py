from sklearn.model_selection import train_test_split

from src.config import RANDOM_STATE
from src.data_loader import load_data
from src.preprocessing import preprocess_data
from src.train import train_model
from src.evaluate import evaluate_model


def main():
    df = load_data()
    X, y = preprocess_data(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y
    )

    model = train_model(X_train, y_train)
    best_threshold = evaluate_model(model, X_test, y_test)

    print(f"\n✅ Suggested tuned threshold for deployment (best F1): {best_threshold:.2f}")


if __name__ == "__main__":
    main()
