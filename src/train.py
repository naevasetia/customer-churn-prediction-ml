import os
import joblib
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score

from preprocess import load_data, split_data, build_preprocessor
from sklearn.pipeline import Pipeline


from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


def main():

    # Path to dataset
    data_path = os.path.join("data", "raw", "bank_customer_churn_dataset.csv")

    # Load and split data
    df = load_data(data_path)
    X_train, X_test, y_train, y_test = split_data(df)

    # Build preprocessing pipeline
    preprocessor = build_preprocessor(X_train)

    # Create full pipeline (preprocessing + model)
    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced"))
        ]
    )

    # Train model
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # Evaluation
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

    print("ROC-AUC Score:", roc_auc_score(y_test, y_prob))

    # Save trained model
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/logistic_model.pkl")

    print("Model saved successfully.")


if __name__ == "__main__":
    main()