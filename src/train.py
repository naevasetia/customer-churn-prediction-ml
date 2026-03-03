import os
import joblib
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.metrics import classification_report, roc_auc_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from preprocess import load_data, split_data, build_preprocessor
from sklearn.pipeline import Pipeline


def evaluate_model(name, model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print(f"\n----- {name} -----")
    print(classification_report(y_test, y_pred))
    print("ROC-AUC Score:", roc_auc_score(y_test, y_prob))

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(4, 3))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"{name} - Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()


def main():

    data_path = os.path.join("data", "raw", "bank_customer_churn_dataset.csv")

    df = load_data(data_path)
    X_train, X_test, y_train, y_test = split_data(df)

    preprocessor = build_preprocessor(X_train)

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced"
        ),

        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            random_state=42
        ),

        "XGBoost": XGBClassifier(
            eval_metric="logloss",
            use_label_encoder=False,
            random_state=42
        )
    }

    for name, classifier in models.items():

        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("classifier", classifier)
            ]
        )

        pipeline.fit(X_train, y_train)
        evaluate_model(name, pipeline, X_test, y_test)

        os.makedirs("models", exist_ok=True)
        joblib.dump(pipeline, f"models/{name.replace(' ', '_').lower()}.pkl")

        print(f"{name} model saved.\n")


if __name__ == "__main__":
    main()