import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


def load_data(path):
    # load csv
    df = pd.read_csv(path)
    df = df.drop(columns=["customer_id"])
    return df


def split_data(df):
    # proper stratified split
    X = df.drop("churn", axis=1)
    y = df["churn"]

    return train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )


def build_preprocessor(X):
    numerical_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()

    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(drop="first", handle_unknown="ignore")

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numerical_cols),
            ("cat", categorical_transformer, categorical_cols),
        ]
    )

    return preprocessor