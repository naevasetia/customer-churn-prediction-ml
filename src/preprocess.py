import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


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
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )


def scale_features(X_train, X_test, numerical_cols):
    # scales numerical features only
    scaler = StandardScaler()

    X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
    X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])

    return X_train, X_test, scaler