# src/modely.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, root_mean_squared_error


def load_and_prepare_data(filepath):
    df = pd.read_csv(filepath).dropna()
    df["year_squared"] = df["year"] ** 2
    df = pd.get_dummies(df, columns=["region", "sub-region", "country_name"], drop_first=True)
    return df


def split_data(df, target_column):
    features = ["year", "year_squared"] + [col for col in df.columns if col.startswith(("region_", "sub-region_", "country_name_"))]
    X = df[features]
    y = df[target_column]
    return train_test_split(X, y, test_size=0.2, random_state=42), X.columns


def train_ridge(X_train, y_train, alpha=1.0):
    model = Ridge(alpha=alpha)
    model.fit(X_train, y_train)
    return model


def train_random_forest(X_train, y_train, n_estimators=200):
    model = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    preds = model.predict(X_test)
    r2 = r2_score(y_test, preds)
    rmse = root_mean_squared_error(y_test, preds)
    return r2, rmse, preds
