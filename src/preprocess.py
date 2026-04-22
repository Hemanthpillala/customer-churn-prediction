"""
Data preprocessing pipeline for Customer Churn Prediction.
Handles missing values, encoding, scaling, and class imbalance (SMOTE).
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import joblib
import os


NUMERIC_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges"]
CATEGORICAL_FEATURES = [
    "gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
    "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
    "PaperlessBilling", "PaymentMethod",
]


def load_and_clean(data_path: str) -> tuple:
    df = pd.read_csv(data_path)

    # Fix TotalCharges (sometimes stored as string with spaces)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df.dropna(inplace=True)

    # Drop customer ID
    if "customerID" in df.columns:
        df.drop("customerID", axis=1, inplace=True)

    # Binary encode target
    df["Churn"] = (df["Churn"] == "Yes").astype(int)

    X = df.drop("Churn", axis=1)
    y = df["Churn"]
    return X, y


def build_preprocessor():
    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([
        ("num", numeric_transformer, NUMERIC_FEATURES),
        ("cat", categorical_transformer, CATEGORICAL_FEATURES),
    ])


def get_pipeline(estimator):
    """Returns an imbalanced-learn pipeline with SMOTE + preprocessor + model."""
    preprocessor = build_preprocessor()
    return ImbPipeline([
        ("preprocessor", preprocessor),
        ("smote", SMOTE(random_state=42)),
        ("classifier", estimator),
    ])
