"""
Load saved model and run full evaluation on held-out test set.

Usage:
    python src/evaluate.py --model_path outputs/best_model.pkl --data_path data/telco_churn.csv
"""

import argparse
import joblib
import json
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (classification_report, confusion_matrix,
                              roc_auc_score, f1_score, accuracy_score)
from preprocess import load_and_clean


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="outputs/best_model.pkl")
    parser.add_argument("--data_path", type=str, default="data/telco_churn.csv")
    args = parser.parse_args()

    X, y = load_and_clean(args.data_path)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    model = joblib.load(args.model_path)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)

    print(f"Accuracy : {acc:.4f}")
    print(f"F1       : {f1:.4f}")
    print(f"AUC-ROC  : {auc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["No Churn", "Churn"]))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))


if __name__ == "__main__":
    main()
