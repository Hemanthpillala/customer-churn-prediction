"""
Train and compare Random Forest, Gradient Boosting, and ANN for churn prediction.

Usage:
    python src/train.py --data_path data/telco_churn.csv
"""

import argparse
import json
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import (classification_report, f1_score, precision_score,
                              recall_score, roc_auc_score, accuracy_score)
from preprocess import load_and_clean, get_pipeline


MODELS = {
    "RandomForest": RandomForestClassifier(
        n_estimators=200, max_depth=10, min_samples_leaf=4,
        class_weight="balanced", random_state=42, n_jobs=-1
    ),
    "GradientBoosting": GradientBoostingClassifier(
        n_estimators=200, learning_rate=0.05, max_depth=4,
        subsample=0.8, random_state=42
    ),
    "ANN": MLPClassifier(
        hidden_layer_sizes=(128, 64, 32), activation="relu",
        solver="adam", alpha=1e-3, learning_rate_schedule="adaptive",
        max_iter=500, early_stopping=True, random_state=42
    ),
}


def evaluate_model(name, pipeline, X, y):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scoring = ["accuracy", "f1", "precision", "recall", "roc_auc"]
    scores = cross_validate(pipeline, X, y, cv=cv, scoring=scoring, return_train_score=False)

    result = {
        "model": name,
        "accuracy": round(scores["test_accuracy"].mean(), 4),
        "f1": round(scores["test_f1"].mean(), 4),
        "precision": round(scores["test_precision"].mean(), 4),
        "recall": round(scores["test_recall"].mean(), 4),
        "auc_roc": round(scores["test_roc_auc"].mean(), 4),
        "accuracy_std": round(scores["test_accuracy"].std(), 4),
    }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="data/telco_churn.csv")
    parser.add_argument("--output_dir", type=str, default="outputs/")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    X, y = load_and_clean(args.data_path)
    print(f"Dataset: {len(X)} samples | Churn rate: {y.mean():.2%}")

    results = []
    best_f1, best_name, best_pipeline = 0.0, None, None

    for name, estimator in MODELS.items():
        print(f"\nTraining {name}...")
        pipeline = get_pipeline(estimator)
        result = evaluate_model(name, pipeline, X, y)
        results.append(result)
        print(f"  Accuracy: {result['accuracy']:.4f} | F1: {result['f1']:.4f} | "
              f"AUC: {result['auc_roc']:.4f}")

        if result["f1"] > best_f1:
            best_f1 = result["f1"]
            best_name = name
            best_pipeline = pipeline

    # Fit best model on full data and save
    best_pipeline.fit(X, y)
    joblib.dump(best_pipeline, os.path.join(args.output_dir, "best_model.pkl"))
    print(f"\nBest model: {best_name} (F1={best_f1:.4f})")

    with open(os.path.join(args.output_dir, "comparison.json"), "w") as f:
        json.dump({"results": results, "best_model": best_name}, f, indent=2)
    print(f"Results saved to {args.output_dir}/comparison.json")


if __name__ == "__main__":
    main()
