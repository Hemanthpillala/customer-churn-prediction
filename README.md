# Customer Churn Prediction

End-to-end ML pipeline comparing Random Forest, Gradient Boosting, and ANN for predicting customer churn. Uses SMOTE for class imbalance, 5-fold cross-validation, and full precision/recall/AUC analysis.

## Results

| Model | Accuracy | F1 | AUC-ROC |
|-------|----------|----|---------|
| Random Forest | 0.812 | 0.794 | 0.861 |
| **Gradient Boosting** | **0.831** | **0.817** | **0.883** |
| ANN (MLP) | 0.808 | 0.789 | 0.857 |

## Architecture

```
Raw CSV → Clean + Encode → SMOTE → [RF / GBM / ANN] → 5-Fold CV → Best Model
```

## Project Structure

```
customer-churn-prediction/
├── src/
│   ├── preprocess.py   # Cleaning, encoding, SMOTE pipeline
│   ├── train.py        # Cross-validated model comparison
│   └── evaluate.py     # Full metrics on held-out test set
├── requirements.txt
└── README.md
```

## Quickstart

```bash
pip install -r requirements.txt

# Train all models and compare
python src/train.py --data_path data/telco_churn.csv

# Evaluate best saved model
python src/evaluate.py --model_path outputs/best_model.pkl --data_path data/telco_churn.csv
```

## Dataset

Tested on the [Telco Customer Churn dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn). Place CSV in `data/telco_churn.csv`.

## Tech Stack

`scikit-learn` · `imbalanced-learn` · `SMOTE` · `Gradient Boosting` · `Random Forest` · `ANN`
