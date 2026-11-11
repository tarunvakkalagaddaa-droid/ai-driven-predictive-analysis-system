"""
AI-Driven Predictive Analysis System
------------------------------------
A simple end-to-end machine learning pipeline that predicts customer churn.

Pipeline:
    1. Load data (CSV) or generate a synthetic dataset
    2. Preprocess (scaling + one-hot encoding)
    3. Train and compare multiple models
    4. Evaluate and save the best model
    5. Predict on new customer data
    6. Plot feature importance

Usage:
    python predictive_system.py train
    python predictive_system.py train --data my_data.csv --target churn
    python predictive_system.py predict --tenure 5 --monthly_charges 85 \
        --support_calls 4 --contract month-to-month --age 30
"""

import argparse
import os

import joblib
import matplotlib

matplotlib.use("Agg")  # headless-safe backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

MODEL_PATH = "best_model.joblib"
DATA_PATH = "customer_data.csv"
PLOT_PATH = "feature_importance.png"
RANDOM_STATE = 42


# ----------------------------------------------------------------------------
# 1. Data
# ----------------------------------------------------------------------------
def generate_synthetic_data(n=2000, seed=RANDOM_STATE):
    """Create a realistic-looking customer churn dataset."""
    rng = np.random.default_rng(seed)

    tenure = rng.integers(1, 72, n)                      # months with company
    monthly_charges = rng.normal(65, 25, n).clip(15, 150)
    support_calls = rng.poisson(2, n)                    # calls to support
    age = rng.integers(18, 75, n)
    contract = rng.choice(
        ["month-to-month", "one-year", "two-year"], n, p=[0.55, 0.25, 0.20]
    )

    # Hidden rule that drives churn probability
    score = (
        -0.04 * tenure
        + 0.025 * monthly_charges
        + 0.35 * support_calls
        - 0.01 * age
        + np.where(contract == "month-to-month", 1.2, 0)
        + np.where(contract == "two-year", -1.0, 0)
        - 0.5
    )
    prob = 1 / (1 + np.exp(-score))
    churn = (rng.random(n) < prob).astype(int)

    return pd.DataFrame(
        {
            "tenure": tenure,
            "monthly_charges": monthly_charges.round(2),
            "support_calls": support_calls,
            "age": age,
            "contract": contract,
            "churn": churn,
        }
    )


def load_data(path=None, target="churn"):
    if path and os.path.exists(path):
        print(f"Loading data from {path}")
        df = pd.read_csv(path)
    else:
        print("No dataset supplied - generating synthetic data")
        df = generate_synthetic_data()
        df.to_csv(DATA_PATH, index=False)
        print(f"Saved synthetic dataset to {DATA_PATH}")
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in data")
    return df


# ----------------------------------------------------------------------------
# 2. Preprocessing
# ----------------------------------------------------------------------------
def build_preprocessor(X):
    numeric_cols = X.select_dtypes(include="number").columns.tolist()
    categorical_cols = X.select_dtypes(exclude="number").columns.tolist()
    return ColumnTransformer(
        [
            ("num", StandardScaler(), numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ]
    )


# ----------------------------------------------------------------------------
# 3-4. Train, compare, evaluate, save
# ----------------------------------------------------------------------------
def train(data_path=None, target="churn"):
    df = load_data(data_path, target)
    X, y = df.drop(columns=[target]), df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    candidates = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, random_state=RANDOM_STATE
        ),
    }

    best_name, best_pipe, best_auc = None, None, -1
    print("\nModel comparison")
    print("-" * 52)
    for name, model in candidates.items():
        pipe = Pipeline([("prep", build_preprocessor(X)), ("model", model)])
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)
        proba = pipe.predict_proba(X_test)[:, 1]
        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds)
        auc = roc_auc_score(y_test, proba)
        print(f"{name:<22} Acc={acc:.3f}  F1={f1:.3f}  AUC={auc:.3f}")
        if auc > best_auc:
            best_name, best_pipe, best_auc = name, pipe, auc

    print(f"\nBest model: {best_name} (AUC={best_auc:.3f})")
    print("\nClassification report:")
    print(classification_report(y_test, best_pipe.predict(X_test)))

    joblib.dump({"pipeline": best_pipe, "columns": X.columns.tolist()}, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

    plot_feature_importance(best_pipe)


# ----------------------------------------------------------------------------
# 5. Predict
# ----------------------------------------------------------------------------
def predict(record: dict):
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("No trained model found. Run: python predictive_system.py train")
    bundle = joblib.load(MODEL_PATH)
    pipe, columns = bundle["pipeline"], bundle["columns"]

    row = pd.DataFrame([record])[columns]
    prob = pipe.predict_proba(row)[0, 1]
    label = "WILL CHURN" if prob >= 0.5 else "WILL STAY"

    print(f"\nInput: {record}")
    print(f"Churn probability: {prob:.1%}")
    print(f"Prediction: {label}")
    return prob


# ----------------------------------------------------------------------------
# 6. Visualisation
# ----------------------------------------------------------------------------
def plot_feature_importance(pipe):
    model = pipe.named_steps["model"]
    names = pipe.named_steps["prep"].get_feature_names_out()

    if hasattr(model, "feature_importances_"):
        values = model.feature_importances_
    else:
        values = np.abs(model.coef_[0])

    order = np.argsort(values)
    plt.figure(figsize=(8, 5))
    plt.barh(np.array(names)[order], values[order], color="#3b82f6")
    plt.xlabel("Importance")
    plt.title("Feature Importance")
    plt.tight_layout()
    plt.savefig(PLOT_PATH, dpi=120)
    plt.close()
    print(f"Feature importance chart saved to {PLOT_PATH}")


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="AI-Driven Predictive Analysis System")
    sub = parser.add_subparsers(dest="command", required=True)

    t = sub.add_parser("train", help="Train and evaluate models")
    t.add_argument("--data", help="Path to a CSV file (optional)")
    t.add_argument("--target", default="churn", help="Target column name")

    p = sub.add_parser("predict", help="Predict churn for one customer")
    p.add_argument("--tenure", type=int, required=True)
    p.add_argument("--monthly_charges", type=float, required=True)
    p.add_argument("--support_calls", type=int, required=True)
    p.add_argument("--age", type=int, required=True)
    p.add_argument(
        "--contract",
        choices=["month-to-month", "one-year", "two-year"],
        required=True,
    )

    args = parser.parse_args()
    if args.command == "train":
        train(args.data, args.target)
    else:
        predict(
            {
                "tenure": args.tenure,
                "monthly_charges": args.monthly_charges,
                "support_calls": args.support_calls,
                "age": args.age,
                "contract": args.contract,
            }
        )


if __name__ == "__main__":
    main()
