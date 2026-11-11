# AI-Driven Predictive Analysis System

A simple, end-to-end machine learning project that predicts **customer churn**
(whether a customer will leave) from historical data.

## Features
- Synthetic dataset generator (or bring your own CSV)
- Preprocessing: scaling numeric features, one-hot encoding categoricals
- Trains and compares Logistic Regression and Random Forest
- Evaluates with Accuracy, F1, ROC-AUC and a classification report
- Saves the best model and predicts for new customers
- Generates a feature-importance chart

## Setup
```bash
pip install -r requirements.txt
```

## Usage
Train (uses synthetic data if no CSV is given):
```bash
python predictive_system.py train
```

Train on your own data (needs a binary 0/1 target column):
```bash
python predictive_system.py train --data my_data.csv --target churn
```

Predict for one customer:
```bash
python predictive_system.py predict --tenure 5 --monthly_charges 85 \
    --support_calls 4 --contract month-to-month --age 30
```

## Output files
| File | Description |
|------|-------------|
| `customer_data.csv` | Generated dataset |
| `best_model.joblib` | Saved best model |
| `feature_importance.png` | Feature importance chart |

## Project flow
```
Data -> Preprocess -> Train & Compare -> Evaluate -> Save -> Predict
```

## Ideas to extend
- Swap in other models (XGBoost, Gradient Boosting)
- Add hyperparameter tuning with `GridSearchCV`
- Wrap `predict()` in a Flask/FastAPI endpoint or Streamlit dashboard
- Adapt it for regression (sales forecasting, price prediction)
