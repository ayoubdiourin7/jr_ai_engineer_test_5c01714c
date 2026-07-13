"""
Starter script for Part A.
This runs end-to-end but has a bug that inflates your evaluation metric.
Find it, fix it, explain it in your README.
"""

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import joblib

df = pd.read_csv("data/customers.csv")

# minimal cleanup so this runs at all -- NOT the bug you're looking for,
# feel free to improve this as part of your EDA writeup
df["age"] = pd.to_numeric(df["age"], errors="coerce")
df["age"] = df["age"].fillna(df["age"].median())
df["monthly_charge"] = df["monthly_charge"].fillna(df["monthly_charge"].median())
df = pd.get_dummies(df, columns=["contract_type"], drop_first=True)

# TODO(candidate): there's a leaky/derived column in here somewhere. Find it.
feature_cols = [c for c in df.columns if c not in ["customer_id", "churned"]]
X = df[feature_cols]
y = df["churned"]


# --- bug lives near here ---
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # fit on the FULL dataset, before the split

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)
# ----------------------------

model = LogisticRegression(max_iter=1000, class_weight="balanced")
model.fit(X_train, y_train)

preds = model.predict_proba(X_test)[:, 1]
print("AUC:", roc_auc_score(y_test, preds))

joblib.dump({"model": model, "scaler": scaler, "feature_cols": feature_cols}, "model.pkl")
