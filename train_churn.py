"""
Starter script for Part A.
This runs end-to-end but has a bug that inflates your evaluation metric.
Find it, fix it, explain it in your README.
"""

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, roc_auc_score
import joblib

df = pd.read_csv("data/customers.csv")

# minimal cleanup so this runs at all -- NOT the bug you're looking for,
# feel free to improve this as part of your EDA writeup
df["age"] = pd.to_numeric(df["age"], errors="coerce")
df = pd.get_dummies(df, columns=["contract_type"], drop_first=True)

""""
df["age"] = df["age"].fillna(df["age"].median())
df["monthly_charge"] = df["monthly_charge"].fillna(df["monthly_charge"].median())
df = pd.get_dummies(df, columns=["contract_type"], drop_first=True)"""


# TODO(candidate): there's a leaky/derived column in here somewhere. Find it.
#feature_cols = [c for c in df.columns if c not in ["customer_id", "churned"]]
# the leaky column is "days_since_last_cancellation_request" 
# `-1` corresponds to customers without a cancellation request and non-negative values correspond to customers who churned
# the model can learn to predict churn based on this column, which is not available at prediction time .
feature_cols =  [c for c in df.columns if c not in ["customer_id", "churned" , "days_since_last_cancellation_request"]]
X = df[feature_cols]
y = df["churned"]


# --- bug lives near here ---
# FIX: split FIRST, then learn medians and scaling from the training set only,
# and apply them to the test set. This prevents test-set statistics from
# leaking into training.
"""scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # fit on the FULL dataset, before the split
"""
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# impute with TRAINING-set medians only
train_medians = X_train[["age", "monthly_charge"]].median()
X_train = X_train.fillna(train_medians)
X_test = X_test.fillna(train_medians) # impute with TRAINING-set medians only

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)   # fit on train only
X_test = scaler.transform(X_test)         # apply (not fit) to test
# ----------------------------

model = LogisticRegression(max_iter=1000, class_weight="balanced")
model.fit(X_train, y_train)

preds = model.predict_proba(X_test)[:, 1]
predicted_labels = model.predict(X_test)

print("AUC:", roc_auc_score(y_test, preds))
print("Churn precision:", precision_score(y_test, predicted_labels, pos_label=1))
print("Churn recall:", recall_score(y_test, predicted_labels, pos_label=1))

joblib.dump({"model": model, "scaler": scaler, "feature_cols": feature_cols , "train_medians": train_medians}, "model.pkl")
