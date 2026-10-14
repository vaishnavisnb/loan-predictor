"""
Loan Approval Predictor - full training pipeline in one script.

Run from the project root (the loan-predictor folder):
    python train.py

Expects: data/train.csv (Kaggle Loan Prediction dataset with a Loan_Status column)
Creates: models/model.pkl, models/columns.pkl, reports/charts.png, reports/model_comparison.csv
"""

import os

import joblib
import matplotlib

matplotlib.use("Agg")  # save charts to a file instead of opening a window
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

DATA_PATH = "data/train.csv"

os.makedirs("models", exist_ok=True)
os.makedirs("reports", exist_ok=True)

# ---------- 1. Load and explore ----------
df = pd.read_csv(DATA_PATH)
print("Shape:", df.shape)
print("\nMissing values:\n", df.isnull().sum())
print("\nTarget distribution:\n", df["Loan_Status"].value_counts())

# ---------- 2. Charts ----------
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
sns.countplot(x="Credit_History", hue="Loan_Status", data=df, ax=axes[0])
sns.countplot(x="Education", hue="Loan_Status", data=df, ax=axes[1])
sns.histplot(df["ApplicantIncome"], bins=40, ax=axes[2])
plt.tight_layout()
plt.savefig("reports/charts.png")
plt.close()
print("\nSaved charts to reports/charts.png")

# ---------- 3. Clean the data ----------
df = df.drop(columns=["Loan_ID"])

cat_cols = [
    "Gender",
    "Married",
    "Dependents",
    "Self_Employed",
    "Credit_History",
    "Loan_Amount_Term",
]
for col in cat_cols:
    df[col] = df[col].fillna(df[col].mode()[0])
df["LoanAmount"] = df["LoanAmount"].fillna(df["LoanAmount"].median())

df["Dependents"] = df["Dependents"].replace("3+", 3).astype(int)
df["Loan_Status"] = df["Loan_Status"].map({"Y": 1, "N": 0})

X = pd.get_dummies(df.drop(columns=["Loan_Status"]), drop_first=True)
y = df["Loan_Status"]

# ---------- 4. Split, train, compare ----------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
}

results = []
for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    results.append(
        {
            "Model": name,
            "Accuracy": accuracy_score(y_test, pred),
            "Precision": precision_score(y_test, pred),
            "Recall": recall_score(y_test, pred),
            "F1": f1_score(y_test, pred),
        }
    )

results_df = pd.DataFrame(results).round(3)
print("\nModel comparison:\n", results_df.to_string(index=False))
results_df.to_csv("reports/model_comparison.csv", index=False)

# ---------- 5. Save the best model (highest F1) ----------
best_name = results_df.sort_values("F1", ascending=False).iloc[0]["Model"]
joblib.dump(models[best_name], "models/model.pkl")
joblib.dump(list(X.columns), "models/columns.pkl")
# Background sample used by SHAP in the app to explain predictions
joblib.dump(X_train.astype(float).sample(100, random_state=42), "models/background.pkl")
print(f"\nBest model: {best_name} (saved to models/model.pkl)")
