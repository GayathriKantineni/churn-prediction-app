import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, roc_auc_score
from xgboost import XGBClassifier

df = pd.read_csv("data/churn.csv")

# 🔥 Fix column names (VERY IMPORTANT)
df.columns = df.columns.str.strip()

# Normalization mapping for variant headers
rename_map = {
    'CustomerID': 'customerID',
    'customerID': 'customerID',
    'Tenure Months': 'tenure',
    'TenureMonths': 'tenure',
    'tenure': 'tenure',
    'Monthly Charges': 'MonthlyCharges',
    'MonthlyCharges': 'MonthlyCharges',
    'Total Charges': 'TotalCharges',
    'TotalCharges': 'TotalCharges',
    'Internet Service': 'InternetService',
    'InternetService': 'InternetService',
    'Contract Type': 'Contract',
    'Contract': 'Contract',
    'Churn Label': 'Churn',
    'Churn_Label': 'Churn',
    'Churn Value': 'Churn',
    'ChurnValue': 'Churn',
    'Churn': 'Churn'
}

df.rename(columns=rename_map, inplace=True)

# Remove duplicate columns created by mapping both Churn Label/Value to Churn
df = df.loc[:, ~df.columns.duplicated()]

print("Columns:", df.columns.tolist())  # debug check

# Drop customerID safely
if "customerID" in df.columns:
    df.drop("customerID", axis=1, inplace=True)

# Fix TotalCharges safely
if "TotalCharges" in df.columns:
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors='coerce')
    df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)

# Determine target column
if "Churn" in df.columns:
    target_col = "Churn"
elif "Churn Value" in df.columns:
    target_col = "Churn Value"
elif "Churn_Label" in df.columns:
    target_col = "Churn_Label"
elif "Churn Label" in df.columns:
    target_col = "Churn Label"
else:
    raise ValueError("No churn target column found in dataset")

# Normalize target values to 0/1
if isinstance(df[target_col], pd.DataFrame):
    # If there are duplicate target columns, take the first one
    y_series = df[target_col].iloc[:, 0]
else:
    y_series = df[target_col]

if y_series.dtype == object:
    y_series = y_series.astype(str).str.strip().map({
        "Yes": 1, "No": 0,
        "Churn": 1, "No Churn": 0,
        "True": 1, "False": 0,
        "1": 1, "0": 0
    })

if not np.issubdtype(y_series.dtype, np.number):
    y_series = pd.to_numeric(y_series, errors='coerce')

if y_series.isna().any():
    raise ValueError(f"Could not parse target values in {target_col}")

df[target_col] = y_series.astype(int)

# Ensure selected feature columns are present
feature_cols = ["tenure", "MonthlyCharges", "TotalCharges", "Contract", "InternetService"]
missing_feats = [c for c in feature_cols if c not in df.columns]
if missing_feats:
    raise ValueError(f"Missing required feature columns: {missing_feats}")

# Map contract and internet to numeric labels
contract_map = {"Month-to-month":0, "One year":1, "Two year":2}
internet_map = {"DSL":0, "Fiber optic":1, "No":2}

df["Contract"] = df["Contract"].map(contract_map)
df["InternetService"] = df["InternetService"].map(internet_map)

# Split
X = df[feature_cols]
y = df["Churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train model
model = XGBClassifier(n_estimators=200, learning_rate=0.05, max_depth=5)
model.fit(X_train, y_train)

# Evaluate
try:
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:,1]
    print(classification_report(y_test, y_pred))
    print("ROC-AUC:", roc_auc_score(y_test, y_prob))
except Exception as e:
    print("Warning: evaluation step failed:", e)

# Save model & scaler
joblib.dump(model, "models/churn_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")

print("✅ Model & scaler saved successfully!")