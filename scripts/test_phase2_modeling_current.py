import os
import sys
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, recall_score
from xgboost import XGBClassifier

sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")
    )
)

from src.data.load_data import load_data
from src.data.preprocess import preprocess_data
from src.features.build_features import build_features


DATA_PATH = r"D:\TELCO Churn\Data\raw\WA_Fn-UseC_-Telco-Customer-Churn.csv"
TARGET_COL = "Churn"


print("=== PHASE 2: MODELING TEST ===")

# 1. Load
print("\n[1] Loading data...")
df = load_data(DATA_PATH)
print(f"Loaded: {df.shape}")


# 2. Preprocess
print("\n[2] Preprocessing...")
df = preprocess_data(df, target_col=TARGET_COL)
print(f"After preprocessing: {df.shape}")


# 3. Feature engineering
print("\n[3] Building features...")
df = build_features(df, target_col=TARGET_COL)

# Convert bool → int
for col in df.select_dtypes(include=["bool"]).columns:
    df[col] = df[col].astype(int)

print(f"After feature engineering: {df.shape}")


# 4. Separate X and y
print("\n[4] Creating X and y...")

X = df.drop(columns=[TARGET_COL])
y = df[TARGET_COL]

print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")

print("\nTarget distribution:")
print(y.value_counts())


# 5. Train/test split
print("\n[5] Train/Test split...")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

print(f"X_train: {X_train.shape}")
print(f"X_test : {X_test.shape}")
print(f"y_train: {y_train.shape}")
print(f"y_test : {y_test.shape}")


# 6. Class imbalance
scale_pos_weight = (
    (y_train == 0).sum()
    /
    (y_train == 1).sum()
)

print(f"\nScale positive weight: {scale_pos_weight:.3f}")


# 7. Train model
print("\n[6] Training XGBoost...")

model = XGBClassifier(
    n_estimators=301,
    learning_rate=0.034,
    max_depth=7,
    subsample=0.95,
    colsample_bytree=0.98,
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    n_jobs=-1,
    eval_metric="logloss"
)

model.fit(X_train, y_train)

print("✅ Model training completed")


# 8. Prediction
print("\n[7] Generating predictions...")

proba = model.predict_proba(X_test)[:, 1]

THRESHOLD = 0.35

y_pred = (proba >= THRESHOLD).astype(int)

print(f"Threshold: {THRESHOLD}")


# 9. Evaluation
print("\n[8] Evaluation")

recall = recall_score(
    y_test,
    y_pred,
    pos_label=1
)

print(f"Recall: {recall:.4f}")

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        digits=3
    )
)

print("\n✅ PHASE 2 MODELING TEST COMPLETED")