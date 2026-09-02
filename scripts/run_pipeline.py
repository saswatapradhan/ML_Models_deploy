#!/usr/bin/env python3

"""
TELCO CUSTOMER CHURN - COMPLETE ML PIPELINE
============================================

Pipeline:

1. Load raw data
2. Validate data quality with Great Expectations
3. Preprocess data
4. Build ML features
5. Save feature schema
6. Split train/test data
7. Handle class imbalance
8. Train XGBoost
9. Evaluate model
10. Log model + feature schema to MLflow
11. Export model artifacts for serving

Serving artifact structure:

src/serving/model/
├── MLmodel
├── model.pkl / model.ubj / model.json
├── conda.yaml
├── python_env.yaml
├── requirements.txt
├── metadata/
├── feature_columns.txt
└── preprocessing.pkl

This guarantees training/serving feature consistency.
"""

# ============================================================
# IMPORTS
# ============================================================

from pathlib import Path
import os
import sys
import time
import argparse
import json
import shutil

import pandas as pd
import mlflow
import mlflow.xgboost

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

from xgboost import XGBClassifier


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            ".."
        )
    )
)


# ============================================================
# MAKE SRC IMPORTABLE
# ============================================================

sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# LOCAL MODULES
# ============================================================

from src.data.load_data import load_data
from src.data.preprocess import preprocess_data
from src.features.build_features import build_features
from src.utils.validate_data import validate_telco_data


# ============================================================
# MAIN PIPELINE
# ============================================================

def main(args):

    print("=" * 70)
    print("TELCO CUSTOMER CHURN - ML PIPELINE")
    print("=" * 70)

    # ========================================================
    # 1. MLFLOW SETUP
    # ========================================================

    print("\n🔧 Configuring MLflow...")

    if args.mlflow_uri:
        mlflow_uri = args.mlflow_uri
    else:
        mlruns_path = PROJECT_ROOT / "mlruns"
        mlruns_path.mkdir(parents=True, exist_ok=True)
        # Windows-safe file URI
        mlflow_uri = mlruns_path.as_uri()

    print(f"📍 MLflow tracking URI: {mlflow_uri}")

    mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment(args.experiment)

    # ========================================================
    # START MLFLOW RUN
    # ========================================================

    with mlflow.start_run():

        # ====================================================
        # LOG PIPELINE CONFIGURATION
        # ====================================================

        mlflow.log_param("model", "xgboost")
        mlflow.log_param("target", args.target)
        mlflow.log_param("threshold", args.threshold)
        mlflow.log_param("test_size", args.test_size)

        # ====================================================
        # PHASE 1 - DATA LOADING
        # ====================================================

        print("\n" + "=" * 70)
        print("PHASE 1 - DATA LOADING")
        print("=" * 70)

        print("🔄 Loading data...")

        df = load_data(args.input)

        print(
            f"✅ Data loaded: "
            f"{df.shape[0]} rows, "
            f"{df.shape[1]} columns"
        )

        mlflow.log_param("input_rows", df.shape[0])
        mlflow.log_param("input_columns", df.shape[1])

        # ====================================================
        # PHASE 2 - DATA VALIDATION
        # ====================================================

        print("\n" + "=" * 70)
        print("PHASE 2 - DATA VALIDATION")
        print("=" * 70)

        print("🔍 Validating data quality with Great Expectations...")

        is_valid, failed = validate_telco_data(df)

        mlflow.log_metric("data_quality_pass", int(is_valid))

        if not is_valid:
            print("❌ Data validation failed.")
            mlflow.log_text(
                json.dumps(failed, indent=2),
                artifact_file="failed_expectations.json"
            )
            raise ValueError(
                f"❌ Data quality check failed. Issues: {failed}"
            )

        print("✅ Data validation passed. Logged to MLflow.")

        # ====================================================
        # PHASE 3 - PREPROCESSING
        # ====================================================

        print("\n" + "=" * 70)
        print("PHASE 3 - DATA PREPROCESSING")
        print("=" * 70)

        print("🔧 Preprocessing data...")

        df = preprocess_data(df, target_col=args.target)

        processed_path = (
            PROJECT_ROOT
            / "data"
            / "processed"
            / "telco_churn_processed.csv"
        )

        processed_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(processed_path, index=False)

        print(f"✅ Processed dataset saved to {processed_path}")
        print(f"   Shape: {df.shape}")

        # ====================================================
        # PHASE 4 - FEATURE ENGINEERING
        # ====================================================

        print("\n" + "=" * 70)
        print("PHASE 4 - FEATURE ENGINEERING")
        print("=" * 70)

        target = args.target

        if target not in df.columns:
            raise ValueError(
                f"Target column '{target}' not found in data."
            )

        print("🛠️ Building features...")

        df_enc = build_features(df, target_col=target)

        # ====================================================
        # BOOLEAN → INTEGER
        # ====================================================

        bool_cols = df_enc.select_dtypes(include=["bool"]).columns

        if len(bool_cols) > 0:
            df_enc[bool_cols] = df_enc[bool_cols].astype(int)
            print(f"🔄 Converted {len(bool_cols)} boolean columns to integers")

        print(f"✅ Feature engineering completed: {df_enc.shape[1]} total columns")

        # ====================================================
        # FEATURE SCHEMA
        # ====================================================

        feature_cols = list(df_enc.drop(columns=[target]).columns)

        print(f"✅ Final model features: {len(feature_cols)}")

        # ====================================================
        # SAVE LOCAL FEATURE SCHEMA
        # ====================================================

        artifacts_dir = PROJECT_ROOT / "artifacts"
        artifacts_dir.mkdir(parents=True, exist_ok=True)

        local_feature_file = artifacts_dir / "feature_columns.json"

        with open(local_feature_file, "w", encoding="utf-8") as f:
            json.dump(feature_cols, f, indent=2)

        print(f"✅ Local feature schema saved: {local_feature_file}")

        # ====================================================
        # PREPROCESSING METADATA
        # ====================================================

        preprocessing_artifact = {
            "feature_columns": feature_cols,
            "target": target
        }

        preprocessing_file = artifacts_dir / "preprocessing.json"

        with open(preprocessing_file, "w", encoding="utf-8") as f:
            json.dump(preprocessing_artifact, f, indent=2)

        print(f"✅ Preprocessing metadata saved: {preprocessing_file}")

        # ====================================================
        # PHASE 5 - TRAIN / TEST SPLIT
        # ====================================================

        print("\n" + "=" * 70)
        print("PHASE 5 - TRAIN / TEST SPLIT")
        print("=" * 70)

        X = df_enc.drop(columns=[target])
        y = df_enc[target]

        print(f"X shape: {X.shape}")
        print(f"y shape: {y.shape}")

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=args.test_size,
            stratify=y,
            random_state=42
        )

        print(f"✅ Train: {X_train.shape[0]} samples")
        print(f"✅ Test : {X_test.shape[0]} samples")

        # ====================================================
        # CLASS DISTRIBUTION
        # ====================================================

        print("\n📊 Training target distribution:")
        print(y_train.value_counts())

        # ====================================================
        # CLASS IMBALANCE
        # ====================================================

        scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

        print(f"📈 Class imbalance ratio: {scale_pos_weight:.2f}")

        mlflow.log_param("scale_pos_weight", scale_pos_weight)

        # ====================================================
        # PHASE 6 - MODEL TRAINING
        # ====================================================

        print("\n" + "=" * 70)
        print("PHASE 6 - XGBOOST TRAINING")
        print("=" * 70)

        print("🤖 Training XGBoost model...")

        model = XGBClassifier(
            n_estimators=301,
            learning_rate=0.034,
            max_depth=7,
            subsample=0.95,
            colsample_bytree=0.98,
            n_jobs=-1,
            random_state=42,
            eval_metric="logloss",
            scale_pos_weight=scale_pos_weight
        )

        # ====================================================
        # LOG MODEL PARAMETERS
        # ====================================================

        mlflow.log_params({
            "n_estimators": 301,
            "learning_rate": 0.034,
            "max_depth": 7,
            "subsample": 0.95,
            "colsample_bytree": 0.98
        })

        # ====================================================
        # TRAIN
        # ====================================================

        train_start = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - train_start

        mlflow.log_metric("train_time", train_time)

        print(f"✅ Model trained in {train_time:.2f} seconds")

        # ====================================================
        # PHASE 7 - MODEL EVALUATION
        # ====================================================

        print("\n" + "=" * 70)
        print("PHASE 7 - MODEL EVALUATION")
        print("=" * 70)

        prediction_start = time.time()
        proba = model.predict_proba(X_test)[:, 1]

        # ====================================================
        # CLASSIFICATION THRESHOLD
        # ====================================================

        y_pred = (proba >= args.threshold).astype(int)
        pred_time = time.time() - prediction_start

        mlflow.log_metric("pred_time", pred_time)

        # ====================================================
        # METRICS
        # ====================================================

        precision = precision_score(y_test, y_pred, pos_label=1)
        recall = recall_score(y_test, y_pred, pos_label=1)
        f1 = f1_score(y_test, y_pred, pos_label=1)
        roc_auc = roc_auc_score(y_test, proba)

        # ====================================================
        # LOG METRICS
        # ====================================================

        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1", f1)
        mlflow.log_metric("roc_auc", roc_auc)

        # ====================================================
        # PRINT METRICS
        # ====================================================

        print("🎯 Model Performance:")
        print(f"   Precision : {precision:.3f}")
        print(f"   Recall    : {recall:.3f}")
        print(f"   F1 Score  : {f1:.3f}")
        print(f"   ROC AUC   : {roc_auc:.3f}")

        # ====================================================
        # CLASSIFICATION REPORT
        # ====================================================

        print("\n📈 Detailed Classification Report:")
        print(classification_report(y_test, y_pred, digits=3))

        # ====================================================
        # PHASE 8 - MLFLOW MODEL ARTIFACT
        # ====================================================

        print("\n" + "=" * 70)
        print("PHASE 8 - MLFLOW MODEL ARTIFACT")
        print("=" * 70)

        print("💾 Saving model to MLflow...")

        # ====================================================
        # 8.1 LOG MODEL TO MLFLOW
        # ====================================================

        model_info = mlflow.xgboost.log_model(
            xgb_model=model,
            artifact_path="model"
        )

        print("✅ Model saved to MLflow")

        # ====================================================
        # 8.2 GET CURRENT RUN ID
        # ====================================================

        run_id = mlflow.active_run().info.run_id
        print(f"🆔 MLflow Run ID: {run_id}")

        # ====================================================
        # 8.3 DOWNLOAD MODEL ARTIFACT FROM MLFLOW
        # ====================================================

        model_uri = f"runs:/{run_id}/model"
        print(f"📍 Model URI: {model_uri}")

        downloaded_model_path = mlflow.artifacts.download_artifacts(
            artifact_uri=model_uri
        )

        downloaded_model_path = Path(downloaded_model_path)
        print(f"📦 Model artifact downloaded: {downloaded_model_path}")

        # ====================================================
        # 8.4 SERVING DIRECTORY
        # ====================================================

        serving_model_dir = PROJECT_ROOT / "src" / "serving" / "model"
        serving_model_dir.mkdir(parents=True, exist_ok=True)

        print(f"📍 Serving directory: {serving_model_dir}")

        # ====================================================
        # 8.5 CLEAN OLD MODEL
        # ====================================================

        for item in serving_model_dir.iterdir():
            if item.is_dir():
                shutil.rmtree(item)
            else:
                item.unlink()

        print("🧹 Old serving artifacts removed")

        # ====================================================
        # 8.6 COPY MLFLOW MODEL FILES
        # ====================================================

        for item in downloaded_model_path.iterdir():
            destination = serving_model_dir / item.name
            if item.is_dir():
                shutil.copytree(item, destination)
            else:
                shutil.copy2(item, destination)

        print("✅ MLflow model files copied to serving directory")

        # ====================================================
        # 8.7 CREATE feature_columns.txt
        # ====================================================

        feature_columns_file = serving_model_dir / "feature_columns.txt"

        with open(feature_columns_file, "w", encoding="utf-8") as f:
            for column in feature_cols:
                f.write(f"{column}\n")

        print("✅ feature_columns.txt saved inside serving directory")

        # ====================================================
        # 8.8 COPY preprocessing.pkl
        # ====================================================

        preprocessing_pkl = artifacts_dir / "preprocessing.pkl"

        if preprocessing_pkl.exists():
            shutil.copy2(
                preprocessing_pkl,
                serving_model_dir / "preprocessing.pkl"
            )
            print("✅ preprocessing.pkl copied to serving directory")
        else:
            print("ℹ️ preprocessing.pkl not found. Skipping optional artifact.")

        # ====================================================
        # 8.9 VERIFY REQUIRED SERVING FILES
        # ====================================================

        required_serving_files = [
            "MLmodel",
            "feature_columns.txt"
        ]

        possible_model_files = [
            "model.pkl",
            "model.ubj",
            "model.json",
            "model.xgb"
        ]

        missing_files = []

        # Verify required files
        for filename in required_serving_files:
            file_path = serving_model_dir / filename
            if not file_path.exists():
                missing_files.append(filename)

        # Verify model file
        model_file = None
        for filename in possible_model_files:
            candidate = serving_model_dir / filename
            if candidate.exists():
                model_file = candidate
                break

        if model_file is None:
            model_candidates = list(serving_model_dir.glob("model.*"))
            model_candidates = [
                path for path in model_candidates if path.name != "MLmodel"
            ]
            if model_candidates:
                model_file = model_candidates[0]

        # Final validation
        if missing_files:
            raise FileNotFoundError(
                f"❌ Required serving artifacts missing: {missing_files}"
            )

        if model_file is None:
            raise FileNotFoundError(
                f"❌ No XGBoost model file found in serving directory: {serving_model_dir}"
            )

        print(f"✅ Model file verified: {model_file.name}")
        print("✅ Required serving artifacts verified")

        # ====================================================
        # 8.10 DISPLAY SERVING CONTENTS
        # ====================================================

        print("\n📦 SERVING MODEL CONTENTS")
        print("-" * 60)

        for item in sorted(serving_model_dir.iterdir()):
            if item.is_dir():
                print(f"   📁 {item.name}/")
            else:
                print(f"   ✅ {item.name}")

        print("-" * 60)
        print("✅ Serving model artifact is ready")
        print(f"📍 Serving model directory: {serving_model_dir}")

        # ====================================================
        # LOG LOCAL ARTIFACTS
        # ====================================================

        mlflow.log_artifact(
            str(local_feature_file),
            artifact_path="metadata"
        )

        mlflow.log_artifact(
            str(preprocessing_file),
            artifact_path="metadata"
        )

        # ====================================================
        # LOG PROCESSED DATASET
        # ====================================================

        mlflow.log_artifact(
            str(processed_path),
            artifact_path="data"
        )

        # ====================================================
        # PHASE 9 - PERFORMANCE SUMMARY
        # ====================================================

        print("\n" + "=" * 70)
        print("PHASE 9 - PERFORMANCE SUMMARY")
        print("=" * 70)

        samples_per_second = len(X_test) / pred_time

        print(f"⏱️ Training time     : {train_time:.2f}s")
        print(f"⏱️ Inference time   : {pred_time:.4f}s")
        print(f"📊 Samples/second   : {samples_per_second:.0f}")
        print(f"🎯 Recall           : {recall:.3f}")
        print(f"🎯 ROC AUC          : {roc_auc:.3f}")

        # ====================================================
        # FINAL
        # ====================================================

        print("\n" + "=" * 70)
        print("✅ COMPLETE ML PIPELINE FINISHED SUCCESSFULLY")
        print("=" * 70)
        print("\n🚀 Ready for Phase 4: Model Serving")


# ============================================================
# COMMAND LINE ARGUMENTS
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Run Telco Customer Churn XGBoost + MLflow pipeline"
    )

    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to raw Telco CSV"
    )

    parser.add_argument(
        "--target",
        type=str,
        default="Churn",
        help="Target column"
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.35,
        help="Classification threshold"
    )

    parser.add_argument(
        "--test_size",
        type=float,
        default=0.2,
        help="Test dataset proportion"
    )

    parser.add_argument(
        "--experiment",
        type=str,
        default="Telco Churn",
        help="MLflow experiment name"
    )

    parser.add_argument(
        "--mlflow_uri",
        type=str,
        default=None,
        help="MLflow tracking URI"
    )

    args = parser.parse_args()

    main(args)