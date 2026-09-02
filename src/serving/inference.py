"""
TELCO CUSTOMER CHURN - PRODUCTION INFERENCE
============================================

Single source of truth for model inference.

The inference module supports:

1. Local development
2. Docker deployment
3. MLflow model artifacts
4. Training/serving feature consistency

Serving artifact structure:

src/serving/model/
│
├── MLmodel
├── model.xgb
├── feature_columns.txt
├── preprocessing.pkl
├── conda.yaml
├── python_env.yaml
├── requirements.txt
└── metadata/
"""

from pathlib import Path
import pandas as pd
import mlflow


# ============================================================
# PATH CONFIGURATION
# ============================================================

# Current file:
#
# src/
#   serving/
#       inference.py
#
# Project root:
#
# Telco-Customer-Churn-ML/

CURRENT_FILE = Path(__file__).resolve()

SERVING_DIR = CURRENT_FILE.parent / "model"

PROJECT_ROOT = CURRENT_FILE.parents[2]

MLRUNS_DIR = PROJECT_ROOT / "mlruns"


print("=" * 70)
print("TELCO CHURN - INFERENCE INITIALIZATION")
print("=" * 70)

print(f"📍 Project root : {PROJECT_ROOT}")
print(f"📍 Serving dir  : {SERVING_DIR}")
print(f"📍 MLruns dir   : {MLRUNS_DIR}")


# ============================================================
# MODEL LOADING
# ============================================================

model = None
MODEL_DIR = None


# ------------------------------------------------------------
# OPTION 1
# Docker / production model
# ------------------------------------------------------------

DOCKER_MODEL_DIR = Path("/app/model")


if DOCKER_MODEL_DIR.exists():

    try:

        print(
            f"🔄 Attempting Docker model: "
            f"{DOCKER_MODEL_DIR}"
        )

        model = mlflow.pyfunc.load_model(
            str(DOCKER_MODEL_DIR)
        )

        MODEL_DIR = DOCKER_MODEL_DIR

        print(
            f"✅ Production model loaded from "
            f"{DOCKER_MODEL_DIR}"
        )

    except Exception as e:

        print(
            f"⚠️ Docker model exists but could "
            f"not be loaded: {e}"
        )


# ------------------------------------------------------------
# OPTION 2
# Local serving directory
# ------------------------------------------------------------

if model is None and SERVING_DIR.exists():

    try:

        print(
            f"🔄 Attempting local serving model: "
            f"{SERVING_DIR}"
        )

        model = mlflow.pyfunc.load_model(
            str(SERVING_DIR)
        )

        MODEL_DIR = SERVING_DIR

        print(
            f"✅ Local serving model loaded from "
            f"{SERVING_DIR}"
        )

    except Exception as e:

        print(
            f"⚠️ Local serving directory could "
            f"not be loaded: {e}"
        )


# ------------------------------------------------------------
# OPTION 3
# MLflow fallback
# ------------------------------------------------------------

if model is None:

    print("🔄 Searching MLflow runs...")

    if not MLRUNS_DIR.exists():

        raise FileNotFoundError(
            f"❌ MLflow directory does not exist: "
            f"{MLRUNS_DIR}"
        )

    model_paths = []

    # Search for MLflow model directories.
    #
    # We specifically look for MLmodel because
    # that identifies an MLflow model artifact.

    for mlmodel_file in MLRUNS_DIR.rglob("MLmodel"):

        parent = mlmodel_file.parent

        model_paths.append(parent)


    if not model_paths:

        raise FileNotFoundError(
            "❌ No MLflow model artifacts found "
            f"inside {MLRUNS_DIR}"
        )


    # Most recently modified model

    latest_model = max(
        model_paths,
        key=lambda p: p.stat().st_mtime
    )


    try:

        print(
            f"📦 Loading latest MLflow model: "
            f"{latest_model}"
        )

        model = mlflow.pyfunc.load_model(
            str(latest_model)
        )

        MODEL_DIR = latest_model

        print(
            f"✅ Fallback MLflow model loaded"
        )

    except Exception as e:

        raise RuntimeError(
            f"❌ Failed to load MLflow model: {e}"
        )


# ------------------------------------------------------------
# Final model check
# ------------------------------------------------------------

if model is None:

    raise RuntimeError(
        "❌ Model could not be loaded."
    )


# ============================================================
# FEATURE SCHEMA
# ============================================================

print()
print("🔄 Loading feature schema...")


# IMPORTANT:
#
# feature_columns.txt is stored here:
#
# src/serving/model/feature_columns.txt
#
# NOT:
#
# src/serving/model/<mlflow-model>/feature_columns.txt
#
# Therefore always prefer SERVING_DIR.

FEATURE_FILE = SERVING_DIR / "feature_columns.txt"


if not FEATURE_FILE.exists():

    # Docker fallback

    docker_feature_file = (
        DOCKER_MODEL_DIR /
        "feature_columns.txt"
    )

    if docker_feature_file.exists():

        FEATURE_FILE = docker_feature_file


if not FEATURE_FILE.exists():

    # MLflow fallback.
    #
    # The pipeline may have stored feature_columns.txt
    # beside the MLflow model directory.

    if MODEL_DIR is not None:

        possible_file = (
            MODEL_DIR.parent /
            "feature_columns.txt"
        )

        if possible_file.exists():

            FEATURE_FILE = possible_file


if not FEATURE_FILE.exists():

    raise FileNotFoundError(
        "❌ feature_columns.txt was not found.\n"
        f"Expected location: {SERVING_DIR}"
    )


# ------------------------------------------------------------
# Read feature schema
# ------------------------------------------------------------

with open(
    FEATURE_FILE,
    "r",
    encoding="utf-8"
) as f:

    FEATURE_COLS = [
        line.strip()
        for line in f
        if line.strip()
    ]


print(
    f"✅ Loaded {len(FEATURE_COLS)} "
    f"feature columns"
)

print(
    f"📍 Feature schema: {FEATURE_FILE}"
)


# ============================================================
# FEATURE TRANSFORMATION CONFIGURATION
# ============================================================

BINARY_MAP = {

    "gender": {
        "Female": 0,
        "Male": 1
    },

    "Partner": {
        "No": 0,
        "Yes": 1
    },

    "Dependents": {
        "No": 0,
        "Yes": 1
    },

    "PhoneService": {
        "No": 0,
        "Yes": 1
    },

    "PaperlessBilling": {
        "No": 0,
        "Yes": 1
    }
}


NUMERIC_COLS = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges"
]


# ============================================================
# SERVING TRANSFORMATION
# ============================================================

def _serve_transform(
    df: pd.DataFrame
) -> pd.DataFrame:

    """
    Transform raw customer data into the exact
    feature structure expected by the model.
    """

    df = df.copy()


    # --------------------------------------------------------
    # Clean column names
    # --------------------------------------------------------

    df.columns = (
        df.columns
        .astype(str)
        .str.strip()
    )


    # --------------------------------------------------------
    # Numeric conversion
    # --------------------------------------------------------

    for column in NUMERIC_COLS:

        if column in df.columns:

            df[column] = pd.to_numeric(
                df[column],
                errors="coerce"
            )

            df[column] = (
                df[column]
                .fillna(0)
            )


    # --------------------------------------------------------
    # Binary encoding
    # --------------------------------------------------------

    for column, mapping in BINARY_MAP.items():

        if column in df.columns:

            df[column] = (

                df[column]
                .astype(str)
                .str.strip()
                .map(mapping)
                .astype("Int64")
                .fillna(0)
                .astype(int)

            )


    # --------------------------------------------------------
    # One-hot encoding
    # --------------------------------------------------------

    categorical_columns = (

        df
        .select_dtypes(
            include=["object", "category"]
        )
        .columns
        .tolist()

    )


    if categorical_columns:

        df = pd.get_dummies(
            df,
            columns=categorical_columns,
            drop_first=True
        )


    # --------------------------------------------------------
    # Boolean → integer
    # --------------------------------------------------------

    boolean_columns = (

        df
        .select_dtypes(
            include=["bool"]
        )
        .columns

    )


    if len(boolean_columns) > 0:

        df[boolean_columns] = (
            df[boolean_columns]
            .astype(int)
        )


    # --------------------------------------------------------
    # Align with training feature schema
    # --------------------------------------------------------

    df = df.reindex(
        columns=FEATURE_COLS,
        fill_value=0
    )


    return df


# ============================================================
# PREDICTION FUNCTION
# ============================================================

def predict(
    input_dict: dict
) -> str:

    """
    Predict customer churn.

    Parameters
    ----------
    input_dict : dict
        Raw customer information.

    Returns
    -------
    str
        "Likely to churn"
        or
        "Not likely to churn"
    """


    # --------------------------------------------------------
    # Create DataFrame
    # --------------------------------------------------------

    df = pd.DataFrame(
        [input_dict]
    )


    # --------------------------------------------------------
    # Transform features
    # --------------------------------------------------------

    df_encoded = _serve_transform(
        df
    )


    # --------------------------------------------------------
    # Validate feature count
    # --------------------------------------------------------

    if df_encoded.shape[1] != len(FEATURE_COLS):

        raise ValueError(
            "❌ Feature count mismatch.\n"
            f"Expected: {len(FEATURE_COLS)}\n"
            f"Received: {df_encoded.shape[1]}"
        )


    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    try:

        predictions = model.predict(
            df_encoded
        )

    except Exception as e:

        raise RuntimeError(
            f"❌ Model prediction failed: {e}"
        )


    # --------------------------------------------------------
    # Normalize prediction
    # --------------------------------------------------------

    if hasattr(
        predictions,
        "tolist"
    ):

        predictions = (
            predictions.tolist()
        )


    if isinstance(
        predictions,
        (list, tuple)
    ):

        if len(predictions) != 1:

            raise ValueError(
                "❌ Expected exactly one "
                "prediction."
            )

        result = predictions[0]

    else:

        result = predictions


    # --------------------------------------------------------
    # Convert to integer
    # --------------------------------------------------------

    try:

        result = int(result)

    except Exception as e:

        raise ValueError(
            f"❌ Invalid model prediction: "
            f"{result}"
        ) from e


    # --------------------------------------------------------
    # Business output
    # --------------------------------------------------------

    if result == 1:

        return "Likely to churn"

    return "Not likely to churn"