import os
import sys

# Make project root importable
PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

sys.path.insert(0, PROJECT_ROOT)

from src.serving.inference import predict


def main():

    print("=" * 60)
    print("=== PHASE 4: MODEL SERVING / INFERENCE TEST ===")
    print("=" * 60)

    # ============================================================
    # TEST CUSTOMER
    # ============================================================

    sample_data = {
        "gender": "Female",
        "Partner": "No",
        "Dependents": "No",

        "PhoneService": "Yes",
        "MultipleLines": "No",

        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "Yes",

        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",

        "tenure": 1,
        "MonthlyCharges": 85.0,
        "TotalCharges": 85.0,
    }

    # ============================================================
    # DISPLAY INPUT
    # ============================================================

    print("\n[1] Customer input...")
    print("-" * 60)

    for key, value in sample_data.items():
        print(f"{key:20} : {value}")

    # ============================================================
    # RUN PREDICTION
    # ============================================================

    print("\n[2] Running inference...")
    print("-" * 60)

    try:

        prediction = predict(sample_data)

        print("\n[3] Prediction result...")
        print("-" * 60)

        print(f"Prediction: {prediction}")

        # ========================================================
        # VALIDATE OUTPUT
        # ========================================================

        allowed_outputs = {
            "Likely to churn",
            "Not likely to churn"
        }

        assert prediction in allowed_outputs

        print("\n[4] Output validation...")
        print("-" * 60)

        print("✅ Prediction output is valid")

        # ========================================================
        # FINAL RESULT
        # ========================================================

        print("\n" + "=" * 60)
        print("✅ PHASE 4 SERVING TEST PASSED")
        print("=" * 60)

    except Exception as e:

        print("\n" + "=" * 60)
        print("❌ PHASE 4 SERVING TEST FAILED")
        print("=" * 60)

        print(f"Error: {e}")

        raise


if __name__ == "__main__":
    main()