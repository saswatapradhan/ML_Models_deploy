from src.serving.inference import predict


def test_prediction_returns_valid_result():

    customer = {
        "gender": "Male",
        "Partner": "No",
        "Dependents": "No",
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "DSL",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "tenure": 12,
        "MonthlyCharges": 50.0,
        "TotalCharges": 600.0,
    }

    result = predict(customer)

    assert result in [
        "Likely to churn",
        "Not likely to churn",
     ]

#    assert result =="THIS SHOULD FAIL"
