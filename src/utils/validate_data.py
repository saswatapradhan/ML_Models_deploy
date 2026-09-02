import pandas as pd
import great_expectations as gx
from typing import Tuple, List
def validate_telco_data(df) -> Tuple[bool, List[str]]:
    """
    Validate Telco Customer Churn data using Great Expectations 1.5.x.

    Validation strategy:

    1. Validate the raw dataset structure and categorical values.
    2. Create a validation copy of the DataFrame.
    3. Apply the same TotalCharges numeric conversion used by
       the preprocessing pipeline.
    4. Fill the known blank TotalCharges values with 0 for
       numeric/business-rule validation.
    5. Run all Great Expectations checks.
    6. Return validation status and failed expectation types.

    The original DataFrame is never modified.
    """

    print("🔍 Starting data validation with Great Expectations...")

    # ============================================================
    # 1. COPY DATA
    # ============================================================

    validation_df = df.copy()

    # ============================================================
    # 2. NORMALIZE TOTALCHARGES FOR VALIDATION
    # ============================================================
    #
    # The original Telco dataset contains 11 blank TotalCharges
    # values.
    #
    # The project's preprocessing pipeline already handles this:
    #
    #     pd.to_numeric(..., errors="coerce")
    #     fillna(0)
    #
    # We reproduce that behavior here ONLY on the validation copy.
    #
    # The original df remains untouched.
    # ============================================================

    if "TotalCharges" in validation_df.columns:

        validation_df["TotalCharges"] = pd.to_numeric(
            validation_df["TotalCharges"],
            errors="coerce"
        )

        validation_df["TotalCharges"] = (
            validation_df["TotalCharges"]
            .fillna(0)
        )

    # ============================================================
    # 3. CREATE GREAT EXPECTATIONS CONTEXT
    # ============================================================

    context = gx.get_context()

    # Create in-memory pandas datasource
    datasource = context.data_sources.add_pandas(
        "telco_pandas"
    )

    # Create dataframe asset
    data_asset = datasource.add_dataframe_asset(
        name="telco_data"
    )

    # Create whole dataframe batch definition
    batch_definition = (
        data_asset.add_batch_definition_whole_dataframe(
            "whole_dataframe"
        )
    )

    # Create batch from validation DataFrame
    batch = batch_definition.get_batch(
        batch_parameters={
            "dataframe": validation_df
        }
    )

    # ============================================================
    # 4. CREATE EXPECTATION SUITE
    # ============================================================

    suite = gx.ExpectationSuite(
        name="telco_validation_suite"
    )

    print(
        "   📋 Validating schema and required columns..."
    )

    expectations = [

        # ========================================================
        # SCHEMA VALIDATION
        # ========================================================

        gx.expectations.ExpectColumnToExist(
            column="customerID"
        ),

        gx.expectations.ExpectColumnValuesToNotBeNull(
            column="customerID"
        ),

        gx.expectations.ExpectColumnToExist(
            column="gender"
        ),

        gx.expectations.ExpectColumnToExist(
            column="Partner"
        ),

        gx.expectations.ExpectColumnToExist(
            column="Dependents"
        ),

        gx.expectations.ExpectColumnToExist(
            column="PhoneService"
        ),

        gx.expectations.ExpectColumnToExist(
            column="InternetService"
        ),

        gx.expectations.ExpectColumnToExist(
            column="Contract"
        ),

        gx.expectations.ExpectColumnToExist(
            column="tenure"
        ),

        gx.expectations.ExpectColumnToExist(
            column="MonthlyCharges"
        ),

        gx.expectations.ExpectColumnToExist(
            column="TotalCharges"
        ),

        # ========================================================
        # BUSINESS LOGIC VALIDATION
        # ========================================================

        gx.expectations.ExpectColumnValuesToBeInSet(
            column="gender",
            value_set=[
                "Male",
                "Female"
            ]
        ),

        gx.expectations.ExpectColumnValuesToBeInSet(
            column="Partner",
            value_set=[
                "Yes",
                "No"
            ]
        ),

        gx.expectations.ExpectColumnValuesToBeInSet(
            column="Dependents",
            value_set=[
                "Yes",
                "No"
            ]
        ),

        gx.expectations.ExpectColumnValuesToBeInSet(
            column="PhoneService",
            value_set=[
                "Yes",
                "No"
            ]
        ),

        gx.expectations.ExpectColumnValuesToBeInSet(
            column="Contract",
            value_set=[
                "Month-to-month",
                "One year",
                "Two year"
            ]
        ),

        gx.expectations.ExpectColumnValuesToBeInSet(
            column="InternetService",
            value_set=[
                "DSL",
                "Fiber optic",
                "No"
            ]
        ),

        # ========================================================
        # NUMERIC VALIDATION
        # ========================================================

        # Tenure cannot be negative
        gx.expectations.ExpectColumnValuesToBeBetween(
            column="tenure",
            min_value=0
        ),

        # Monthly charges cannot be negative
        gx.expectations.ExpectColumnValuesToBeBetween(
            column="MonthlyCharges",
            min_value=0
        ),

        # Total charges cannot be negative.
        #
        # No max_value is specified because TotalCharges is
        # cumulative and can be much greater than $200.
        gx.expectations.ExpectColumnValuesToBeBetween(
            column="TotalCharges",
            min_value=0
        ),

        # ========================================================
        # RANGE VALIDATION
        # ========================================================

        # Telco dataset/business constraint:
        # maximum expected tenure = 120 months
        gx.expectations.ExpectColumnValuesToBeBetween(
            column="tenure",
            min_value=0,
            max_value=120
        ),

        # Monthly charges expected range
        gx.expectations.ExpectColumnValuesToBeBetween(
            column="MonthlyCharges",
            min_value=0,
            max_value=200
        ),

        # ========================================================
        # MISSING VALUE VALIDATION
        # ========================================================

        # These fields must never be missing.
        gx.expectations.ExpectColumnValuesToNotBeNull(
            column="tenure"
        ),

        gx.expectations.ExpectColumnValuesToNotBeNull(
            column="MonthlyCharges"
        ),

        # ========================================================
        # DATA CONSISTENCY
        # ========================================================

        # TotalCharges should generally be greater than or equal
        # to MonthlyCharges.
        #
        # TotalCharges has already been normalized above using
        # the same strategy as preprocess_data().
        gx.expectations.ExpectColumnPairValuesAToBeGreaterThanB(
            column_A="TotalCharges",
            column_B="MonthlyCharges",
            or_equal=True,
            mostly=0.95
        ),
    ]

    # ============================================================
    # 5. ADD EXPECTATIONS
    # ============================================================

    for expectation in expectations:

        suite.add_expectation(
            expectation
        )

    print(
        f"   ⚙️ Running complete validation suite "
        f"({len(expectations)} checks)..."
    )

    # ============================================================
    # 6. RUN VALIDATION
    # ============================================================

    results = batch.validate(
        expect=suite
    )

    # ============================================================
    # 7. PROCESS RESULTS
    # ============================================================

    failed_expectations = []

    for result in results["results"]:

        if not result["success"]:

            expectation_config = (
                result["expectation_config"]
            )

            expectation_type = (
                expectation_config["type"]
            )

            kwargs = expectation_config.get(
                "kwargs",
                {}
            )

            column = kwargs.get(
                "column",
                "N/A"
            )

            print("\n❌ FAILED EXPECTATION")

            print(
                f"   Type   : {expectation_type}"
            )

            print(
                f"   Column : {column}"
            )

            # Print detailed result information
            result_details = result.get(
                "result",
                {}
            )

            if result_details:

                print(
                    f"   Result : {result_details}"
                )

            failed_expectations.append(
                expectation_type
            )

    # ============================================================
    # 8. VALIDATION SUMMARY
    # ============================================================

    total_checks = len(
        results["results"]
    )

    passed_checks = sum(
        1
        for result in results["results"]
        if result["success"]
    )

    failed_checks = (
        total_checks - passed_checks
    )

    print("\n" + "=" * 60)

    print(
        "DATA VALIDATION SUMMARY"
    )

    print("=" * 60)

    print(
        f"Total checks : {total_checks}"
    )

    print(
        f"Passed       : {passed_checks}"
    )

    print(
        f"Failed       : {failed_checks}"
    )

    # ============================================================
    # 9. FINAL RESULT
    # ============================================================

    if results["success"]:

        print(
            f"✅ Data validation PASSED: "
            f"{passed_checks}/{total_checks} "
            f"checks successful"
        )

    else:

        print(
            f"❌ Data validation FAILED: "
            f"{failed_checks}/{total_checks} "
            f"checks failed"
        )

        print(
            f"   Failed expectations: "
            f"{failed_expectations}"
        )

    print("=" * 60)

    return (
        results["success"],
        failed_expectations
    )