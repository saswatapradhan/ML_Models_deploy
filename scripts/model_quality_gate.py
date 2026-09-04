import json
import sys
from pathlib import Path


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

METRICS_FILE = PROJECT_ROOT / "artifacts" / "metrics.json"
BASELINE_FILE = PROJECT_ROOT / "config" / "model_baseline.json"


# ============================================================
# LOAD JSON FILE
# ============================================================

def load_json(file_path):

    if not file_path.exists():
        print("❌ QUALITY GATE FAILED")
        print(f"File not found: {file_path}")
        sys.exit(1)

    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


# ============================================================
# QUALITY GATE
# ============================================================

def run_quality_gate(metrics, baseline):

    print("=" * 70)
    print("MODEL QUALITY GATE")
    print("=" * 70)

    print(f"Metrics file : {METRICS_FILE}")
    print(f"Baseline file: {BASELINE_FILE}")
    print()

    gate_passed = True

    for metric_name, baseline_value in baseline.items():

        actual_value = metrics.get(metric_name)

        if actual_value is None:
            print(f"❌ {metric_name.upper():<10} Missing")
            gate_passed = False
            continue

        passed = actual_value >= baseline_value

        status = "✅ PASS" if passed else "❌ FAIL"

        print(
            f"{status}  "
            f"{metric_name.upper():<10} "
            f"Actual: {actual_value:.4f}  "
            f"Required: {baseline_value:.4f}"
        )

        if not passed:
            gate_passed = False

    print()
    print("-" * 70)

    if gate_passed:
        print("✅ MODEL QUALITY GATE PASSED")
        print("Model meets or exceeds the baseline metrics.")
        print("-" * 70)
        return 0

    print("❌ MODEL QUALITY GATE FAILED")
    print("Model performance is below the established baseline.")
    print("-" * 70)
    return 1


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    metrics = load_json(METRICS_FILE)
    baseline = load_json(BASELINE_FILE)

    exit_code = run_quality_gate(metrics, baseline)

    sys.exit(exit_code)