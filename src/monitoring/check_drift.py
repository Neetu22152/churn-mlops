"""
Data drift monitoring using Evidently AI.

The idea: periodically (e.g. via a scheduled Airflow/cron job) compare
recent "production" data against the training reference data. If the
distributions have drifted meaningfully, that's a signal the model may need
retraining — this is the crux of what separates an MLOps pipeline from a
one-off trained model.

Run:
    python -m src.monitoring.check_drift --current data/raw/telco_churn_recent.csv

In production this would run on a schedule and push `drift_detected` to an
alerting channel (Slack/email/PagerDuty) rather than just printing it.
"""
import argparse
import json
from pathlib import Path

import pandas as pd

REFERENCE_DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "raw" / "telco_churn.csv"
REPORT_DIR = Path(__file__).resolve().parents[2] / "monitoring_reports"
REPORT_DIR.mkdir(exist_ok=True)


def check_drift(current_data_path: str, reference_data_path: Path = REFERENCE_DATA_PATH):
    try:
        from evidently.report import Report
        from evidently.metric_preset import DataDriftPreset
    except ImportError:
        print(
            "evidently not installed. Install with `pip install evidently` "
            "to run real drift detection. Falling back to a simple summary-"
            "statistics comparison."
        )
        return _simple_drift_check(current_data_path, reference_data_path)

    reference = pd.read_csv(reference_data_path)
    current = pd.read_csv(current_data_path)

    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference, current_data=current)

    report_path = REPORT_DIR / "drift_report.html"
    report.save_html(str(report_path))

    result = report.as_dict()
    drift_detected = result["metrics"][0]["result"]["dataset_drift"]

    print(f"Drift detected: {drift_detected}")
    print(f"Full report saved to {report_path}")

    return {"drift_detected": drift_detected, "report_path": str(report_path)}


def _simple_drift_check(current_data_path: str, reference_data_path: Path, threshold: float = 0.15):
    """Lightweight fallback: flag drift if any numeric column's mean has
    shifted by more than `threshold` (as a fraction of the reference mean).
    Not a substitute for Evidently's statistical tests, but useful when the
    package isn't available and something is still better than nothing.
    """
    reference = pd.read_csv(reference_data_path)
    current = pd.read_csv(current_data_path)

    numeric_cols = ["tenure", "monthly_charges", "total_charges"]
    drift_flags = {}
    for col in numeric_cols:
        ref_mean = reference[col].mean()
        cur_mean = current[col].mean()
        pct_change = abs(cur_mean - ref_mean) / (abs(ref_mean) + 1e-9)
        drift_flags[col] = {
            "reference_mean": round(float(ref_mean), 2),
            "current_mean": round(float(cur_mean), 2),
            "pct_change": round(float(pct_change), 4),
            "drifted": bool(pct_change > threshold),
        }

    any_drift = bool(any(v["drifted"] for v in drift_flags.values()))
    result = {"drift_detected": any_drift, "details": drift_flags}
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--current", required=True, help="Path to recent/production data CSV")
    parser.add_argument("--reference", default=str(REFERENCE_DATA_PATH))
    args = parser.parse_args()

    check_drift(args.current, Path(args.reference))
