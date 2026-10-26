"""Data Validation and Quality Assurance for Insider Threat Telemetry.

Performs schema verification, completeness checks, duplicate detection,
timestamp integrity validation, label distribution analysis, and statistical profiling.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


# Required schema for the raw access telemetry
REQUIRED_COLUMNS: list[str] = [
    "event_id",
    "user_id",
    "timestamp",
    "department",
    "file_id",
    "action",
    "bytes_transferred",
    "destination_type",
    "is_sensitive",
    "is_anomaly",
]

EXPECTED_DEPARTMENTS: set[str] = {
    "Engineering",
    "Finance",
    "HR",
    "Sales",
    "Legal",
    "IT_Admin",
    "Marketing",
}

EXPECTED_DESTINATIONS: set[str] = {
    "internal_server",
    "cloud_drive",
    "local_storage",
    "usb_drive",
    "personal_cloud",
    "external_email",
}


class DataValidationError(Exception):
    """Raised when access logs fail fundamental schema or integrity checks."""


def validate_dataset(data: pd.DataFrame | str | Path) -> dict[str, Any]:
    """Execute comprehensive schema validation and quality metrics calculation.

    Args:
        data: Either a pandas DataFrame or path to a CSV file.

    Returns:
        Dictionary containing validation status, errors, warnings, and statistical profiles.
    """
    if isinstance(data, (str, Path)):
        path = Path(data)
        if not path.exists():
            raise FileNotFoundError(f"Dataset file not found at: {path}")
        df = pd.read_csv(path)
    elif isinstance(data, pd.DataFrame):
        df = data.copy()
    else:
        raise TypeError(f"Expected DataFrame or path, received {type(data)}")

    errors: list[str] = []
    warnings: list[str] = []

    # 1. Schema Validation (Required Columns)
    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_cols:
        errors.append(f"Missing required schema columns: {missing_cols}")

    # 2. Row and Column Counts
    num_rows, num_cols = df.shape
    if num_rows == 0:
        errors.append("Dataset contains 0 rows (empty dataset).")

    # 3. Missing / Null Value Audit
    missing_counts = df.isnull().sum().to_dict()
    total_missing = sum(missing_counts.values())
    if total_missing > 0:
        errors.append(f"Found {total_missing} null values across columns: {missing_counts}")

    # 4. Duplicate Event ID Check
    duplicate_events = 0
    if "event_id" in df.columns:
        duplicate_events = int(df["event_id"].duplicated().sum())
        if duplicate_events > 0:
            errors.append(f"Found {duplicate_events} duplicate event_id values.")

    # 5. Timestamp Integrity
    valid_timestamps = True
    min_ts, max_ts = None, None
    if "timestamp" in df.columns and not df["timestamp"].empty:
        try:
            ts_series = pd.to_datetime(df["timestamp"])
            min_ts = str(ts_series.min())
            max_ts = str(ts_series.max())
            if ts_series.isnull().any():
                errors.append("Some timestamps could not be parsed as valid datetimes.")
                valid_timestamps = False
        except Exception as exc:
            errors.append(f"Failed to parse timestamp column: {exc}")
            valid_timestamps = False

    # 6. Label & Flag Distribution Audit
    anomaly_dist: dict[str, Any] = {}
    if "is_anomaly" in df.columns:
        unique_vals = set(df["is_anomaly"].dropna().unique())
        if not unique_vals.issubset({0, 1}):
            errors.append(f"is_anomaly contains invalid non-binary values: {unique_vals}")

        counts = df["is_anomaly"].value_counts().to_dict()
        normal_cnt = counts.get(0, 0)
        anomaly_cnt = counts.get(1, 0)
        anomaly_rate = (anomaly_cnt / num_rows) if num_rows > 0 else 0.0

        anomaly_dist = {
            "normal_events": int(normal_cnt),
            "anomaly_events": int(anomaly_cnt),
            "anomaly_ratio": float(anomaly_rate),
        }

        # Defensive sanity warning: if anomaly rate > 25% or < 0.5%
        if anomaly_rate > 0.25:
            warnings.append(f"Unusually high anomaly rate: {anomaly_rate:.1%}")
        elif anomaly_rate < 0.005:
            warnings.append(f"Extremely sparse anomaly rate: {anomaly_rate:.2%}")

    # 7. Sensitive Flag Audit
    sensitive_dist: dict[str, Any] = {}
    if "is_sensitive" in df.columns:
        unique_sens = set(df["is_sensitive"].dropna().unique())
        if not unique_sens.issubset({0, 1}):
            errors.append(f"is_sensitive contains non-binary values: {unique_sens}")
        sens_counts = df["is_sensitive"].value_counts().to_dict()
        sensitive_dist = {
            "non_sensitive": int(sens_counts.get(0, 0)),
            "sensitive": int(sens_counts.get(1, 0)),
            "sensitive_ratio": float((sens_counts.get(1, 0) / num_rows) if num_rows > 0 else 0.0),
        }

    # 8. Bytes Transferred Statistical Profile
    bytes_stats: dict[str, Any] = {}
    if "bytes_transferred" in df.columns:
        if (df["bytes_transferred"] < 0).any():
            errors.append("Detected negative values in bytes_transferred column.")
        desc = df["bytes_transferred"].describe().to_dict()
        bytes_stats = {
            "min_bytes": float(desc.get("min", 0)),
            "q25_bytes": float(desc.get("25%", 0)),
            "median_bytes": float(desc.get("50%", 0)),
            "q75_bytes": float(desc.get("75%", 0)),
            "max_bytes": float(desc.get("max", 0)),
            "mean_bytes": float(desc.get("mean", 0)),
            "std_bytes": float(desc.get("std", 0)),
        }

    # 9. Domain Categories Validation
    if "department" in df.columns:
        unknown_depts = set(df["department"].dropna().unique()) - EXPECTED_DEPARTMENTS
        if unknown_depts:
            warnings.append(f"Unrecognized departments encountered: {unknown_depts}")

    if "destination_type" in df.columns:
        unknown_dest = set(df["destination_type"].dropna().unique()) - EXPECTED_DESTINATIONS
        if unknown_dest:
            warnings.append(f"Unrecognized destination types encountered: {unknown_dest}")

    # Compile Validation Summary
    is_valid = len(errors) == 0

    return {
        "is_valid": is_valid,
        "errors": errors,
        "warnings": warnings,
        "shape": {"rows": num_rows, "columns": num_cols},
        "missing_values": missing_counts,
        "duplicate_events": duplicate_events,
        "timestamp_profile": {
            "is_valid": valid_timestamps,
            "min_timestamp": min_ts,
            "max_timestamp": max_ts,
        },
        "label_distribution": anomaly_dist,
        "sensitivity_distribution": sensitive_dist,
        "bytes_transferred_profile": bytes_stats,
        "cardinality": {
            "unique_users": int(df["user_id"].nunique()) if "user_id" in df.columns else 0,
            "unique_files": int(df["file_id"].nunique()) if "file_id" in df.columns else 0,
            "departments": int(df["department"].nunique()) if "department" in df.columns else 0,
            "actions": int(df["action"].nunique()) if "action" in df.columns else 0,
        },
    }


def generate_quality_report(validation_result: dict[str, Any]) -> str:
    """Format validation metrics into a readable terminal data quality report."""
    shape = validation_result["shape"]
    labels = validation_result.get("label_distribution", {})
    bytes_p = validation_result.get("bytes_transferred_profile", {})
    card = validation_result.get("cardinality", {})
    ts = validation_result.get("timestamp_profile", {})

    status_icon = "[PASS]" if validation_result["is_valid"] else "[FAIL]"

    lines = [
        "=" * 70,
        f" DATA QUALITY & SCHEMA AUDIT REPORT {status_icon}".center(70),
        "=" * 70,
        f"Dataset Shape:         {shape['rows']:,} rows x {shape['columns']} columns",
        f"Duplicate Event IDs:   {validation_result['duplicate_events']}",
        f"Total Missing Cells:   {sum(validation_result['missing_values'].values())}",
        f"Timestamp Coverage:    {ts.get('min_timestamp')} -> {ts.get('max_timestamp')}",
        "-" * 70,
        "ENTITY CARDINALITY:",
        f"  Unique Users:        {card.get('unique_users', 0)}",
        f"  Unique Files:        {card.get('unique_files', 0)}",
        f"  Departments:         {card.get('departments', 0)}",
        f"  Action Types:        {card.get('actions', 0)}",
        "-" * 70,
        "GROUND-TRUTH LABEL DISTRIBUTION:",
        f"  Benign Baseline:     {labels.get('normal_events', 0):,} events ({1 - labels.get('anomaly_ratio', 0):.1%})",
        f"  Synthetic Anomalies: {labels.get('anomaly_events', 0):,} events ({labels.get('anomaly_ratio', 0):.1%})",
        "-" * 70,
        "VOLUME PROFILE (bytes_transferred):",
        f"  Min:                 {bytes_p.get('min_bytes', 0):,.0f} B",
        f"  Median:              {bytes_p.get('median_bytes', 0):,.0f} B ({bytes_p.get('median_bytes', 0) / (1024*1024):.2f} MB)",
        f"  Mean:                {bytes_p.get('mean_bytes', 0):,.0f} B ({bytes_p.get('mean_bytes', 0) / (1024*1024):.2f} MB)",
        f"  Max (Exfil Peak):    {bytes_p.get('max_bytes', 0):,.0f} B ({bytes_p.get('max_bytes', 0) / (1024*1024):.2f} MB)",
        "=" * 70,
    ]

    if validation_result["errors"]:
        lines.append("CRITICAL SCHEMA ERRORS:")
        for err in validation_result["errors"]:
            lines.append(f"  [X] {err}")
        lines.append("=" * 70)

    if validation_result["warnings"]:
        lines.append("WARNINGS:")
        for warn in validation_result["warnings"]:
            lines.append(f"  [!] {warn}")
        lines.append("=" * 70)

    return "\n".join(lines)


def main() -> None:
    """CLI runner for dataset quality validation."""
    parser = argparse.ArgumentParser(description="Validate access log schema and compute quality report.")
    parser.add_argument(
        "--input",
        type=str,
        default="data/raw/access_logs.csv",
        help="Path to access logs CSV (default: data/raw/access_logs.csv)",
    )

    args = parser.parse_args()
    try:
        report_data = validate_dataset(args.input)
        print(generate_quality_report(report_data))
        if not report_data["is_valid"]:
            raise SystemExit(1)
    except Exception as exc:
        print(f"[ERROR] Validation failed: {exc}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
