"""Data Preprocessing and Hygiene Module for Access Telemetry.

This module enforces schema compliance, deterministic cleaning, duplicate resolution,
timestamp parsing, categorical normalization, and generates a record-level audit
report documenting all preprocessing decisions.
"""

from __future__ import annotations

import argparse
import json 
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.data.validate_data import REQUIRED_COLUMNS, DataValidationError


def clean_access_logs(
    data: pd.DataFrame | str | Path,
    output_path: Optional[str | Path] = None,
    report_path: Optional[str | Path] = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Clean raw access logs with strict schema validation and an auditable decision trail.

    Args:
        data: Either a pandas DataFrame or path to raw access logs CSV.
        output_path: Optional destination path to save cleaned CSV.
        report_path: Optional destination path to save JSON audit report.

    Returns:
        Tuple of (cleaned_df, audit_summary_dict).

    Raises:
        DataValidationError: If mandatory schema columns are missing.
        ValueError: If input DataFrame is completely empty.
    """
    if isinstance(data, (str, Path)):
        csv_path = Path(data)
        if not csv_path.exists():
            raise FileNotFoundError(f"Source logs not found at: {csv_path}")
        df = pd.read_csv(csv_path)
    elif isinstance(data, pd.DataFrame):
        df = data.copy()
    else:
        raise TypeError(f"Expected DataFrame or path, received {type(data)}")

    initial_row_count = len(df)
    if initial_row_count == 0:
        raise ValueError("Cannot preprocess an empty DataFrame (0 rows).")

    # 1. Schema Validation (Fail loudly if required columns are absent)
    missing_columns = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_columns:
        raise DataValidationError(f"Missing required schema columns: {missing_columns}")

    audit: dict[str, Any] = {
        "initial_rows": initial_row_count,
        "final_rows": 0,
        "retained_percentage": 0.0,
        "dropped_missing_values_count": 0,
        "dropped_duplicates_count": 0,
        "dropped_duplicates_event_ids": [],
        "dropped_invalid_timestamps_count": 0,
        "dropped_negative_bytes_count": 0,
        "dropped_invalid_labels_count": 0,
        "categorical_normalization": {},
        "timestamp_bounds": {},
        "audit_decisions": [],
    }

    # Track row validity mask
    valid_mask = pd.Series(True, index=df.index)

    # 2. Duplicate Event ID Detection (Keep first occurrence, flag duplicates)
    duplicated_mask = df["event_id"].duplicated(keep="first")
    dup_count = int(duplicated_mask.sum())
    if dup_count > 0:
        dup_ids = df.loc[duplicated_mask, "event_id"].head(20).tolist()
        audit["dropped_duplicates_count"] = dup_count
        audit["dropped_duplicates_event_ids"] = dup_ids
        audit["audit_decisions"].append(
            f"Dropped {dup_count} duplicate event_id occurrences (retained first instance)."
        )
        valid_mask &= ~duplicated_mask

    # 3. Timestamp Parsing and Temporal Validity
    parsed_timestamps = pd.to_datetime(df["timestamp"], errors="coerce")
    invalid_ts_mask = parsed_timestamps.isna()
    invalid_ts_count = int(invalid_ts_mask.sum())
    if invalid_ts_count > 0:
        audit["dropped_invalid_timestamps_count"] = invalid_ts_count
        audit["audit_decisions"].append(
            f"Dropped {invalid_ts_count} records due to unparseable or corrupted timestamps."
        )
        valid_mask &= ~invalid_ts_mask

    # 4. Numeric Bounds: Non-negative Bytes Transferred
    numeric_bytes = pd.to_numeric(df["bytes_transferred"], errors="coerce")
    invalid_bytes_mask = numeric_bytes.isna() | (numeric_bytes < 0)
    invalid_bytes_count = int(invalid_bytes_mask.sum())
    if invalid_bytes_count > 0:
        audit["dropped_negative_bytes_count"] = invalid_bytes_count
        audit["audit_decisions"].append(
            f"Dropped {invalid_bytes_count} records with negative or NaN bytes_transferred."
        )
        valid_mask &= ~invalid_bytes_mask

    # 5. Label Validity: Binary Constraints {0, 1}
    sens_numeric = pd.to_numeric(df["is_sensitive"], errors="coerce")
    anom_numeric = pd.to_numeric(df["is_anomaly"], errors="coerce")
    invalid_labels_mask = (
        sens_numeric.isna()
        | ~sens_numeric.isin([0, 1])
        | anom_numeric.isna()
        | ~anom_numeric.isin([0, 1])
    )
    invalid_labels_count = int(invalid_labels_mask.sum())
    if invalid_labels_count > 0:
        audit["dropped_invalid_labels_count"] = invalid_labels_count
        audit["audit_decisions"].append(
            f"Dropped {invalid_labels_count} records with non-binary is_sensitive or is_anomaly flags."
        )
        valid_mask &= ~invalid_labels_mask

    # 6. Critical Missing Values across Core Identifiers and Categoricals
    core_cols = ["user_id", "department", "file_id", "action", "destination_type"]
    missing_core_mask = df[core_cols].isna().any(axis=1)
    missing_core_count = int((missing_core_mask & valid_mask).sum())
    if missing_core_count > 0:
        audit["dropped_missing_values_count"] = missing_core_count
        audit["audit_decisions"].append(
            f"Dropped {missing_core_count} records containing missing values in core identifier fields {core_cols}."
        )
        valid_mask &= ~missing_core_mask

    # Apply filtering
    clean_df = df[valid_mask].copy()

    # Assign parsed and validated types
    clean_df["timestamp"] = parsed_timestamps[valid_mask]
    clean_df["bytes_transferred"] = numeric_bytes[valid_mask].astype(np.int64)
    clean_df["is_sensitive"] = sens_numeric[valid_mask].astype(int)
    clean_df["is_anomaly"] = anom_numeric[valid_mask].astype(int)

    # 7. Categorical Normalization (Standardize casing and trim whitespace)
    # Never invent missing information; purely standardize strings
    norm_summary: dict[str, int] = {}
    for col in ["user_id", "file_id", "event_id"]:
        clean_df[col] = clean_df[col].astype(str).str.strip()

    clean_df["department"] = clean_df["department"].astype(str).str.strip()
    clean_df["action"] = clean_df["action"].astype(str).str.strip().str.lower()
    clean_df["destination_type"] = (
        clean_df["destination_type"].astype(str).str.strip().str.lower()
    )
    if "anomaly_scenario" in clean_df.columns:
        clean_df["anomaly_scenario"] = (
            clean_df["anomaly_scenario"].astype(str).str.strip().str.lower()
        )

    norm_summary["departments"] = int(clean_df["department"].nunique())
    norm_summary["actions"] = int(clean_df["action"].nunique())
    norm_summary["destination_types"] = int(clean_df["destination_type"].nunique())
    audit["categorical_normalization"] = norm_summary

    # Chronological Ordering Guarantee
    clean_df.sort_values(by=["timestamp", "event_id"], ascending=True, inplace=True)
    clean_df.reset_index(drop=True, inplace=True)

    # Final Audit Summary Calculations
    final_row_count = len(clean_df)
    audit["final_rows"] = final_row_count
    audit["retained_percentage"] = (
        round((final_row_count / initial_row_count) * 100, 2)
        if initial_row_count > 0
        else 0.0
    )
    if final_row_count > 0:
        audit["timestamp_bounds"] = {
            "start": str(clean_df["timestamp"].min()),
            "end": str(clean_df["timestamp"].max()),
        }

    # Save outputs if paths provided
    if output_path is not None:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        # Format timestamp to ISO string for portability
        export_df = clean_df.copy()
        export_df["timestamp"] = export_df["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
        export_df.to_csv(out, index=False)

    if report_path is not None:
        rep = Path(report_path)
        rep.parent.mkdir(parents=True, exist_ok=True)
        with open(rep, "w", encoding="utf-8") as f:
            json.dump(audit, f, indent=2)

    return clean_df, audit


def main() -> None:
    """CLI entry point for data preprocessing."""
    parser = argparse.ArgumentParser(
        description="Clean raw access telemetry and generate an audit report."
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/raw/access_logs.csv",
        help="Path to raw access logs CSV (default: data/raw/access_logs.csv)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/processed/clean_access_logs.csv",
        help="Output path for cleaned CSV (default: data/processed/clean_access_logs.csv)",
    )
    parser.add_argument(
        "--report",
        type=str,
        default="reports/data_quality_report.json",
        help="Output path for JSON audit report (default: reports/data_quality_report.json)",
    )

    args = parser.parse_args()

    print(f"[*] Preprocessing raw telemetry from: {args.input}")
    clean_df, audit = clean_access_logs(
        data=args.input,
        output_path=args.output,
        report_path=args.report,
    )

    print(f"[+] Cleaning complete!")
    print(f"    Initial rows:       {audit['initial_rows']:,}")
    print(f"    Retained rows:      {audit['final_rows']:,} ({audit['retained_percentage']}%)")
    print(f"    Dropped duplicates: {audit['dropped_duplicates_count']}")
    print(f"    Dropped timestamps: {audit['dropped_invalid_timestamps_count']}")
    print(f"    Dropped byte errors:{audit['dropped_negative_bytes_count']}")
    print(f"    Cleaned logs saved: {args.output}")
    print(f"    Audit report saved: {args.report}")


if __name__ == "__main__":
    main()
