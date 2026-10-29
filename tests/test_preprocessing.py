"""Unit tests for telemetry cleaning and data preprocessing module."""
 
from __future__ import annotations

import pandas as pd
import pytest

from src.data.preprocess import clean_access_logs
from src.data.validate_data import DataValidationError


@pytest.fixture
def base_valid_data() -> pd.DataFrame:
    """Minimal valid DataFrame satisfying schema requirements."""
    return pd.DataFrame([
        {
            "event_id": "EVT-001",
            "user_id": "USR-100",
            "timestamp": "2026-01-05 09:00:00",
            "department": "Engineering",
            "file_id": "FILE-010",
            "action": "read",
            "bytes_transferred": 1024,
            "destination_type": "internal_server",
            "is_sensitive": 0,
            "is_anomaly": 0,
            "anomaly_scenario": "benign",
        },
        {
            "event_id": "EVT-002",
            "user_id": "USR-100",
            "timestamp": "2026-01-05 10:15:00",
            "department": "Engineering",
            "file_id": "FILE-020",
            "action": "download",
            "bytes_transferred": 2048,
            "destination_type": "cloud_drive",
            "is_sensitive": 1,
            "is_anomaly": 0,
            "anomaly_scenario": "benign",
        },
    ])


def test_missing_required_columns_raises(base_valid_data: pd.DataFrame) -> None:
    """Verify that omitting a mandatory column raises DataValidationError."""
    corrupt = base_valid_data.drop(columns=["timestamp"])
    with pytest.raises(DataValidationError) as exc_info:
        clean_access_logs(corrupt)
    assert "Missing required schema columns" in str(exc_info.value)


def test_empty_dataframe_raises() -> None:
    """Verify that passing an empty DataFrame raises ValueError."""
    empty_df = pd.DataFrame(columns=["event_id", "timestamp", "user_id"])
    with pytest.raises(ValueError) as exc_info:
        clean_access_logs(empty_df)
    assert "Cannot preprocess an empty DataFrame" in str(exc_info.value)


def test_invalid_timestamps_handled(base_valid_data: pd.DataFrame) -> None:
    """Verify corrupted timestamps are dropped and recorded in audit summary."""
    corrupt = base_valid_data.copy()
    corrupt.loc[1, "timestamp"] = "not-a-valid-timestamp"

    clean_df, audit = clean_access_logs(corrupt)
    assert len(clean_df) == 1
    assert clean_df.iloc[0]["event_id"] == "EVT-001"
    assert audit["dropped_invalid_timestamps_count"] == 1
    assert any("unparseable or corrupted timestamps" in d for d in audit["audit_decisions"])


def test_duplicate_event_ids_dropped(base_valid_data: pd.DataFrame) -> None:
    """Verify duplicate event IDs are detected and first occurrence is preserved."""
    duplicated = pd.concat([base_valid_data, base_valid_data.iloc[[0]]], ignore_index=True)
    assert len(duplicated) == 3

    clean_df, audit = clean_access_logs(duplicated)
    assert len(clean_df) == 2
    assert audit["dropped_duplicates_count"] == 1
    assert "EVT-001" in audit["dropped_duplicates_event_ids"]


def test_negative_and_nan_bytes_dropped(base_valid_data: pd.DataFrame) -> None:
    """Verify negative and NaN transfer sizes are dropped and logged."""
    corrupt = base_valid_data.copy()
    corrupt.loc[0, "bytes_transferred"] = -500
    corrupt.loc[1, "bytes_transferred"] = float("nan")

    # Add a valid third row
    valid_row = base_valid_data.iloc[[0]].copy()
    valid_row["event_id"] = "EVT-003"
    valid_row["bytes_transferred"] = 5000
    df = pd.concat([corrupt, valid_row], ignore_index=True)

    clean_df, audit = clean_access_logs(df)
    assert len(clean_df) == 1
    assert clean_df.iloc[0]["event_id"] == "EVT-003"
    assert audit["dropped_negative_bytes_count"] == 2


def test_invalid_synthetic_labels_dropped(base_valid_data: pd.DataFrame) -> None:
    """Verify records with non-binary is_sensitive or is_anomaly are dropped."""
    corrupt = base_valid_data.copy()
    corrupt.loc[0, "is_anomaly"] = 99
    corrupt.loc[1, "is_sensitive"] = -1

    valid_row = base_valid_data.iloc[[0]].copy()
    valid_row["event_id"] = "EVT-003"
    valid_row["is_anomaly"] = 1
    valid_row["is_sensitive"] = 1
    df = pd.concat([corrupt, valid_row], ignore_index=True)

    clean_df, audit = clean_access_logs(df)
    assert len(clean_df) == 1
    assert clean_df.iloc[0]["event_id"] == "EVT-003"
    assert audit["dropped_invalid_labels_count"] == 2


def test_categorical_normalization(base_valid_data: pd.DataFrame) -> None:
    """Verify casing is normalized and surrounding whitespace is stripped."""
    messy = base_valid_data.copy()
    messy.loc[0, "action"] = "  DOWNLOAD  "
    messy.loc[0, "destination_type"] = " Cloud_Drive "
    messy.loc[0, "department"] = "  Engineering  "

    clean_df, audit = clean_access_logs(messy)
    assert clean_df.iloc[0]["action"] == "download"
    assert clean_df.iloc[0]["destination_type"] == "cloud_drive"
    assert clean_df.iloc[0]["department"] == "Engineering"
    assert audit["categorical_normalization"]["departments"] == 1


def test_missing_core_identifier_dropped(base_valid_data: pd.DataFrame) -> None:
    """Verify missing user_id or file_id drops record cleanly."""
    corrupt = base_valid_data.copy()
    corrupt.loc[0, "user_id"] = None

    clean_df, audit = clean_access_logs(corrupt)
    assert len(clean_df) == 1
    assert audit["dropped_missing_values_count"] == 1
