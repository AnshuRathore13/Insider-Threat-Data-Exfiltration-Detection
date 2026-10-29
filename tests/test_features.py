"""Unit and integration tests for behavioral and temporal feature engineering."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
 
from src.features.build_features import (
    build_full_feature_pipeline,
    extract_behavioral_features,
    extract_temporal_features,
)


@pytest.fixture
def sequence_logs() -> pd.DataFrame:
    """Deterministic 4-event sequence for temporal and behavioral verification."""
    return pd.DataFrame([
        {
            "event_id": "EVT-001",
            "user_id": "USR-101",
            "timestamp": "2026-01-05 09:00:00",
            "department": "Engineering",
            "file_id": "FILE-001",
            "action": "read",
            "bytes_transferred": 1000,
            "destination_type": "internal_server",
            "is_sensitive": 0,
            "is_anomaly": 0,
            "anomaly_scenario": "benign",
        },
        {
            "event_id": "EVT-002",
            "user_id": "USR-101",
            "timestamp": "2026-01-05 11:30:00",
            "department": "Engineering",
            "file_id": "FILE-002",
            "action": "download",
            "bytes_transferred": 2000,
            "destination_type": "cloud_drive",
            "is_sensitive": 1,
            "is_anomaly": 0,
            "anomaly_scenario": "benign",
        },
        {
            "event_id": "EVT-003",
            "user_id": "USR-101",
            "timestamp": "2026-01-05 23:15:00",  # After-hours
            "department": "Engineering",
            "file_id": "FILE-001",
            "action": "upload",
            "bytes_transferred": 50000,  # Massive spike
            "destination_type": "usb_drive",
            "is_sensitive": 0,
            "is_anomaly": 1,
            "anomaly_scenario": "volume_exfiltration",
        },
        {
            "event_id": "EVT-004",
            "user_id": "USR-102",  # Brand new cold-start user
            "timestamp": "2026-01-06 10:00:00",
            "department": "Finance",
            "file_id": "FILE-099",
            "action": "read",
            "bytes_transferred": 500,
            "destination_type": "internal_server",
            "is_sensitive": 0,
            "is_anomaly": 0,
            "anomaly_scenario": "benign",
        },
    ])


def test_reproducible_feature_generation(sequence_logs: pd.DataFrame) -> None:
    """Verify that multiple feature builds from identical inputs yield exact equality."""
    f1, t1 = build_full_feature_pipeline(sequence_logs)
    f2, t2 = build_full_feature_pipeline(sequence_logs)

    pd.testing.assert_frame_equal(f1, f2)
    pd.testing.assert_frame_equal(t1, t2)


def test_feature_target_row_alignment(sequence_logs: pd.DataFrame) -> None:
    """Verify row count and order strictly match between features and targets."""
    features, targets = build_full_feature_pipeline(sequence_logs)
    assert len(features) == len(targets) == len(sequence_logs)
    assert list(targets["event_id"]) == ["EVT-001", "EVT-002", "EVT-003", "EVT-004"]


def test_no_target_leakage(sequence_logs: pd.DataFrame) -> None:
    """Verify prohibited identifiers and labels are completely absent from features."""
    features, targets = build_full_feature_pipeline(sequence_logs)
    prohibited = {"is_anomaly", "anomaly_scenario", "event_id", "user_id", "file_id", "timestamp"}
    assert prohibited.isdisjoint(features.columns)

    # Targets must hold the ground-truth labels
    assert "is_anomaly" in targets.columns
    assert "event_id" in targets.columns


def test_historical_features_never_use_future_events(sequence_logs: pd.DataFrame) -> None:
    """Verify that behavioral metrics strictly utilize past events."""
    beh = extract_behavioral_features(sequence_logs, window_days=7)

    # Event 1: First event for USR-101. Prior stats MUST be 0.
    assert beh.loc[0, "user_prior_event_count"] == 0
    assert beh.loc[0, "user_prior_distinct_files"] == 0
    assert beh.loc[0, "user_prior_sensitive_count"] == 0
    assert beh.loc[0, "user_prior_total_bytes"] == 0.0

    # Event 2: Second event for USR-101. Prior count MUST be 1.
    assert beh.loc[1, "user_prior_event_count"] == 1
    assert beh.loc[1, "user_prior_distinct_files"] == 1
    assert beh.loc[1, "user_prior_total_bytes"] == 1000.0

    # Event 3: Third event for USR-101. Prior count MUST be 2.
    assert beh.loc[2, "user_prior_event_count"] == 2
    assert beh.loc[2, "user_prior_distinct_files"] == 2  # FILE-001 and FILE-002
    assert beh.loc[2, "user_prior_sensitive_count"] == 1  # From Event 2
    assert beh.loc[2, "user_prior_total_bytes"] == 3000.0  # 1000 + 2000

    # Crucial assertion: Event 1 or 2 NEVER see the 50,000 bytes from Event 3!
    assert beh.loc[0, "user_prior_total_bytes"] < 50000
    assert beh.loc[1, "user_prior_total_bytes"] < 50000


def test_cold_start_new_user(sequence_logs: pd.DataFrame) -> None:
    """Verify new user (USR-102 in Event 4) receives safe 0s without errors."""
    beh = extract_behavioral_features(sequence_logs, window_days=7)
    row_user2 = beh.iloc[3]

    assert row_user2["user_prior_event_count"] == 0
    assert row_user2["user_prior_distinct_files"] == 0
    assert row_user2["user_prior_total_bytes"] == 0.0
    assert row_user2["user_bytes_deviation"] == 0.0
    assert row_user2["user_unusual_hour_ratio"] == 0.0


def test_zero_variance_deviation_handling() -> None:
    """Verify zero standard deviation in historical transfers returns 0 deviation."""
    zero_var_df = pd.DataFrame([
        {
            "event_id": f"EVT-{i}",
            "user_id": "USR-CONST",
            "timestamp": f"2026-01-05 0{i}:00:00",
            "file_id": "FILE-1",
            "bytes_transferred": 1000,  # Constant volume, std=0
            "is_sensitive": 0,
        }
        for i in range(1, 4)
    ])

    beh = extract_behavioral_features(zero_var_df)
    # Deviation should be safely 0.0, avoiding ZeroDivisionError or NaN
    assert not np.isnan(beh["user_bytes_deviation"].iloc[-1])
    assert beh["user_bytes_deviation"].iloc[-1] == 0.0


def test_temporal_features(sequence_logs: pd.DataFrame) -> None:
    """Verify hour_of_day, weekend, and after-hours flags."""
    temp = extract_temporal_features(sequence_logs)
    # Event 1: 09:00 -> not after hours
    assert temp.loc[0, "hour_of_day"] == 9
    assert temp.loc[0, "is_after_hours"] == 0
    # Event 3: 23:15 -> after hours
    assert temp.loc[2, "hour_of_day"] == 23
    assert temp.loc[2, "is_after_hours"] == 1


def test_empty_dataframe_returns_empty() -> None:
    """Verify feature pipeline handles empty input gracefully."""
    empty_df = pd.DataFrame(columns=["timestamp", "user_id", "file_id", "bytes_transferred", "action", "destination_type", "department", "is_sensitive", "is_anomaly", "event_id"])
    f, t = build_full_feature_pipeline(empty_df)
    assert len(f) == 0
    assert len(t) == 0
