"""Unit and integration tests for synthetic access log generation and schema validation.

Verifies:
1. Deterministic reproducibility given the same seed.
2. Complete schema coverage of all required columns.
3. Timestamp format validity and chronological sanity.
4. Uniqueness of event IDs across the entire dataset.
5. Ground-truth label validity and expected anomaly ratio bounds.
6. Value range sanity (bytes_transferred >= 0, valid categorical domains).
7. Schema validator detection of missing columns and duplicates.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
import tempfile
import pytest

from src.data.generate_data import (
    DEPARTMENTS,
    DESTINATION_TYPES,
    generate_synthetic_access_logs,
    save_access_logs,
)
from src.data.validate_data import REQUIRED_COLUMNS, validate_dataset


@pytest.fixture(scope="module")
def sample_records() -> list[dict]:
    """Generate a modest batch of synthetic records for fast, reliable unit testing."""
    return generate_synthetic_access_logs(num_events=500, seed=42, anomaly_ratio=0.05)


def test_reproducibility() -> None:
    """Verify that identical random seeds produce identical event sequences."""
    run1 = generate_synthetic_access_logs(num_events=300, seed=123)
    run2 = generate_synthetic_access_logs(num_events=300, seed=123)

    assert len(run1) == len(run2) == 300
    assert run1 == run2, "Generator output was not deterministic across identical seeds."

    # Different seeds should produce divergent streams
    run3 = generate_synthetic_access_logs(num_events=300, seed=999)
    assert run1 != run3, "Different seeds unexpectedly produced identical outputs."


def test_required_columns(sample_records: list[dict]) -> None:
    """Verify that all required defensive schema columns are present in each record."""
    first_record = sample_records[0]
    for col in REQUIRED_COLUMNS:
        assert col in first_record, f"Missing required column in generated record: {col}"


def test_unique_event_ids(sample_records: list[dict]) -> None:
    """Ensure every event receives a strictly unique event_id with no collisions."""
    event_ids = [r["event_id"] for r in sample_records]
    assert len(event_ids) == len(set(event_ids)), "Found duplicate event_id values."
    # Ensure standard format prefix
    assert all(eid.startswith("EVT-") for eid in event_ids)


def test_valid_timestamps(sample_records: list[dict]) -> None:
    """Verify all timestamps parse as valid ISO datetime strings within expected bounds."""
    timestamps = [datetime.strptime(r["timestamp"], "%Y-%m-%d %H:%M:%S") for r in sample_records]
    assert len(timestamps) == len(sample_records)

    # Verify chronological ordering
    for i in range(len(timestamps) - 1):
        assert timestamps[i] <= timestamps[i + 1], f"Timestamps out of chronological order at index {i}"


def test_valid_ground_truth_labels(sample_records: list[dict]) -> None:
    """Ensure is_anomaly and is_sensitive flags are strictly binary {0, 1}."""
    for r in sample_records:
        assert r["is_anomaly"] in (0, 1), f"Invalid is_anomaly value: {r['is_anomaly']}"
        assert r["is_sensitive"] in (0, 1), f"Invalid is_sensitive value: {r['is_sensitive']}"

    # Verify anomaly distribution roughly matches requested 5%
    anomaly_count = sum(r["is_anomaly"] for r in sample_records)
    ratio = anomaly_count / len(sample_records)
    assert 0.02 <= ratio <= 0.10, f"Anomaly ratio {ratio:.3f} deviated significantly from 0.05"


def test_value_range_and_domain_sanity(sample_records: list[dict]) -> None:
    """Verify bytes transferred are positive and categorical fields use recognized domains."""
    depts_set = set(DEPARTMENTS)
    dest_set = set(DESTINATION_TYPES)

    for r in sample_records:
        assert r["bytes_transferred"] > 0, "bytes_transferred must be strictly positive"
        assert r["department"] in depts_set, f"Unknown department: {r['department']}"
        assert r["destination_type"] in dest_set, f"Unknown destination: {r['destination_type']}"


def test_schema_validator_on_generated_data() -> None:
    """Integration test: generate data, save to disk, and pass through validate_dataset."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_csv = Path(tmp_dir) / "test_logs.csv"
        records = generate_synthetic_access_logs(num_events=200, seed=42)
        save_access_logs(records, tmp_csv)

        report = validate_dataset(tmp_csv)
        assert report["is_valid"] is True
        assert len(report["errors"]) == 0
        assert report["shape"]["rows"] == 200
        assert report["duplicate_events"] == 0


def test_schema_validator_catches_errors() -> None:
    """Verify validator flags corrupted schema with missing columns and invalid labels."""
    import pandas as pd

    corrupt_df = pd.DataFrame([
        {
            "event_id": "EVT-001",
            "user_id": "USR-0001",
            # missing timestamp
            "department": "Engineering",
            "file_id": "FILE-0001",
            "action": "read",
            "bytes_transferred": -500,  # invalid negative bytes
            "destination_type": "internal_server",
            "is_sensitive": 99,  # invalid non-binary
            "is_anomaly": 42,  # invalid non-binary
        }
    ])

    report = validate_dataset(corrupt_df)
    assert report["is_valid"] is False
    assert any("Missing required schema columns" in err for err in report["errors"])
    assert any("negative values in bytes_transferred" in err for err in report["errors"])
    assert any("is_anomaly contains invalid non-binary" in err for err in report["errors"])
