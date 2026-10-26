"""Synthetic Access Log Generator for Insider Threat & Data Exfiltration Detection.

This module simulates realistic enterprise file-access telemetry, establishing
both benign behavioral baselines and controlled synthetic threat scenarios:
1. Normal business-hour operations aligned with user departments.
2. Unusual after-hours access (time-of-day anomalies).
3. Bulk sensitive-file access spikes (cardinality / reconnaissance anomalies).
4. Massive transfer volumes (data staging / exfiltration anomalies).
5. Unauthorized destination channels (USB, personal cloud, external email).
6. Cross-department perimeter probing (privilege abuse).

IMPORTANT ARCHITECTURAL & INTERVIEW NOTE:
- `is_anomaly` is a synthetic ground-truth evaluation label ONLY.
- It represents synthetic rule-generated truth for benchmarking detection models.
- It MUST NEVER be used as an input feature in any feature matrix or model training.
- In real-world enterprise environments, ground-truth insider threat labels are
  typically unavailable or heavily biased. Rule-based synthetic labels serve as
  controlled evaluation benchmarks, not real-world confirmed threats.
- This dataset is entirely deterministic, pseudorandom, and contains NO real PII
  or enterprise credentials.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timedelta
import math
from pathlib import Path
import random
from typing import Any, Dict, List, Optional

try:
    import pandas as pd
except ImportError:  # pragma: no cover
    pd = None  # type: ignore


# Constant metadata dictionaries for realistic corporate simulation
DEPARTMENTS = [
    "Engineering",
    "Finance",
    "HR",
    "Sales",
    "Legal",
    "IT_Admin",
    "Marketing",
]

NORMAL_ACTIONS = ["read", "write", "download", "modify", "preview"]
ANOMALY_ACTIONS = ["download", "upload", "read", "delete"]

DESTINATION_TYPES = [
    "internal_server",
    "cloud_drive",
    "local_storage",
    "usb_drive",
    "personal_cloud",
    "external_email",
]

# Destination weights for normal users (mostly internal and approved cloud storage)
NORMAL_DESTINATION_WEIGHTS = [0.55, 0.35, 0.08, 0.01, 0.005, 0.005]


def _build_synthetic_entities(
    num_users: int = 50,
    num_files: int = 250,
    seed: int = 42,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    """Create deterministic users and file repositories.

    Args:
        num_users: Number of synthetic employees.
        num_files: Number of synthetic corporate files.
        seed: Random seed for deterministic generation.

    Returns:
        Tuple of (users_dict, files_dict).
    """
    rng = random.Random(seed)

    # Generate synthetic employees
    users: dict[str, dict[str, Any]] = {}
    for i in range(1, num_users + 1):
        uid = f"USR-{i:04d}"
        dept = rng.choice(DEPARTMENTS)
        # Normal daily typical volume base (e.g. 50 KB to 5 MB per transaction)
        mean_bytes = rng.randint(40_000, 2_000_000)
        users[uid] = {
            "user_id": uid,
            "department": dept,
            "mean_bytes": mean_bytes,
            "is_rogue_candidate": (i % 7 == 0),  # Designated candidate for anomaly scenarios
        }

    # Generate synthetic files with sensitivity classifications
    files: dict[str, dict[str, Any]] = {}
    for j in range(1, num_files + 1):
        fid = f"FILE-{j:04d}"
        owner_dept = rng.choice(DEPARTMENTS)
        # ~25% of enterprise files are sensitive / confidential
        is_sensitive = 1 if rng.random() < 0.25 else 0
        base_size = rng.randint(20_000, 15_000_000)  # 20 KB to 15 MB base file size
        files[fid] = {
            "file_id": fid,
            "department": owner_dept,
            "is_sensitive": is_sensitive,
            "base_size": base_size,
        }

    return users, files


def _random_normal_timestamp(
    start_date: datetime,
    days_range: int,
    rng: random.Random,
) -> datetime:
    """Generate a realistic timestamp during standard business days/hours.

    85% Mon-Fri between 08:30 and 18:00, 15% mild after-hours/weekends.
    """
    day_offset = rng.randint(0, days_range - 1)
    base_day = start_date + timedelta(days=day_offset)

    if rng.random() < 0.85:
        # Standard work hours (08:30 to 18:00)
        hour = rng.randint(8, 17)
        minute = rng.randint(0, 59)
        second = rng.randint(0, 59)
    else:
        # Mild evening/morning work
        hour = rng.choice([7, 18, 19, 20, 21])
        minute = rng.randint(0, 59)
        second = rng.randint(0, 59)

    return base_day.replace(hour=hour, minute=minute, second=second)


def _random_anomaly_timestamp(
    start_date: datetime,
    days_range: int,
    rng: random.Random,
) -> datetime:
    """Generate an unusual after-hours timestamp (01:00 AM - 04:59 AM)."""
    day_offset = rng.randint(0, days_range - 1)
    base_day = start_date + timedelta(days=day_offset)
    hour = rng.randint(1, 4)
    minute = rng.randint(0, 59)
    second = rng.randint(0, 59)
    return base_day.replace(hour=hour, minute=minute, second=second)


def generate_synthetic_access_logs(
    num_events: int = 20000,
    seed: int = 42,
    anomaly_ratio: float = 0.05,
    start_date: Optional[datetime] = None,
    days_range: int = 60,
) -> list[dict[str, Any]]:
    """Generate deterministic, realistic synthetic access telemetry.

    Simulates normal enterprise file access alongside 5 distinct insider threat
    patterns to support unsupervised and supervised anomaly detection research.

    Args:
        num_events: Total number of log rows to generate.
        seed: Random seed ensuring 100% reproducibility across environments.
        anomaly_ratio: Proportion of events that reflect insider threat anomalies.
        start_date: Starting date for simulation timestamps. Defaults to 2026-01-05.
        days_range: Time span in days for access logs.

    Returns:
        List of event dictionary records sorted chronologically.
    """
    if start_date is None:
        start_date = datetime(2026, 1, 5, 0, 0, 0)

    rng = random.Random(seed)
    users, files = _build_synthetic_entities(num_users=60, num_files=300, seed=seed)

    user_ids = list(users.keys())
    file_ids = list(files.keys())

    # Pre-filter files for quick lookup
    sensitive_file_ids = [fid for fid, f in files.items() if f["is_sensitive"] == 1]
    dept_files: dict[str, list[str]] = {dept: [] for dept in DEPARTMENTS}
    for fid, f in files.items():
        dept_files[f["department"]].append(fid)

    num_anomalies = int(num_events * anomaly_ratio)
    num_normals = num_events - num_anomalies

    records: list[dict[str, Any]] = []

    # ---------------------------------------------------------
    # 1. Generate Normal / Benign Baseline Events
    # ---------------------------------------------------------
    for _ in range(num_normals):
        user = users[rng.choice(user_ids)]
        user_dept = user["department"]

        # Normal users primarily access their own department files (85%)
        if rng.random() < 0.85 and dept_files[user_dept]:
            file_id = rng.choice(dept_files[user_dept])
        else:
            file_id = rng.choice(file_ids)

        file_info = files[file_id]
        ts = _random_normal_timestamp(start_date, days_range, rng)
        action = rng.choice(NORMAL_ACTIONS)

        # Normal file transfer sizes: log-normal around user's typical volume
        # Log-normal with moderate variance
        mu = math.log(max(10_000, user["mean_bytes"]))
        raw_bytes = int(rng.lognormvariate(mu=mu, sigma=0.5))
        bytes_transferred = max(512, min(raw_bytes, 45_000_000))  # Capped at 45 MB normal

        # Destination type based on normal corporate weights
        destination_type = rng.choices(
            DESTINATION_TYPES,
            weights=NORMAL_DESTINATION_WEIGHTS,
            k=1,
        )[0]

        records.append({
            "user_id": user["user_id"],
            "timestamp": ts,
            "department": user_dept,
            "file_id": file_id,
            "action": action,
            "bytes_transferred": bytes_transferred,
            "destination_type": destination_type,
            "is_sensitive": file_info["is_sensitive"],
            "is_anomaly": 0,
            "anomaly_scenario": "benign",
        })

    # ---------------------------------------------------------
    # 2. Generate Controlled Anomaly Scenarios (~5%)
    # ---------------------------------------------------------
    # Scenario breakdown:
    # A. Unusual after-hours access (25% of anomalies)
    # B. Bulk sensitive-file access (25% of anomalies)
    # C. Massive data volume exfiltration (25% of anomalies)
    # D. Prohibited external destination (USB / Personal Cloud / External Email) (15%)
    # E. Cross-department privilege / perimeter probing (10%)

    scenario_counts = {
        "after_hours": int(num_anomalies * 0.25),
        "bulk_sensitive": int(num_anomalies * 0.25),
        "volume_exfiltration": int(num_anomalies * 0.25),
        "prohibited_destination": int(num_anomalies * 0.15),
        "cross_dept_probing": num_anomalies - (
            int(num_anomalies * 0.25) * 3 + int(num_anomalies * 0.15)
        ),
    }

    # Scenario A: After-Hours Access
    for _ in range(scenario_counts["after_hours"]):
        user = users[rng.choice(user_ids)]
        file_id = rng.choice(sensitive_file_ids if rng.random() < 0.7 else file_ids)
        file_info = files[file_id]
        ts = _random_anomaly_timestamp(start_date, days_range, rng)
        action = rng.choice(["download", "read", "upload"])
        bytes_transferred = rng.randint(500_000, 50_000_000)
        destination_type = rng.choice(["local_storage", "cloud_drive", "usb_drive"])

        records.append({
            "user_id": user["user_id"],
            "timestamp": ts,
            "department": user["department"],
            "file_id": file_id,
            "action": action,
            "bytes_transferred": bytes_transferred,
            "destination_type": destination_type,
            "is_sensitive": file_info["is_sensitive"],
            "is_anomaly": 1,
            "anomaly_scenario": "after_hours_access",
        })

    # Scenario B: Bulk Sensitive-File Access (Spike in Cardinality)
    # Clustered events by rogue employees
    rogue_users = [u for u in users.values() if u["is_rogue_candidate"]]
    burst_user = rng.choice(rogue_users if rogue_users else list(users.values()))
    burst_base_ts = start_date + timedelta(days=rng.randint(10, days_range - 5), hours=rng.randint(9, 16))

    for idx in range(scenario_counts["bulk_sensitive"]):
        file_id = rng.choice(sensitive_file_ids)
        file_info = files[file_id]
        # Occurring within a rapid 15-minute window
        ts = burst_base_ts + timedelta(seconds=idx * rng.randint(2, 20))
        action = rng.choice(["download", "read"])
        bytes_transferred = rng.randint(200_000, 12_000_000)
        destination_type = rng.choice(["local_storage", "usb_drive", "cloud_drive"])

        records.append({
            "user_id": burst_user["user_id"],
            "timestamp": ts,
            "department": burst_user["department"],
            "file_id": file_id,
            "action": action,
            "bytes_transferred": bytes_transferred,
            "destination_type": destination_type,
            "is_sensitive": 1,
            "is_anomaly": 1,
            "anomaly_scenario": "bulk_sensitive_access",
        })

    # Scenario C: Massive Volume Exfiltration
    for _ in range(scenario_counts["volume_exfiltration"]):
        user = users[rng.choice(user_ids)]
        file_id = rng.choice(sensitive_file_ids if rng.random() < 0.8 else file_ids)
        file_info = files[file_id]
        ts = _random_normal_timestamp(start_date, days_range, rng)
        action = rng.choice(["download", "upload"])
        # Huge transfers: 150 MB to 4.5 GB (far beyond normal 50 MB upper ceiling)
        bytes_transferred = rng.randint(150_000_000, 4_500_000_000)
        destination_type = rng.choice(["usb_drive", "personal_cloud", "cloud_drive"])

        records.append({
            "user_id": user["user_id"],
            "timestamp": ts,
            "department": user["department"],
            "file_id": file_id,
            "action": action,
            "bytes_transferred": bytes_transferred,
            "destination_type": destination_type,
            "is_sensitive": file_info["is_sensitive"],
            "is_anomaly": 1,
            "anomaly_scenario": "volume_exfiltration",
        })

    # Scenario D: Prohibited Exfiltration Destination (USB, Personal Cloud, External Email)
    for _ in range(scenario_counts["prohibited_destination"]):
        user = users[rng.choice(user_ids)]
        file_id = rng.choice(sensitive_file_ids if rng.random() < 0.75 else file_ids)
        file_info = files[file_id]
        ts = _random_normal_timestamp(start_date, days_range, rng)
        action = rng.choice(["download", "upload"])
        bytes_transferred = rng.randint(1_000_000, 100_000_000)
        destination_type = rng.choice(["usb_drive", "personal_cloud", "external_email"])

        records.append({
            "user_id": user["user_id"],
            "timestamp": ts,
            "department": user["department"],
            "file_id": file_id,
            "action": action,
            "bytes_transferred": bytes_transferred,
            "destination_type": destination_type,
            "is_sensitive": file_info["is_sensitive"],
            "is_anomaly": 1,
            "anomaly_scenario": "prohibited_destination",
        })

    # Scenario E: Cross-Department Privilege / Perimeter Probing
    for _ in range(scenario_counts["cross_dept_probing"]):
        user = users[rng.choice(user_ids)]
        user_dept = user["department"]
        # Intentionally pick files belonging to other departments (e.g. Sales user probing HR/Finance/Engineering)
        other_depts = [d for d in DEPARTMENTS if d != user_dept]
        target_dept = rng.choice(other_depts)
        target_files = dept_files.get(target_dept, file_ids)
        file_id = rng.choice(target_files if target_files else file_ids)
        file_info = files[file_id]

        ts = _random_normal_timestamp(start_date, days_range, rng)
        action = rng.choice(["read", "download", "modify"])
        bytes_transferred = rng.randint(100_000, 15_000_000)
        destination_type = rng.choice(["internal_server", "local_storage"])

        records.append({
            "user_id": user["user_id"],
            "timestamp": ts,
            "department": user_dept,
            "file_id": file_id,
            "action": action,
            "bytes_transferred": bytes_transferred,
            "destination_type": destination_type,
            "is_sensitive": file_info["is_sensitive"],
            "is_anomaly": 1,
            "anomaly_scenario": "cross_dept_probing",
        })

    # Sort all events chronologically
    records.sort(key=lambda r: r["timestamp"])

    # Assign sequential, strictly unique event IDs and format timestamps
    formatted_records: list[dict[str, Any]] = []
    for idx, r in enumerate(records, start=1):
        formatted_records.append({
            "event_id": f"EVT-{idx:08d}",
            "user_id": r["user_id"],
            "timestamp": r["timestamp"].strftime("%Y-%m-%d %H:%M:%S"),
            "department": r["department"],
            "file_id": r["file_id"],
            "action": r["action"],
            "bytes_transferred": r["bytes_transferred"],
            "destination_type": r["destination_type"],
            "is_sensitive": r["is_sensitive"],
            "is_anomaly": r["is_anomaly"],
            # Keep scenario tag for initial ground-truth validation & documentation
            "anomaly_scenario": r["anomaly_scenario"],
        })

    return formatted_records


def generate_to_dataframe(
    num_events: int = 20000,
    seed: int = 42,
    anomaly_ratio: float = 0.05,
) -> Any:
    """Generate synthetic events directly as a pandas DataFrame.

    Requires pandas to be installed in the active environment.
    """
    if pd is None:
        raise ImportError("pandas is required for generate_to_dataframe. Install pandas first.")
    records = generate_synthetic_access_logs(
        num_events=num_events,
        seed=seed,
        anomaly_ratio=anomaly_ratio,
    )
    df = pd.DataFrame(records)
    # Parse timestamp as datetime64
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def save_access_logs(
    records: list[dict[str, Any]],
    output_path: str | Path,
) -> Path:
    """Write generated log records to CSV on disk.

    Args:
        records: List of event dictionary records.
        output_path: Target path for the CSV file.

    Returns:
        Resolved Path of written file.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if not records:
        raise ValueError("Cannot save empty records list.")

    fieldnames = list(records[0].keys())

    with open(path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    return path


def main() -> None:
    """CLI entrypoint for reproducible synthetic data generation."""
    parser = argparse.ArgumentParser(
        description="Generate deterministic synthetic access logs for insider threat detection.",
    )
    parser.add_argument(
        "--num-events",
        type=int,
        default=20000,
        help="Number of access log records to generate (default: 20000)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic generation (default: 42)",
    )
    parser.add_argument(
        "--anomaly-ratio",
        type=float,
        default=0.05,
        help="Proportion of events labeled as anomaly scenarios (default: 0.05)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/raw/access_logs.csv",
        help="Output CSV filepath (default: data/raw/access_logs.csv)",
    )

    args = parser.parse_args()

    print(f"[*] Generating {args.num_events:,} synthetic access events (seed={args.seed}, anomaly_ratio={args.anomaly_ratio:.1%})...")
    records = generate_synthetic_access_logs(
        num_events=args.num_events,
        seed=args.seed,
        anomaly_ratio=args.anomaly_ratio,
    )

    out_path = save_access_logs(records, args.output)
    file_size_mb = out_path.stat().st_size / (1024 * 1024)

    print(f"[+] Successfully saved {len(records):,} records to: {out_path} ({file_size_mb:.2f} MB)")
    print("[!] Security Note: 'is_anomaly' is a synthetic ground-truth benchmark label.")
    print("    Never expose 'is_anomaly' or 'anomaly_scenario' to model feature sets.")


if __name__ == "__main__":
    main()
