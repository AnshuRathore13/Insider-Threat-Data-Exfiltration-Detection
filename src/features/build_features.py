"""Behavioral and Temporal Feature Engineering Pipeline.

Transforms cleaned access telemetry into high-dimensional feature matrices while
strictly preventing temporal lookahead leakage and target leakage.

Key Principles: 
1. ONE ROW PER EVENT: Stable row alignment with `targets.csv`.
2. STRICTLY PAST-ONLY: User historical features (counts, velocity, deviations)
   only incorporate events occurring strictly before the current event's timestamp.
3. COLD-START ROBUSTNESS: Safe numerical fallbacks for new users or 0-variance baselines.
4. TARGET & METADATA ISOLATION: Prohibited columns (`is_anomaly`, `event_id`,
   `user_id`, `file_id`, `timestamp`) are segregated to `targets.csv` and never
   appear in `features.csv`.
"""

from __future__ import annotations

import argparse
from collections import deque
from datetime import timedelta
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.features.build_graph import compute_past_only_graph_features


# Standard categorical domains to ensure deterministic column schemas
STANDARD_ACTIONS = [
    "read",
    "write",
    "download",
    "upload",
    "delete",
    "modify",
    "preview",
]

STANDARD_DESTINATIONS = [
    "internal_server",
    "cloud_drive",
    "local_storage",
    "usb_drive",
    "personal_cloud",
    "external_email",
]

STANDARD_DEPARTMENTS = [
    "Engineering",
    "Finance",
    "HR",
    "Sales",
    "Legal",
    "IT_Admin",
    "Marketing",
]


def extract_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract circadian and calendar features from parsed timestamps.

    Features:
    - hour_of_day: Integer 0-23.
    - day_of_week: Integer 0-6 (0=Monday, 6=Sunday).
    - is_weekend: Binary flag (1 if Saturday/Sunday).
    - is_after_hours: Binary flag (1 if before 08:00 or at/after 18:00).
    """
    ts = pd.to_datetime(df["timestamp"])
    hour = ts.dt.hour
    day = ts.dt.dayofweek

    return pd.DataFrame(
        {
            "hour_of_day": hour,
            "day_of_week": day,
            "is_weekend": (day >= 5).astype(int),
            "is_after_hours": ((hour < 8) | (hour >= 18)).astype(int),
        },
        index=df.index,
    )


def extract_event_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract static event attributes and one-hot encoded categories.

    Features:
    - log_bytes_transferred: log1p of bytes transferred.
    - is_sensitive_file: Binary indicator of file sensitivity.
    - Action one-hot categories.
    - Destination one-hot categories.
    - Department one-hot categories.
    """
    feat_df = pd.DataFrame(index=df.index)

    # Volume & sensitivity
    feat_df["log_bytes_transferred"] = np.log1p(df["bytes_transferred"].astype(float))
    feat_df["is_sensitive_file"] = df["is_sensitive"].astype(int)

    # One-hot encoding with fixed columns
    action_clean = df["action"].astype(str).str.lower().str.strip()
    for act in sorted(STANDARD_ACTIONS):
        feat_df[f"action_{act}"] = (action_clean == act).astype(int)

    dest_clean = df["destination_type"].astype(str).str.lower().str.strip()
    for dst in sorted(STANDARD_DESTINATIONS):
        feat_df[f"dest_{dst}"] = (dest_clean == dst).astype(int)

    dept_clean = df["department"].astype(str).str.strip()
    for dpt in sorted(STANDARD_DEPARTMENTS):
        clean_dpt_name = dpt.lower().replace(" ", "_")
        feat_df[f"dept_{clean_dpt_name}"] = (dept_clean == dpt).astype(int)

    return feat_df


def extract_behavioral_features(
    df: pd.DataFrame,
    window_days: int = 7,
) -> pd.DataFrame:
    """Compute rolling window and expanding user historical behavior strictly from past events.

    Ensures zero future leakage by evaluating historical stats BEFORE incorporating
    the current event into the user's state.

    Features:
    - user_prior_event_count: Total cumulative events logged by user prior to this event.
    - user_prior_distinct_files: Distinct files accessed by user in past `window_days`.
    - user_prior_sensitive_count: Sensitive file accesses by user in past `window_days`.
    - user_prior_total_bytes: Cumulative bytes transferred in past `window_days`.
    - user_prior_mean_bytes: Average bytes per event in past `window_days`.
    - user_bytes_deviation: Z-score deviation from user's historical transfer baseline.
    - user_unusual_hour_ratio: Historical ratio of user's events occurring off-hours.

    Args:
        df: Cleaned logs sorted chronologically.
        window_days: Rolling window in days for velocity and cardinality calculation.

    Returns:
        DataFrame of behavioral features aligned with df.
    """
    n_rows = len(df)
    if n_rows == 0:
        return pd.DataFrame(
            columns=[
                "user_prior_event_count",
                "user_prior_distinct_files",
                "user_prior_sensitive_count",
                "user_prior_total_bytes",
                "user_prior_mean_bytes",
                "user_bytes_deviation",
                "user_unusual_hour_ratio",
            ]
        )

    ts_series = pd.to_datetime(df["timestamp"])
    window_delta = timedelta(days=window_days)

    # State containers per user
    user_windows: dict[str, deque] = {}  # deque of (timestamp, file_id, is_sensitive, bytes)
    user_cum_count: dict[str, int] = {}
    user_cum_after_hours: dict[str, int] = {}
    user_cum_bytes: dict[str, float] = {}
    user_cum_bytes_sq: dict[str, float] = {}

    out_prior_event_count = np.zeros(n_rows, dtype=int)
    out_prior_distinct_files = np.zeros(n_rows, dtype=int)
    out_prior_sensitive_count = np.zeros(n_rows, dtype=int)
    out_prior_total_bytes = np.zeros(n_rows, dtype=float)
    out_prior_mean_bytes = np.zeros(n_rows, dtype=float)
    out_bytes_deviation = np.zeros(n_rows, dtype=float)
    out_unusual_hour_ratio = np.zeros(n_rows, dtype=float)

    for idx, (_, row) in enumerate(df.iterrows()):
        u = str(row["user_id"])
        ts = ts_series.iloc[idx]
        f = str(row["file_id"])
        b = float(row["bytes_transferred"])
        is_sens = int(row.get("is_sensitive", 0))
        h = ts.hour
        is_off_hour = 1 if (h < 8 or h >= 18) else 0

        # --- 1. EVALUATE HISTORICAL WINDOW BEFORE CURRENT EVENT ---
        if u in user_windows:
            w_deque = user_windows[u]
            # Prune events outside rolling window
            cutoff = ts - window_delta
            while w_deque and w_deque[0][0] < cutoff:
                w_deque.popleft()

            w_len = len(w_deque)
            if w_len > 0:
                out_prior_distinct_files[idx] = len({item[1] for item in w_deque})
                out_prior_sensitive_count[idx] = sum(item[2] for item in w_deque)
                w_tot_bytes = sum(item[3] for item in w_deque)
                out_prior_total_bytes[idx] = w_tot_bytes
                out_prior_mean_bytes[idx] = w_tot_bytes / w_len

        # Cumulative expanding history
        past_events = user_cum_count.get(u, 0)
        out_prior_event_count[idx] = past_events

        if past_events > 0:
            past_off = user_cum_after_hours.get(u, 0)
            out_unusual_hour_ratio[idx] = past_off / past_events

        # Deviation (Z-score) against user's prior history
        if past_events >= 2:
            c_bytes = user_cum_bytes[u]
            c_bytes_sq = user_cum_bytes_sq[u]
            mean_b = c_bytes / past_events
            var_b = max(0.0, (c_bytes_sq / past_events) - (mean_b ** 2))
            std_b = math.sqrt(var_b)
            if std_b > 1.0:
                # Bound z-score between -5.0 and +10.0 for numerical safety
                z = (b - mean_b) / std_b
                out_bytes_deviation[idx] = max(-5.0, min(z, 10.0))
            else:
                out_bytes_deviation[idx] = 0.0
        else:
            out_bytes_deviation[idx] = 0.0  # Cold-start default

        # --- 2. UPDATE USER STATE FOR FUTURE EVENTS ---
        if u not in user_windows:
            user_windows[u] = deque()
            user_cum_count[u] = 0
            user_cum_after_hours[u] = 0
            user_cum_bytes[u] = 0.0
            user_cum_bytes_sq[u] = 0.0

        user_windows[u].append((ts, f, is_sens, b))
        user_cum_count[u] += 1
        user_cum_after_hours[u] += is_off_hour
        user_cum_bytes[u] += b
        user_cum_bytes_sq[u] += (b ** 2)

    return pd.DataFrame(
        {
            "user_prior_event_count": out_prior_event_count,
            "user_prior_distinct_files": out_prior_distinct_files,
            "user_prior_sensitive_count": out_prior_sensitive_count,
            "user_prior_total_bytes": out_prior_total_bytes,
            "user_prior_mean_bytes": out_prior_mean_bytes,
            "user_bytes_deviation": out_bytes_deviation,
            "user_unusual_hour_ratio": out_unusual_hour_ratio,
        },
        index=df.index,
    )


def build_full_feature_pipeline(
    df: pd.DataFrame,
    window_days: int = 7,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Execute end-to-end feature extraction, returning features and targets separately.

    Guarantees:
    - EXACT row alignment between features and targets.
    - Zero target or prohibited metadata leakage into features.

    Args:
        df: Cleaned access logs DataFrame, sorted chronologically.
        window_days: Rolling window in days for behavioral aggregations.

    Returns:
        Tuple of (features_df, targets_df).
    """
    if len(df) == 0:
        return pd.DataFrame(), pd.DataFrame()

    # Ensure chronological order
    df_sorted = df.sort_values(by=["timestamp", "event_id"]).reset_index(drop=True)

    # 1. Temporal Features
    temporal_df = extract_temporal_features(df_sorted)

    # 2. Event and Categorical Features
    event_df = extract_event_features(df_sorted)

    # 3. Behavioral Past-Only Features
    behavioral_df = extract_behavioral_features(df_sorted, window_days=window_days)

    # 4. NetworkX Graph Features (Strictly past-only)
    graph_df = compute_past_only_graph_features(df_sorted)

    # Combine all numerical & categorical features
    features_df = pd.concat(
        [temporal_df, event_df, behavioral_df, graph_df],
        axis=1,
    )

    # Targets & Metadata Dataframe
    target_columns = ["event_id", "user_id", "timestamp", "is_anomaly"]
    if "anomaly_scenario" in df_sorted.columns:
        target_columns.append("anomaly_scenario")

    targets_df = df_sorted[target_columns].copy()

    # Defensive Sanity Check: Verify no prohibited columns exist in features_df
    prohibited_cols = {"is_anomaly", "anomaly_scenario", "event_id", "user_id", "file_id", "timestamp"}
    leaked_cols = prohibited_cols.intersection(features_df.columns)
    if leaked_cols:
        raise ValueError(f"CRITICAL TARGET LEAKAGE: Prohibited columns found in features: {leaked_cols}")

    # Verify no NaN values
    if features_df.isna().any().any():
        nan_cols = features_df.columns[features_df.isna().any()].tolist()
        raise ValueError(f"Engineered features contain NaN values in columns: {nan_cols}")

    return features_df, targets_df


def plot_feature_distributions(
    features_df: pd.DataFrame,
    output_path: str | Path,
) -> None:
    """Generate diagnostic distribution plots of core engineered features."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    key_features = [
        "log_bytes_transferred",
        "hour_of_day",
        "user_prior_event_count",
        "user_bytes_deviation",
        "graph_user_degree",
        "graph_user_sensitive_ratio",
    ]

    # Select available features from key list
    plot_cols = [c for c in key_features if c in features_df.columns]
    if not plot_cols:
        return

    sns.set_theme(style="whitegrid", palette="muted")
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 9), dpi=120)
    axes = axes.flatten()

    for idx, col in enumerate(plot_cols):
        ax = axes[idx]
        data = features_df[col].dropna()
        sns.histplot(data, kde=True, ax=ax, color="#1f77b4", bins=30)
        ax.set_title(f"Distribution: {col}", fontsize=11, fontweight="bold")
        ax.set_xlabel(col)
        ax.set_ylabel("Frequency")

    plt.suptitle(
        "Key Engineered Behavioral, Temporal & Graph Feature Distributions",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )
    plt.tight_layout()
    plt.savefig(out, bbox_inches="tight")
    plt.close()


def main() -> None:
    """CLI runner for feature extraction."""
    parser = argparse.ArgumentParser(description="Extract features from cleaned access logs.")
    parser.add_argument(
        "--input",
        type=str,
        default="data/processed/clean_access_logs.csv",
        help="Path to cleaned access logs CSV",
    )
    parser.add_argument(
        "--features-out",
        type=str,
        default="data/processed/features.csv",
        help="Path to save features CSV",
    )
    parser.add_argument(
        "--targets-out",
        type=str,
        default="data/processed/targets.csv",
        help="Path to save targets CSV",
    )
    parser.add_argument(
        "--plot-out",
        type=str,
        default="reports/figures/feature_distributions.png",
        help="Path to save feature distributions plot",
    )
    parser.add_argument(
        "--window-days",
        type=int,
        default=7,
        help="Rolling historical window in days (default: 7)",
    )

    args = parser.parse_args()

    print(f"[*] Loading cleaned telemetry from: {args.input}")
    df = pd.read_csv(args.input)

    print(f"[*] Extracting temporal, event, behavioral, and graph features...")
    features_df, targets_df = build_full_feature_pipeline(df, window_days=args.window_days)

    print(f"[+] Feature matrix shape: {features_df.shape[0]:,} rows x {features_df.shape[1]} columns")
    print(f"[+] Target matrix shape:  {targets_df.shape[0]:,} rows x {targets_df.shape[1]} columns")

    # Save CSV outputs
    Path(args.features_out).parent.mkdir(parents=True, exist_ok=True)
    features_df.to_csv(args.features_out, index=False)
    print(f"[+] Saved features to: {args.features_out}")

    Path(args.targets_out).parent.mkdir(parents=True, exist_ok=True)
    targets_df.to_csv(args.targets_out, index=False)
    print(f"[+] Saved targets to:  {args.targets_out}")

    # Plot feature distributions
    print(f"[*] Generating feature distributions plot to: {args.plot_out}")
    plot_feature_distributions(features_df, args.plot_out)
    print(f"[+] Plot saved successfully.")


if __name__ == "__main__":
    main()
