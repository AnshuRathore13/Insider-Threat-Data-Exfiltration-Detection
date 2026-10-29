"""Unit tests for NetworkX graph feature extraction and topology analysis."""

from __future__ import annotations

import pathlib
import tempfile
import networkx as nx
import pandas as pd
import pytest

from src.features.build_graph import (
    build_aggregate_graph,
    compute_past_only_graph_features,
    save_graph_visualization,
)


@pytest.fixture
def tiny_graph_events() -> pd.DataFrame:
    """Deterministic sequence of 4 access events across 2 users, 2 files, 2 destinations."""
    return pd.DataFrame([
        {
            "event_id": "EVT-01",
            "user_id": "USR-A",
            "timestamp": "2026-01-05 09:00:00",
            "department": "Engineering",
            "file_id": "FILE-1",
            "destination_type": "internal_server",
            "bytes_transferred": 1000,
            "is_sensitive": 0,
        },
        {
            "event_id": "EVT-02",
            "user_id": "USR-A",
            "timestamp": "2026-01-05 10:00:00",
            "department": "Engineering",
            "file_id": "FILE-2",  # Second unique file for USR-A
            "destination_type": "internal_server",
            "bytes_transferred": 2000,
            "is_sensitive": 1,  # Sensitive file
        },
        {
            "event_id": "EVT-03",
            "user_id": "USR-B",  # New user
            "timestamp": "2026-01-05 11:00:00",
            "department": "Finance",
            "file_id": "FILE-1",  # Connecting to FILE-1 previously accessed by USR-A
            "destination_type": "cloud_drive",
            "bytes_transferred": 500,
            "is_sensitive": 0,
        },
        {
            "event_id": "EVT-04",
            "user_id": "USR-A",
            "timestamp": "2026-01-05 12:00:00",
            "department": "Engineering",
            "file_id": "FILE-1",  # Re-accessing FILE-1
            "destination_type": "usb_drive",
            "bytes_transferred": 8000,
            "is_sensitive": 0,
        },
    ])


def test_past_only_graph_degree_correctness(tiny_graph_events: pd.DataFrame) -> None:
    """Verify incremental graph degree evolution adheres strictly to past-only events."""
    gf = compute_past_only_graph_features(
        tiny_graph_events,
        total_expected_files=10,
        total_departments=5,
    )

    # Event 1: USR-A -> FILE-1
    # Before Event 1, graph was empty:
    assert gf.loc[0, "graph_user_degree"] == 0
    assert gf.loc[0, "graph_file_degree"] == 0
    assert gf.loc[0, "graph_destination_degree"] == 0
    assert gf.loc[0, "graph_user_sensitive_files"] == 0

    # Event 2: USR-A -> FILE-2 (sensitive)
    # Before Event 2, USR-A had accessed FILE-1 (user_degree=1). FILE-2 is seen for first time (file_degree=0):
    assert gf.loc[1, "graph_user_degree"] == 1
    assert gf.loc[1, "graph_file_degree"] == 0
    assert gf.loc[1, "graph_destination_degree"] == 1  # internal_server had 1 user (USR-A)
    assert gf.loc[1, "graph_user_sensitive_files"] == 0  # FILE-1 was not sensitive

    # Event 3: USR-B -> FILE-1
    # Before Event 3: USR-B is brand new (user_degree=0). FILE-1 was accessed by USR-A (file_degree=1):
    assert gf.loc[2, "graph_user_degree"] == 0
    assert gf.loc[2, "graph_file_degree"] == 1  # Accessed by USR-A in Event 1!
    assert gf.loc[2, "graph_destination_degree"] == 0  # cloud_drive seen for first time

    # Event 4: USR-A -> FILE-1
    # Before Event 4: USR-A has connected to FILE-1 and FILE-2 (user_degree=2).
    # FILE-1 has been accessed by USR-A and USR-B (file_degree=2).
    # USR-A accessed FILE-2 which was sensitive (sensitive_files=1):
    assert gf.loc[3, "graph_user_degree"] == 2
    assert gf.loc[3, "graph_file_degree"] == 2
    assert gf.loc[3, "graph_user_sensitive_files"] == 1
    assert gf.loc[3, "graph_user_sensitive_ratio"] == 0.5  # 1 / 2


def test_isolated_and_cold_start_nodes() -> None:
    """Verify brand new / isolated nodes receive 0s and no NaNs or KeyErrors."""
    cold_df = pd.DataFrame([
        {
            "user_id": "USR-NEW",
            "file_id": "FILE-NEW",
            "destination_type": "NEW_DEST",
            "department": "HR",
            "is_sensitive": 0,
        }
    ])
    gf = compute_past_only_graph_features(cold_df)
    assert gf.loc[0, "graph_user_degree"] == 0
    assert gf.loc[0, "graph_file_degree"] == 0
    assert gf.loc[0, "graph_destination_degree"] == 0
    assert gf.loc[0, "graph_user_degree_centrality"] == 0.0
    assert not gf.isna().any().any()


def test_empty_graph_input() -> None:
    """Verify empty dataframe returns valid empty graph features dataframe."""
    empty_df = pd.DataFrame(columns=["user_id", "file_id", "destination_type", "department", "is_sensitive"])
    gf = compute_past_only_graph_features(empty_df)
    assert len(gf) == 0
    assert "graph_user_degree" in gf.columns


def test_aggregate_graph_and_visualization(tiny_graph_events: pd.DataFrame) -> None:
    """Verify aggregate graph construction and image export."""
    G = build_aggregate_graph(tiny_graph_events)
    assert G.number_of_nodes() > 0
    assert G.number_of_edges() > 0

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_img = pathlib.Path(tmp_dir) / "test_graph.png"
        save_graph_visualization(G, tmp_img)
        assert tmp_img.exists()
        assert tmp_img.stat().st_size > 0
