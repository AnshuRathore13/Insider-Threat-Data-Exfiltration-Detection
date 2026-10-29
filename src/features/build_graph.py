"""NetworkX Graph Analytics for Insider Threat and Exfiltration Detection.

This module constructs bipartite/tripartite access graphs linking:
    Users <---> Files <---> Destinations 

Key Architectural Principles:
1. PREDICTIVE FEATURES ARE STRICTLY PAST-ONLY:
   An event at timestamp t MUST NOT receive graph information from events occurring at
   or after t. We compute graph metrics using an expanding historical graph state
   evaluated prior to edge insertion.
2. AGGREGATE GRAPH IS PURELY DESCRIPTIVE:
   The aggregate graph is built post-hoc for SOC analyst visualization and global
   topology inspection in `reports/figures/access_graph.png`.
3. TARGET LEAKAGE IS PREVENTED:
   No graph edges or node attributes incorporate the `is_anomaly` synthetic label.
4. COLD-START RESILIENCE:
   New nodes with zero prior connections receive safe defaults (0 degree, 0 centrality).

Cybersecurity Meanings of Graph Metrics:
- user_degree: Breadth of files touched. High degree suggests reconnaissance,
  directory harvesting, or privilege overreach.
- file_degree: Number of users accessing an asset. High degree indicates public/shared
  resources, while low degree indicates private/niche assets.
- destination_degree: Popularity of exfiltration channel. Rare channels (e.g. USB)
  have low overall degree across the enterprise.
- user_sensitive_ratio: Proportion of targeted assets that are classified confidential.
- user_access_diversity: Cross-department reach across departmental file silos.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd


def compute_past_only_graph_features(
    df: pd.DataFrame,
    total_expected_files: int = 300,
    total_departments: int = 7,
) -> pd.DataFrame:
    """Compute strictly past-only NetworkX graph features for every event.

    Events must be pre-sorted chronologically. For each event at index i:
    1. Look up node metrics in the historical graph state accumulated strictly BEFORE event i.
    2. Add/update the event's connections (user, file, destination) into the graph state.

    Args:
        df: Cleaned access logs DataFrame, sorted chronologically.
        total_expected_files: Denominator for degree centrality normalization.
        total_departments: Denominator for department diversity normalization.

    Returns:
        DataFrame containing graph feature columns aligned row-by-row with input df.
    """
    if len(df) == 0:
        return pd.DataFrame(
            columns=[
                "graph_user_degree",
                "graph_file_degree",
                "graph_destination_degree",
                "graph_user_sensitive_files",
                "graph_user_sensitive_ratio",
                "graph_user_access_diversity",
                "graph_user_degree_centrality",
            ]
        )

    # State tracking: strictly accumulated from past events
    user_files: dict[str, set[str]] = {}
    file_users: dict[str, set[str]] = {}
    dest_users: dict[str, set[str]] = {}
    user_sensitive_files: dict[str, set[str]] = {}
    user_departments: dict[str, set[str]] = {}

    rows_user_degree: list[int] = []
    rows_file_degree: list[int] = []
    rows_dest_degree: list[int] = []
    rows_sens_files: list[int] = []
    rows_sens_ratio: list[float] = []
    rows_diversity: list[float] = []
    rows_centrality: list[float] = []

    for _, row in df.iterrows():
        u = str(row["user_id"])
        f = str(row["file_id"])
        d = str(row["destination_type"])
        dept = str(row["department"])
        is_sens = int(row.get("is_sensitive", 0))

        # 1. EVALUATE HISTORICAL STATE (Before current event is added)
        u_files = user_files.get(u, set())
        f_users = file_users.get(f, set())
        d_users = dest_users.get(d, set())
        u_sens = user_sensitive_files.get(u, set())
        u_depts = user_departments.get(u, set())

        u_deg = len(u_files)
        f_deg = len(f_users)
        d_deg = len(d_users)
        s_count = len(u_sens)
        s_ratio = (s_count / u_deg) if u_deg > 0 else 0.0
        diversity = (len(u_depts) / total_departments) if total_departments > 0 else 0.0
        centrality = (u_deg / total_expected_files) if total_expected_files > 0 else 0.0

        rows_user_degree.append(u_deg)
        rows_file_degree.append(f_deg)
        rows_dest_degree.append(d_deg)
        rows_sens_files.append(s_count)
        rows_sens_ratio.append(s_ratio)
        rows_diversity.append(diversity)
        rows_centrality.append(centrality)

        # 2. UPDATE HISTORICAL STATE (Incorporating current event for future lookups)
        if u not in user_files:
            user_files[u] = set()
            user_sensitive_files[u] = set()
            user_departments[u] = set()
        user_files[u].add(f)
        user_departments[u].add(dept)
        if is_sens == 1:
            user_sensitive_files[u].add(f)

        if f not in file_users:
            file_users[f] = set()
        file_users[f].add(u)

        if d not in dest_users:
            dest_users[d] = set()
        dest_users[d].add(u)

    graph_features_df = pd.DataFrame(
        {
            "graph_user_degree": rows_user_degree,
            "graph_file_degree": rows_file_degree,
            "graph_destination_degree": rows_dest_degree,
            "graph_user_sensitive_files": rows_sens_files,
            "graph_user_sensitive_ratio": rows_sens_ratio,
            "graph_user_access_diversity": rows_diversity,
            "graph_user_degree_centrality": rows_centrality,
        },
        index=df.index,
    )

    return graph_features_df


def build_aggregate_graph(
    df: pd.DataFrame,
    sample_users: Optional[int] = 20,
    sample_files: Optional[int] = 50,
) -> nx.Graph:
    """Build a comprehensive NetworkX aggregate graph for global visualization.

    Note: This aggregate graph incorporates the full dataset and is designed
    for exploratory visualization and SOC analyst dashboards. It is NEVER used
    directly to derive predictive event-level features (which require past-only snapshots).

    Args:
        df: Cleaned access logs DataFrame.
        sample_users: Optional cap on user count to keep visual readable.
        sample_files: Optional cap on file count to keep visual readable.

    Returns:
        NetworkX Graph with annotated node and edge attributes.
    """
    G = nx.Graph()

    if len(df) == 0:
        return G

    # Filter to top active entities if dataset is large, ensuring clean rendering
    if sample_users is not None and df["user_id"].nunique() > sample_users:
        top_users = df["user_id"].value_counts().head(sample_users).index
        plot_df = df[df["user_id"].isin(top_users)].copy()
    else:
        plot_df = df.copy()

    if sample_files is not None and plot_df["file_id"].nunique() > sample_files:
        top_files = plot_df["file_id"].value_counts().head(sample_files).index
        plot_df = plot_df[plot_df["file_id"].isin(top_files)].copy()

    # Add Nodes
    for _, row in plot_df.iterrows():
        u = f"User:{row['user_id']}"
        f = f"File:{row['file_id']}"
        d = f"Dest:{row['destination_type']}"

        if not G.has_node(u):
            G.add_node(u, node_type="user", label=row["user_id"], dept=row["department"])
        if not G.has_node(f):
            G.add_node(
                f,
                node_type="file",
                label=row["file_id"],
                is_sensitive=row.get("is_sensitive", 0),
            )
        if not G.has_node(d):
            G.add_node(d, node_type="destination", label=row["destination_type"])

        # Edge User <--> File
        if G.has_edge(u, f):
            G[u][f]["weight"] += 1
            G[u][f]["bytes"] += int(row["bytes_transferred"])
        else:
            G.add_edge(u, f, weight=1, bytes=int(row["bytes_transferred"]))

        # Edge User <--> Destination
        if G.has_edge(u, d):
            G[u][d]["weight"] += 1
        else:
            G.add_edge(u, d, weight=1)

    return G


def save_graph_visualization(
    G: nx.Graph,
    output_path: str | Path,
    seed: int = 42,
) -> None:
    """Render and save an enterprise access graph visualization to disk."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    if G.number_of_nodes() == 0:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "Empty Graph", horizontalalignment="center")
        plt.savefig(out)
        plt.close(fig)
        return

    plt.figure(figsize=(14, 10), dpi=150)
    pos = nx.spring_layout(G, seed=seed, k=0.35, iterations=40)

    # Color mapping by node type
    node_colors: list[str] = []
    node_sizes: list[int] = []

    for node, data in G.nodes(data=True):
        ntype = data.get("node_type", "")
        if ntype == "user":
            node_colors.append("#1f77b4")  # Blue
            node_sizes.append(260)
        elif ntype == "destination":
            node_colors.append("#2ca02c")  # Green
            node_sizes.append(400)
        elif ntype == "file":
            if data.get("is_sensitive", 0) == 1:
                node_colors.append("#d62728")  # Red for sensitive assets
                node_sizes.append(180)
            else:
                node_colors.append("#ff7f0e")  # Orange for regular files
                node_sizes.append(120)
        else:
            node_colors.append("#7f7f7f")
            node_sizes.append(100)

    # Draw edges with subtle alpha
    nx.draw_networkx_edges(G, pos, alpha=0.15, edge_color="#444444", width=0.8)

    # Draw nodes
    nx.draw_networkx_nodes(
        G,
        pos,
        node_color=node_colors,
        node_size=node_sizes,
        alpha=0.9,
    )

    # Selected labels for users and destinations to avoid visual clutter
    label_dict = {
        node: data.get("label", node)
        for node, data in G.nodes(data=True)
        if data.get("node_type") in ("user", "destination")
    }
    nx.draw_networkx_labels(G, pos, labels=label_dict, font_size=7, font_family="sans-serif")

    # Legend annotations
    legend_elements = [
        plt.Line2D([0], [0], marker="o", color="w", label="User Entity", markerfacecolor="#1f77b4", markersize=10),
        plt.Line2D([0], [0], marker="o", color="w", label="Sensitive Asset", markerfacecolor="#d62728", markersize=9),
        plt.Line2D([0], [0], marker="o", color="w", label="Standard File", markerfacecolor="#ff7f0e", markersize=8),
        plt.Line2D([0], [0], marker="o", color="w", label="Exfiltration Channel", markerfacecolor="#2ca02c", markersize=11),
    ]
    plt.legend(handles=legend_elements, loc="upper right", frameon=True, fontsize=9)

    plt.title(
        "Enterprise Access Network Topology (Bipartite User-File-Destination Graph)\n"
        "Descriptive SOC Inspection Graph — Predictive Features Derived from Past-Only Temporal Snapshots",
        fontsize=11,
        pad=15,
        fontweight="bold",
    )
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out, bbox_inches="tight")
    plt.close()


def main() -> None:
    """CLI runner to compute past-only graph features and generate visual."""
    parser = argparse.ArgumentParser(
        description="Construct NetworkX access graph and compute graph features."
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/processed/clean_access_logs.csv",
        help="Path to cleaned access logs CSV",
    )
    parser.add_argument(
        "--output-figure",
        type=str,
        default="reports/figures/access_graph.png",
        help="Path to save descriptive graph figure",
    )

    args = parser.parse_args()
    print(f"[*] Reading telemetry from: {args.input}")
    df = pd.read_csv(args.input)

    print("[*] Computing strictly past-only graph features...")
    g_features = compute_past_only_graph_features(df)
    print(f"[+] Computed {g_features.shape[1]} graph features for {len(g_features):,} events.")

    print(f"[*] Building aggregate network graph and exporting visual to: {args.output_figure}")
    G = build_aggregate_graph(df)
    save_graph_visualization(G, args.output_figure)
    print(f"[+] Aggregate graph exported: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges.")


if __name__ == "__main__":
    main()
