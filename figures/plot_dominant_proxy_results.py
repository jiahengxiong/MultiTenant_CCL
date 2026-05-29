#!/usr/bin/env python3
"""Generate publication-style dominant-collective proxy result figures."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.table import Table


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "experiment" / "dominant vs full.json"
OUT_DIR = ROOT / "figures"


def load_results() -> tuple[dict[str, dict[str, float]], list[tuple[int, bool, float, float]]]:
    data = json.loads(DATA_PATH.read_text())

    profile_order = ["GPT13B", "LLaMA65B", "DeepSeek16B"]
    seen: dict[str, dict[str, float]] = {}
    for result in data["results"]:
        assignments = result.get("workload_assignment", {})
        full_programs = result.get("full_programs", {})
        for tenant_id, profile in assignments.items():
            profile_name = profile["profile_name"]
            if profile_name in seen:
                continue

            volume_by_collective: defaultdict[str, float] = defaultdict(float)
            for op in full_programs.get(tenant_id, []):
                volume_by_collective[op["collective"]] += float(op.get("single_flow_size_bits", 0.0))

            total_volume = sum(volume_by_collective.values())
            if total_volume <= 0:
                continue

            dominant = volume_by_collective["reducescatter"]
            allgather = volume_by_collective["allgather"]
            seen[profile_name] = {
                "ReduceScatter": 100.0 * dominant / total_volume,
                "AllGather": 100.0 * allgather / total_volume,
                "Other": max(0.0, 100.0 * (total_volume - dominant - allgather) / total_volume),
            }

    volume_shares = {name: seen[name] for name in profile_order if name in seen}

    validation_rows: list[tuple[int, bool, float, float]] = []
    for result in sorted(data["results"], key=lambda item: int(item["tenant_count"])):
        tenant_count = int(result["tenant_count"])
        dominant_mapping = result["results"]["dominant_mapping"]
        full_mapping = result["results"]["full_mapping"]
        same_mapping = dominant_mapping["mapping"] == full_mapping["mapping"]
        avg_jct_diff = 100.0 * (
            dominant_mapping["avg_jct"] - full_mapping["avg_jct"]
        ) / full_mapping["avg_jct"]
        makespan_diff = 100.0 * (
            dominant_mapping["makespan"] - full_mapping["makespan"]
        ) / full_mapping["makespan"]
        validation_rows.append((tenant_count, same_mapping, avg_jct_diff, makespan_diff))

    return volume_shares, validation_rows


def set_publication_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save_all(fig: plt.Figure, stem: str) -> None:
    for ext in ("pdf", "svg", "png"):
        fig.savefig(OUT_DIR / f"{stem}.{ext}", bbox_inches="tight", pad_inches=0.02)


def plot_volume_share(volume_shares: dict[str, dict[str, float]]) -> None:
    fig, ax = plt.subplots(figsize=(3.45, 1.55), dpi=600)
    colors = {
        "ReduceScatter": "#D55E00",
        "AllGather": "#0072B2",
        "Other": "#B8B8B8",
    }

    labels = list(volume_shares.keys())
    y_positions = range(len(labels))
    left = [0.0] * len(labels)

    for collective in ("ReduceScatter", "AllGather", "Other"):
        values = [volume_shares[name][collective] for name in labels]
        bars = ax.barh(
            y_positions,
            values,
            left=left,
            height=0.52,
            color=colors[collective],
            edgecolor="white",
            linewidth=0.45,
            label=collective,
        )
        for idx, (bar, value) in enumerate(zip(bars, values)):
            if value >= 12:
                ax.text(
                    left[idx] + value / 2,
                    bar.get_y() + bar.get_height() / 2,
                    f"{value:.1f}%",
                    ha="center",
                    va="center",
                    color="white",
                    fontsize=6.8,
                    fontweight="bold",
                )
        left = [base + value for base, value in zip(left, values)]

    ax.set_yticks(list(y_positions), labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("Share of communication volume in full program (%)")
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.grid(axis="x", color="#D9D9D9", linewidth=0.45)
    ax.set_axisbelow(True)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_linewidth(0.6)
    ax.tick_params(axis="y", length=0)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.24),
        ncol=3,
        frameon=False,
        handlelength=1.1,
        columnspacing=1.0,
    )
    save_all(fig, "dominant_collective_volume_share")
    plt.close(fig)


def plot_validation_table(validation_rows: list[tuple[int, bool, float, float]]) -> None:
    fig, ax = plt.subplots(figsize=(3.45, 1.20), dpi=600)
    ax.axis("off")

    columns = ["Tenants", "Same mapping", "Avg. JCT diff.", "Makespan diff."]
    rows = [
        [str(tenant_count), "Yes" if same else "No", f"{avg_diff:.1f}%", f"{ms_diff:.1f}%"]
        for tenant_count, same, avg_diff, ms_diff in validation_rows
    ]

    table = Table(ax, bbox=[0, 0, 1, 1])
    col_widths = [0.18, 0.36, 0.23, 0.23]
    all_rows = [columns] + rows
    row_height = 1 / len(all_rows)

    for row_idx, row in enumerate(all_rows):
        for col_idx, value in enumerate(row):
            cell = table.add_cell(
                row_idx,
                col_idx,
                col_widths[col_idx],
                row_height,
                text=value,
                loc="center",
                facecolor="#F0F3F6" if row_idx == 0 else "white",
                edgecolor="#333333",
            )
            cell.set_linewidth(0.55 if row_idx == 0 else 0.35)
            cell.PAD = 0.02
            text = cell.get_text()
            text.set_fontsize(7.4)
            if row_idx == 0:
                text.set_fontweight("bold")
            if row_idx > 0 and col_idx == 1:
                text.set_color("#007A3D")
                text.set_fontweight("bold")

    ax.add_table(table)
    save_all(fig, "dominant_vs_full_validation_table")
    plt.close(fig)


def write_latex_table(validation_rows: list[tuple[int, bool, float, float]]) -> None:
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Dominant-collective proxy validation. Optimizing only the dominant collective selects the same rank mapping as optimizing the full communication program on the evaluated traces.}",
        r"\label{tab:dominant-full-validation}",
        r"\begin{tabular}{cccc}",
        r"\toprule",
        r"\# Tenants & Same rank mapping & Avg. JCT difference & Makespan difference \\",
        r"\midrule",
    ]
    for tenant_count, same, avg_diff, ms_diff in validation_rows:
        same_text = "Yes" if same else "No"
        lines.append(f"{tenant_count} & {same_text} & {avg_diff:.1f}\\% & {ms_diff:.1f}\\% \\\\")
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
            "",
        ]
    )
    (OUT_DIR / "dominant_vs_full_validation_table.tex").write_text("\n".join(lines))


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    set_publication_style()
    volume_shares, validation_rows = load_results()
    plot_volume_share(volume_shares)
    plot_validation_table(validation_rows)
    write_latex_table(validation_rows)


if __name__ == "__main__":
    main()
