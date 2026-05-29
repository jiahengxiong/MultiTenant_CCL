#!/usr/bin/env python3
"""Generate publication-style ILP oracle validation results."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.table import Table


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_DIR = ROOT / "experiment"
OUT_DIR = ROOT / "figures"


def load_rows() -> list[dict[str, float | int | str]]:
    specs = [
        ("Single", EXPERIMENT_DIR / "mapping_vs_ilp_single.json"),
        ("Multi", EXPERIMENT_DIR / "mapping_vs_ilp_multi.json"),
    ]
    rows: list[dict[str, float | int | str]] = []
    for program, path in specs:
        data = json.loads(path.read_text())
        for result in sorted(data["results"], key=lambda item: int(item["tenant_count"])):
            heuristic = result["heuristic"]
            ilp = result["ilp"]
            avg_gap = 100.0 * (heuristic["avg_jct"] - ilp["avg_jct"]) / ilp["avg_jct"]
            makespan_gap = 100.0 * (heuristic["makespan"] - ilp["makespan"]) / ilp["makespan"]
            rows.append(
                {
                    "program": program,
                    "tenants": int(result["tenant_count"]),
                    "avg_gap": avg_gap,
                    "makespan_gap": makespan_gap,
                    "heuristic_runtime": float(heuristic["runtime_seconds"]),
                    "ilp_runtime": float(ilp["runtime_seconds"]),
                    "speedup": float(ilp["runtime_seconds"]) / max(float(heuristic["runtime_seconds"]), 1e-12),
                    "heuristic_avg": float(heuristic["avg_jct"]),
                    "ilp_avg": float(ilp["avg_jct"]),
                    "heuristic_makespan": float(heuristic["makespan"]),
                    "ilp_makespan": float(ilp["makespan"]),
                }
            )
    return rows


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


def plot_gap_bars(rows: list[dict[str, float | int | str]]) -> None:
    labels = [f"{row['program']}-{row['tenants']}T" for row in rows]
    avg_gaps = [float(row["avg_gap"]) for row in rows]
    makespan_gaps = [float(row["makespan_gap"]) for row in rows]
    x = list(range(len(rows)))
    width = 0.36

    fig, ax = plt.subplots(figsize=(3.45, 1.75), dpi=600)
    ax.bar(
        [idx - width / 2 for idx in x],
        avg_gaps,
        width,
        label="Avg. JCT",
        color="#0072B2",
        edgecolor="black",
        linewidth=0.35,
    )
    ax.bar(
        [idx + width / 2 for idx in x],
        makespan_gaps,
        width,
        label="Makespan",
        color="#D55E00",
        edgecolor="black",
        linewidth=0.35,
    )

    max_gap = max(max(avg_gaps), max(makespan_gaps))
    ax.set_ylim(0, max(0.016, max_gap * 1.25))
    ax.set_ylabel("Gap to MILP optimum (%)")
    ax.set_xticks(x, labels, rotation=22, ha="right")
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.45)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_linewidth(0.6)
    ax.spines["bottom"].set_linewidth(0.6)
    ax.legend(loc="upper left", frameon=False, ncol=2, bbox_to_anchor=(0.0, 1.18))

    for idx, value in enumerate(avg_gaps):
        if value > 0:
            ax.text(
                idx - width / 2,
                value + 0.0006,
                f"{value:.3f}%",
                ha="center",
                va="bottom",
                fontsize=6.6,
            )

    fig.tight_layout(pad=0.15)
    save_all(fig, "ilp_oracle_optimality_gap")
    plt.close(fig)


def plot_validation_table(rows: list[dict[str, float | int | str]]) -> None:
    fig, ax = plt.subplots(figsize=(7.1, 1.75), dpi=600)
    ax.axis("off")

    columns = [
        "Program",
        "Tenants",
        "Heur. Avg JCT",
        "ILP Avg JCT",
        "Heur. makespan",
        "ILP makespan",
        "Heur. time (s)",
        "ILP time (s)",
    ]
    table_rows = [
        [
            str(row["program"]),
            f"{row['tenants']}",
            f"{float(row['heuristic_avg']):.6f}",
            f"{float(row['ilp_avg']):.6f}",
            f"{float(row['heuristic_makespan']):.6f}",
            f"{float(row['ilp_makespan']):.6f}",
            f"{float(row['heuristic_runtime']):.3f}",
            f"{float(row['ilp_runtime']):.1f}",
        ]
        for row in rows
    ]

    table = Table(ax, bbox=[0, 0, 1, 1])
    all_rows = [columns] + table_rows
    col_widths = [0.15, 0.09, 0.14, 0.14, 0.15, 0.15, 0.09, 0.09]
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
            text.set_fontsize(7.2)
            if row_idx == 0:
                text.set_fontweight("bold")
    ax.add_table(table)
    save_all(fig, "ilp_oracle_validation_table")
    plt.close(fig)


def write_latex_table(rows: list[dict[str, float | int | str]]) -> None:
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Comparison between the heuristic and the MILP oracle on oracle-solvable instances. Runtime is solver-only wall-clock time.}",
        r"\label{tab:ilp-oracle-validation}",
        r"\begin{tabular}{lccccccc}",
        r"\toprule",
        r"Program & \# Tenants & Heur. Avg. JCT & ILP Avg. JCT & Heur. makespan & ILP makespan & Heur. time (s) & ILP time (s) \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            f"{row['program']} & {row['tenants']} & "
            f"{float(row['heuristic_avg']):.6f} & {float(row['ilp_avg']):.6f} & "
            f"{float(row['heuristic_makespan']):.6f} & {float(row['ilp_makespan']):.6f} & "
            f"{float(row['heuristic_runtime']):.3f} & {float(row['ilp_runtime']):.1f} \\\\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
            "",
        ]
    )
    (OUT_DIR / "ilp_oracle_validation_table.tex").write_text("\n".join(lines))


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    set_publication_style()
    rows = load_rows()
    plot_gap_bars(rows)
    plot_validation_table(rows)
    write_latex_table(rows)


if __name__ == "__main__":
    main()
