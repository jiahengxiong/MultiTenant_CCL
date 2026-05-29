#!/usr/bin/env python3
"""Generate publication-style main-result figures from experiment JSON files."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.table import Table


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_DIR = ROOT / "experiment"
OUT_DIR = ROOT / "figures"


SCENARIOS = [
    ("Low contention, heterogeneous", "Low_contension.json"),
    ("High contention, heterogeneous", "High_contension.json"),
    ("Low contention, homogeneous", "Low_contension_homo.json"),
    ("High contention, homogeneous", "High_contension_homo.json"),
]

METHODS = [
    ("default", "Default", "#4D4D4D", "o", "-"),
    ("locality", "Locality", "#009E73", "s", "-"),
    ("mapping", "Rank mapping", "#0072B2", "^", "-"),
    ("default_plus_harmonics", "Default + harmonics", "#4D4D4D", "o", "--"),
    ("locality_plus_harmonics", "Locality + harmonics", "#009E73", "s", "--"),
    ("mapping_plus_harmonics", "Rank mapping + harmonics", "#0072B2", "^", "--"),
]


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


def load_scenario(filename: str) -> list[dict]:
    data = json.loads((EXPERIMENT_DIR / filename).read_text())
    return sorted(data["results"], key=lambda item: int(item["tenant_count"]))


def normalized_series(results: list[dict], method: str, metric: str) -> tuple[list[int], list[float]]:
    tenants: list[int] = []
    values: list[float] = []
    for result in results:
        tenants.append(int(result["tenant_count"]))
        default = float(result["averages"]["default"][metric])
        current = float(result["averages"][method][metric])
        values.append(current / default)
    return tenants, values


def save_all(fig: plt.Figure, stem: str) -> None:
    for ext in ("pdf", "svg", "png"):
        fig.savefig(OUT_DIR / f"{stem}.{ext}", bbox_inches="tight", pad_inches=0.02)


def plot_metric(metric: str, ylabel: str, stem: str, ylim: tuple[float, float]) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(7.1, 3.75), dpi=600, sharex=True)
    axes_flat = axes.ravel()

    legend_handles: list = []
    legend_labels: list[str] = []
    for ax, (title, filename) in zip(axes_flat, SCENARIOS):
        results = load_scenario(filename)
        tenants = [int(result["tenant_count"]) for result in results]
        for method_key, method_label, color, marker, linestyle in METHODS:
            x_values, y_values = normalized_series(results, method_key, metric)
            line = ax.plot(
                x_values,
                y_values,
                color=color,
                marker=marker,
                markersize=3.5,
                linewidth=1.05,
                linestyle=linestyle,
                markerfacecolor="white" if linestyle == "--" else color,
                markeredgewidth=0.75,
                label=method_label,
            )[0]
            if len(legend_handles) < len(METHODS):
                legend_handles.append(line)
                legend_labels.append(method_label)

        ax.set_title(title, pad=4)
        ax.set_xticks(tenants)
        ax.set_ylim(*ylim)
        ax.grid(axis="y", color="#D9D9D9", linewidth=0.45)
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        ax.spines["left"].set_linewidth(0.6)
        ax.spines["bottom"].set_linewidth(0.6)

    fig.legend(
        legend_handles,
        legend_labels,
        loc="upper center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 1.025),
        columnspacing=1.15,
        handlelength=2.0,
    )

    fig.supxlabel("Number of tenants", y=0.02, fontsize=8)
    fig.supylabel(ylabel, x=0.015, fontsize=8)
    fig.tight_layout(rect=[0.035, 0.055, 1.0, 0.90], h_pad=0.75, w_pad=0.75)
    save_all(fig, stem)
    plt.close(fig)


def write_summary_table() -> None:
    method_keys = [
        "default",
        "locality",
        "mapping",
        "default_plus_harmonics",
        "locality_plus_harmonics",
        "mapping_plus_harmonics",
    ]
    method_labels = ["Default", "Locality", "Mapping", "Default+H", "Locality+H", "Mapping+H"]
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Average reduction relative to default placement across tenant counts.}",
        r"\label{tab:main-result-summary}",
        r"\begin{tabular}{llrrrrrr}",
        r"\toprule",
        r"Scenario & Metric & Default & Locality & Mapping & Default+H & Locality+H & Mapping+H \\",
        r"\midrule",
    ]
    for title, filename in SCENARIOS:
        results = load_scenario(filename)
        for metric, label in [("avg_jct", "Avg. JCT"), ("makespan", "Makespan")]:
            reductions = []
            for method in method_keys:
                ratios = []
                for result in results:
                    default = float(result["averages"]["default"][metric])
                    current = float(result["averages"][method][metric])
                    ratios.append(current / default)
                reductions.append(100.0 * (1.0 - sum(ratios) / len(ratios)))
            lines.append(
                f"{title} & {label} & "
                + " & ".join(f"{value:.1f}\\%" for value in reductions)
                + r" \\"
            )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table*}",
            "",
        ]
    )
    (OUT_DIR / "main_result_summary_table.tex").write_text("\n".join(lines))

    fig, ax = plt.subplots(figsize=(7.1, 1.75), dpi=600)
    ax.axis("off")

    table_rows = []
    for title, filename in SCENARIOS:
        results = load_scenario(filename)
        for metric, label in [("avg_jct", "Avg. JCT"), ("makespan", "Makespan")]:
            row = [title, label]
            for method in method_keys:
                ratios = []
                for result in results:
                    default = float(result["averages"]["default"][metric])
                    current = float(result["averages"][method][metric])
                    ratios.append(current / default)
                reduction = 100.0 * (1.0 - sum(ratios) / len(ratios))
                row.append(f"{reduction:.1f}%")
            table_rows.append(row)

    columns = ["Scenario", "Metric", *method_labels]
    all_rows = [columns, *table_rows]
    col_widths = [0.245, 0.105, 0.09, 0.09, 0.09, 0.12, 0.125, 0.135]
    row_height = 1 / len(all_rows)
    table = Table(ax, bbox=[0, 0, 1, 1])
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
            cell.PAD = 0.015
            text = cell.get_text()
            text.set_fontsize(6.6)
            if row_idx == 0:
                text.set_fontweight("bold")
            if row_idx > 0 and col_idx in (4, 7):
                text.set_fontweight("bold")
            if row_idx > 0 and col_idx >= 2 and str(value).startswith("-"):
                text.set_color("#B00020")
    ax.add_table(table)
    save_all(fig, "main_result_summary_table")
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    set_publication_style()
    plot_metric(
        metric="avg_jct",
        ylabel="Normalized Avg. JCT",
        stem="main_avg_jct_normalized",
        ylim=(0.0, 1.08),
    )
    plot_metric(
        metric="makespan",
        ylabel="Normalized makespan",
        stem="main_makespan_normalized",
        ylim=(0.0, 1.50),
    )
    write_summary_table()


if __name__ == "__main__":
    main()
