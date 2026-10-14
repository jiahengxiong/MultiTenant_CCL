#!/usr/bin/env python3
"""Generate publication-style main-result figures from experiment JSON files."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.table import Table


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_DIR = ROOT / "experiment"
OUT_DIR = ROOT / "figures"


SCENARIOS = [
    ("Low contention, heterogeneous", "Low_contension.json"),
    ("Low contention, homogeneous", "Low_contension_homo.json"),
    ("High contention, heterogeneous", "High_contension.json"),
    ("High contention, homogeneous", "High_contension_homo.json"),
]

MIN_TENANT_COUNT = 3

METHODS = [
    ("default", "Default", "#4D4D4D", ""),
    ("locality", "Locality", "#009E73", ""),
    ("mapping", "DARM", "#0072B2", ""),
    ("default_plus_harmonics", "Default+H", "#999999", "//"),
    ("locality_plus_harmonics", "Locality+H", "#56B4A5", "//"),
    ("mapping_plus_harmonics", "DARM+H", "#56B4E9", "//"),
    ("qp", "QP", "#E69F00", ""),
    ("qp_plus_harmonics", "QP+H", "#AA4499", "//"),
]

LINE_METHODS = [
    ("default", "Default", "#222222", "o", "-", "#222222", -0.120),
    ("locality", "Locality", "#D55E00", "s", "-", "#D55E00", -0.072),
    ("mapping", "DARM", "#0072B2", "^", "-", "#0072B2", -0.024),
    ("default_plus_harmonics", "Default+H", "#6F6F6F", "D", "--", "white", 0.024),
    ("locality_plus_harmonics", "Locality+H", "#CC79A7", "P", "-.", "white", 0.072),
    ("mapping_plus_harmonics", "DARM+H", "#009E73", "v", "--", "white", 0.120),
    ("qp", "QP", "#E69F00", "X", "-", "#E69F00", 0.0),
    ("qp_plus_harmonics", "QP+H", "#AA4499", "*", "--", "white", 0.0),
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
    return [
        item
        for item in sorted(data["results"], key=lambda item: int(item["tenant_count"]))
        if int(item["tenant_count"]) >= MIN_TENANT_COUNT
    ]


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


def scenario_short_name(title: str) -> str:
    title = title.replace(" contention", "")
    return title.replace(", ", "\n")


def scenario_panel_title(title: str) -> str:
    level = "Low contention" if title.startswith("Low") else "High contention"
    workload = "mixed workloads" if "heterogeneous" in title else "GPT-only workloads"
    return f"{level}, {workload}"


def plot_metric(metric: str, ylabel: str, stem: str, ylim: tuple[float, float]) -> None:
    fig, ax = plt.subplots(figsize=(7.1, 2.35), dpi=600)
    group_x = list(range(len(SCENARIOS)))
    width = 0.78 / len(METHODS)
    offsets = [width * (idx - (len(METHODS) - 1) / 2) for idx in range(len(METHODS))]

    for method_idx, (method_key, method_label, color, hatch) in enumerate(METHODS):
        values: list[float] = []
        for _, filename in SCENARIOS:
            results = load_scenario(filename)
            _, ratios = normalized_series(results, method_key, metric)
            values.append(sum(ratios) / len(ratios))
        bars = ax.bar(
            [x + offsets[method_idx] for x in group_x],
            values,
            width=width,
            label=method_label,
            color=color,
            hatch=hatch,
            edgecolor="black",
            linewidth=0.35,
        )
        for bar, value in zip(bars, values):
            if method_key in {"mapping", "mapping_plus_harmonics"}:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    value + 0.018,
                    f"{value:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=6.2,
                )

    ax.axhline(1.0, color="#666666", linewidth=0.6, linestyle=":")
    ax.set_xticks(group_x, [scenario_short_name(title) for title, _ in SCENARIOS])
    ax.set_ylim(ylim[0], max(ylim[1], max(bar.get_height() for bar in ax.patches) * 1.06))
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.45)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_linewidth(0.6)
    ax.spines["bottom"].set_linewidth(0.6)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.25),
        ncol=4 if len(METHODS) > 6 else 6,
        frameon=False,
        columnspacing=0.8,
        handlelength=1.4,
    )
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.88], pad=0.2)
    save_all(fig, stem)
    plt.close(fig)


def plot_avg_jct_by_tenant() -> None:
    fig, axes = plt.subplots(2, 2, figsize=(7.1, 3.25), dpi=600, sharex=True, sharey=True)
    line_halo = [pe.Stroke(linewidth=2.6, foreground="white"), pe.Normal()]
    all_ratios = []
    for ax, (title, filename) in zip(axes.flat, SCENARIOS):
        results = load_scenario(filename)
        for idx, (
            method_key,
            method_label,
            color,
            marker,
            linestyle,
            facecolor,
            x_offset,
        ) in enumerate(LINE_METHODS):
            tenants, ratios = normalized_series(results, method_key, "avg_jct")
            all_ratios.extend(ratios)
            x_values = [tenant + x_offset for tenant in tenants]
            ax.plot(
                x_values,
                ratios,
                label=method_label,
                color=color,
                marker=marker,
                linestyle=linestyle,
                linewidth=1.25,
                markersize=4.7,
                markerfacecolor=facecolor,
                markeredgewidth=0.85,
                markeredgecolor=color,
                solid_capstyle="round",
                dash_capstyle="round",
                path_effects=line_halo,
                zorder=10 + idx,
            )
        ax.axhline(1.0, color="#8A8A8A", linewidth=0.65, linestyle=":")
        ax.set_title(scenario_panel_title(title), pad=3.0)
        ax.set_xticks([3, 4, 5, 6, 7, 8])
        ax.set_xlim(2.72, 8.28)
        ax.set_ylim(0.2, 1.08)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.grid(axis="y", color="#E0E0E0", linewidth=0.5)
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        ax.spines["left"].set_linewidth(0.7)
        ax.spines["bottom"].set_linewidth(0.7)
        ax.tick_params(axis="both", width=0.7, length=3.0)

    axes[0, 0].set_ylim(min(0.2, min(all_ratios) * 0.9), max(1.08, max(all_ratios) * 1.04))
    for ax in axes[:, 0]:
        ax.set_ylabel("Normalized Avg. JCT")
    for ax in axes[-1, :]:
        ax.set_xlabel("Number of tenants")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.935),
        ncol=4 if len(LINE_METHODS) > 6 else 6,
        frameon=False,
        columnspacing=0.85,
        handlelength=1.55,
        handletextpad=0.3,
    )
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.81 if len(LINE_METHODS) > 6 else 0.865], pad=0.24)
    save_all(fig, "main_avg_jct_by_tenant")
    plt.close(fig)


def write_summary_table() -> None:
    method_keys = [method[0] for method in METHODS]
    method_labels = [method[1] for method in METHODS]
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Average reduction relative to default placement across tenant counts 3--8.}",
        r"\label{tab:main-result-summary}",
        r"\begin{tabular}{ll" + "r" * len(method_keys) + "}",
        r"\toprule",
        "Scenario & Metric & " + " & ".join(method_labels) + r" \\",
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
    col_widths = [0.245, 0.105] + [0.65 / len(method_keys)] * len(method_keys)
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
    global METHODS, LINE_METHODS
    # Permit plotting original six-method snapshots as well as augmented results.
    available = set.intersection(*[
        set(item["averages"]) for _, filename in SCENARIOS for item in load_scenario(filename)
    ])
    METHODS = [method for method in METHODS if method[0] in available]
    LINE_METHODS = [method for method in LINE_METHODS if method[0] in available]
    if len(LINE_METHODS) > 6:
        LINE_METHODS = [(*method[:-1], 0.048 * (idx - (len(LINE_METHODS) - 1) / 2))
                        for idx, method in enumerate(LINE_METHODS)]
    OUT_DIR.mkdir(exist_ok=True)
    set_publication_style()
    plot_metric(
        metric="avg_jct",
        ylabel="Norm. avg. completion",
        stem="main_avg_jct_normalized",
        ylim=(0.0, 1.08),
    )
    plot_avg_jct_by_tenant()
    plot_metric(
        metric="makespan",
        ylabel="Normalized makespan",
        stem="main_makespan_normalized",
        ylim=(0.0, 1.15),
    )
    write_summary_table()


if __name__ == "__main__":
    main()
