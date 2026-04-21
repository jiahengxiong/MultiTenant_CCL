from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


SCHEME_ORDER = [
    ("baseline_random_mapping", "Baseline Default Mapping", "#6B7280"),
    ("harmonics_baseline", "Harmonics Baseline", "#C0841A"),
    ("proposed_mapping_cg", "Proposed Mapping (CG)", "#2563EB"),
    ("proposed_mapping_cg_plus_harmonics", "Proposed Mapping (CG) + Harmonics Baseline", "#059669"),
]


def save_baseline_vs_proposed_mapping_figures(payload: dict, output_dir: str | Path) -> list[Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results_by_tenant_count = payload["results_by_tenant_count"]
    tenant_counts = sorted(int(key) for key in results_by_tenant_count.keys())

    figure_specs = [
        ("makespan", "Baseline vs Proposed Mapping: Makespan", "Makespan (s)", "baseline_vs_proposed_mapping_makespan.png"),
        ("avg_jct", "Baseline vs Proposed Mapping: Average JCT", "Average JCT (s)", "baseline_vs_proposed_mapping_avg_jct.png"),
    ]

    written_files = []
    for metric_key, title, ylabel, filename in figure_specs:
        x = np.arange(len(tenant_counts))
        width = 0.2
        fig, ax = plt.subplots(figsize=(11, 6))

        for index, (scheme_key, label, color) in enumerate(SCHEME_ORDER):
            values = [
                results_by_tenant_count[str(tenant_count)][scheme_key][metric_key]
                for tenant_count in tenant_counts
            ]
            offset = (index - 1.5) * width
            ax.bar(x + offset, values, width, label=label, color=color)

        ax.set_xlabel("Number of Tenants")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels([str(tenant_count) for tenant_count in tenant_counts])
        ax.legend()
        ax.grid(axis="y", linestyle="--", alpha=0.7)

        plt.tight_layout()
        output_path = output_dir / filename
        plt.savefig(output_path)
        plt.close(fig)
        written_files.append(output_path)

    return written_files
