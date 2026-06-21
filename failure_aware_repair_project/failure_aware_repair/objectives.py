from __future__ import annotations

from .models import RepairObjective


def repair_better(candidate: RepairObjective, incumbent: RepairObjective, *, tol: float = 1e-6) -> bool:
    if candidate.avg_jct < incumbent.avg_jct - tol:
        return True
    if candidate.avg_jct > incumbent.avg_jct + tol:
        return False
    if candidate.makespan < incumbent.makespan - tol:
        return True
    if candidate.makespan > incumbent.makespan + tol:
        return False
    return int(candidate.extra_switches) < int(incumbent.extra_switches)


def repair_sort_key(objective: RepairObjective) -> tuple[float, float, int]:
    return (
        round(float(objective.avg_jct), 6),
        round(float(objective.makespan), 6),
        int(objective.extra_switches),
    )
