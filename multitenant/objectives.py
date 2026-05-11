from __future__ import annotations

ObjectivePair = tuple[float, float]


def lexicographic_better(
    candidate: ObjectivePair,
    incumbent: ObjectivePair,
    *,
    tol: float = 1e-9,
) -> bool:
    """Compare (makespan, avg_jct) with avg_jct as the primary objective."""

    cand_makespan, cand_avg_jct = candidate
    inc_makespan, inc_avg_jct = incumbent
    if cand_avg_jct < inc_avg_jct - tol:
        return True
    if cand_avg_jct > inc_avg_jct + tol:
        return False
    return cand_makespan < inc_makespan - tol


def lexicographic_lower_bound_can_improve(
    lower_bound: ObjectivePair,
    incumbent: ObjectivePair,
    *,
    tol: float = 1e-9,
) -> bool:
    """Return whether a lexicographic lower bound can still beat incumbent."""

    lb_makespan, lb_avg_jct = lower_bound
    inc_makespan, inc_avg_jct = incumbent
    if lb_avg_jct > inc_avg_jct + tol:
        return False
    if lb_avg_jct < inc_avg_jct - tol:
        return True
    return lb_makespan < inc_makespan - tol
