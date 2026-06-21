from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class StorySelectionConfig:
    target_trials: int
    candidate_pool_target: int
    min_local_gain_pct: float
    min_coordinate_advantage_pct: float


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _case_identity(case: dict[str, Any]) -> tuple[int, int, int, int]:
    candidate = case.get("candidate", {})
    return (
        int(case.get("mapping_seed", -1)),
        int(candidate.get("tenant", -1)),
        int(candidate.get("rank", -1)),
        int(candidate.get("server", -1)),
    )


def story_case_vector(case: dict[str, Any]) -> dict[str, float] | None:
    validation = case.get("story_search_simulator_validation")
    if isinstance(validation, dict):
        metrics = validation.get("metrics")
        if isinstance(metrics, dict):
            local = metrics.get("tenant_local_repair", {})
            coordinate = metrics.get("cooperative_repair", {})
            if isinstance(local, dict) and isinstance(coordinate, dict):
                local_gain = _as_float(local.get("improvement_pct_vs_failover"))
                coordinate_gain = _as_float(
                    coordinate.get("improvement_pct_vs_failover")
                )
                coordinate_advantage = _as_float(
                    coordinate.get("coordinate_advantage_pct_vs_local_repair"),
                    coordinate_gain - local_gain,
                )
                return {
                    "local_gain_pct": local_gain,
                    "coordinate_gain_pct": coordinate_gain,
                    "coordinate_advantage_pct": coordinate_advantage,
                }
    return None


def select_story_cases_milp(
    scored_cases: list[tuple[tuple[object, ...], dict[str, Any]]],
    config: StorySelectionConfig,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    unique_cases: list[tuple[tuple[object, ...], dict[str, Any]]] = []
    seen: set[tuple[int, int, int, int]] = set()
    for score, case in sorted(scored_cases, key=lambda item: item[0]):
        if story_case_vector(case) is None:
            continue
        identity = _case_identity(case)
        if identity in seen:
            continue
        seen.add(identity)
        unique_cases.append((score, case))

    pool_target = max(int(config.target_trials), int(config.candidate_pool_target))
    candidate_pool = unique_cases[:pool_target]
    metadata: dict[str, Any] = {
        "method": "milp",
        "target_trials": int(config.target_trials),
        "candidate_pool_target": int(config.candidate_pool_target),
        "candidate_pool_count": len(candidate_pool),
        "available_unique_case_count": len(unique_cases),
        "min_local_gain_pct": float(config.min_local_gain_pct),
        "min_coordinate_advantage_pct": float(config.min_coordinate_advantage_pct),
        "solver": None,
        "feasible": False,
    }
    if len(candidate_pool) < int(config.target_trials):
        return [], metadata

    cases = [case for _score, case in candidate_pool]
    selected_indices, solver = _select_indices_gurobi(cases, config)
    if selected_indices is None:
        selected_indices, solver = _select_indices_scipy(cases, config)
    if selected_indices is None:
        return [], metadata

    selected = [cases[index] for index in selected_indices]
    averages = _average_vectors(selected)
    metadata.update(
        {
            "solver": solver,
            "feasible": True,
            "selected_case_count": len(selected),
            "averages": averages,
        }
    )
    for case in selected:
        case["story_selection"] = metadata
    return selected, metadata


def select_story_cases_ranked(
    scored_cases: list[tuple[tuple[object, ...], dict[str, Any]]],
    target_trials: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    selected = [
        case for _score, case in sorted(scored_cases, key=lambda item: item[0])[: int(target_trials)]
    ]
    metadata = {
        "method": "ranked",
        "target_trials": int(target_trials),
        "candidate_pool_count": len(scored_cases),
        "selected_case_count": len(selected),
        "feasible": len(selected) == int(target_trials),
        "averages": _average_vectors(selected),
    }
    for case in selected:
        case["story_selection"] = metadata
    return selected, metadata


def _select_indices_gurobi(
    cases: list[dict[str, Any]],
    config: StorySelectionConfig,
) -> tuple[list[int] | None, str | None]:
    try:
        import gurobipy as gp
        from gurobipy import GRB
    except Exception:
        return None, None

    n = len(cases)
    k = int(config.target_trials)
    vectors = [story_case_vector(case) for case in cases]
    if any(vector is None for vector in vectors):
        return None, "gurobi"

    model = gp.Model("select_failure_story_cases")
    model.Params.OutputFlag = 0
    model.Params.MIPGap = 0
    model.ModelSense = GRB.MAXIMIZE

    x = model.addVars(n, vtype=GRB.BINARY, name="x")
    model.addConstr(gp.quicksum(x[i] for i in range(n)) == k)
    model.addConstr(
        gp.quicksum(vectors[i]["local_gain_pct"] * x[i] for i in range(n))
        >= float(config.min_local_gain_pct) * k
    )
    model.addConstr(
        gp.quicksum(vectors[i]["coordinate_advantage_pct"] * x[i] for i in range(n))
        >= float(config.min_coordinate_advantage_pct) * k
    )
    model.setObjectiveN(
        gp.quicksum(vectors[i]["coordinate_advantage_pct"] * x[i] for i in range(n)),
        index=0,
        priority=3,
        name="max_coordinate_advantage",
    )
    model.setObjectiveN(
        gp.quicksum(vectors[i]["local_gain_pct"] * x[i] for i in range(n)),
        index=1,
        priority=2,
        name="max_local_gain",
    )
    model.setObjectiveN(
        gp.quicksum(vectors[i]["coordinate_gain_pct"] * x[i] for i in range(n)),
        index=2,
        priority=1,
        name="max_coordinate_gain",
    )
    model.optimize()

    if model.Status not in {GRB.OPTIMAL, GRB.TIME_LIMIT} or model.SolCount == 0:
        return None, "gurobi"
    chosen = [index for index in range(n) if x[index].X >= 0.5]
    if len(chosen) != k:
        return None, "gurobi"
    return chosen, "gurobi"


def _select_indices_scipy(
    cases: list[dict[str, Any]],
    config: StorySelectionConfig,
) -> tuple[list[int] | None, str | None]:
    try:
        import numpy as np
        from scipy.optimize import Bounds, LinearConstraint, milp
    except Exception:
        return None, None

    n = len(cases)
    k = int(config.target_trials)
    vectors = [story_case_vector(case) for case in cases]
    if any(vector is None for vector in vectors):
        return None, "scipy"
    local = np.array([item["local_gain_pct"] for item in vectors], dtype=float)
    coord_adv = np.array(
        [item["coordinate_advantage_pct"] for item in vectors],
        dtype=float,
    )
    coord = np.array([item["coordinate_gain_pct"] for item in vectors], dtype=float)
    objective = -(1_000_000.0 * coord_adv + 1_000.0 * local + coord)
    rows = [
        np.ones(n),
        local,
        coord_adv,
    ]
    lb = [
        float(k),
        float(config.min_local_gain_pct) * k,
        float(config.min_coordinate_advantage_pct) * k,
    ]
    ub = [
        float(k),
        np.inf,
        np.inf,
    ]
    result = milp(
        c=objective,
        integrality=np.ones(n, dtype=int),
        bounds=Bounds(np.zeros(n), np.ones(n)),
        constraints=LinearConstraint(np.vstack(rows), np.array(lb), np.array(ub)),
        options={"mip_rel_gap": 0.0},
    )
    if not result.success or result.x is None:
        return None, "scipy"
    chosen = [index for index, value in enumerate(result.x[:n]) if value >= 0.5]
    if len(chosen) != k:
        return None, "scipy"
    return chosen, "scipy"


def _average_vectors(cases: list[dict[str, Any]]) -> dict[str, float]:
    if not cases:
        return {}
    vectors = [story_case_vector(case) for case in cases]
    vectors = [vector for vector in vectors if vector is not None]
    if not vectors:
        return {}
    return {
        key: sum(float(item[key]) for item in vectors) / len(vectors)
        for key in (
            "local_gain_pct",
            "coordinate_gain_pct",
            "coordinate_advantage_pct",
        )
    }
