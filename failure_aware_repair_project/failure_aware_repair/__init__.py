"""Failure-aware collective structure repair.

This package is intentionally isolated from the original project modules.  It
uses the existing topology, workload, estimator, and simulator APIs as
dependencies, while keeping all repair-specific code in this new project
directory.
"""

from .evaluator import RepairEvaluator
from .heuristic import CooperativeRepairHeuristic, TenantLocalRepairHeuristic
from .milp import (
    CollapsedRepairMILPSolver,
    ExactRepairEnumerator,
    FailureAwareRepairTimeExpandedILPSolver,
    ProxyRepairMILPSolver,
)
from .models import (
    FailureEvent,
    RepairMode,
    RepairObjective,
    RepairResult,
    RepairScenario,
    RepairWorkload,
    SwitchCounts,
)
from .problem import FailureAwareMappingProblem
from .protection import (
    build_candidate_server_sets,
    build_failover_mapping,
    check_repair_feasible,
    count_switches,
    failover_strategy_constraints,
    infer_failed_rank,
    repair_strategy_constraints,
    validate_global_protection_pool,
    working_nodes_by_tenant,
)
from .strategies import (
    COOPERATIVE_REPAIR,
    REPAIR_FAILED_SERVER_ONLY,
    REPAIR_STRATEGIES,
    REPAIR_STRATEGY_BY_NAME,
    TENANT_LOCAL_REPAIR,
    RepairStrategySpec,
)
from .strategy_solver import (
    FailureAwareMappingStrategySolver,
    StrategySolveResult,
    solve_repair_strategies,
    solve_repair_strategy,
)

__all__ = [
    "CooperativeRepairHeuristic",
    "CollapsedRepairMILPSolver",
    "ExactRepairEnumerator",
    "FailureAwareRepairTimeExpandedILPSolver",
    "FailureAwareMappingProblem",
    "FailureAwareMappingStrategySolver",
    "FailureEvent",
    "RepairEvaluator",
    "RepairMode",
    "RepairObjective",
    "RepairResult",
    "RepairScenario",
    "RepairStrategySpec",
    "RepairWorkload",
    "SwitchCounts",
    "StrategySolveResult",
    "TenantLocalRepairHeuristic",
    "ProxyRepairMILPSolver",
    "COOPERATIVE_REPAIR",
    "REPAIR_FAILED_SERVER_ONLY",
    "REPAIR_STRATEGIES",
    "REPAIR_STRATEGY_BY_NAME",
    "TENANT_LOCAL_REPAIR",
    "build_candidate_server_sets",
    "build_failover_mapping",
    "check_repair_feasible",
    "count_switches",
    "failover_strategy_constraints",
    "infer_failed_rank",
    "repair_strategy_constraints",
    "solve_repair_strategies",
    "solve_repair_strategy",
    "validate_global_protection_pool",
    "working_nodes_by_tenant",
]
