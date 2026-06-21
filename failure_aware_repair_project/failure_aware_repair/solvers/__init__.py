"""Small solver helpers used by failure-aware repair experiments."""

from .contention_guided_repair import (
    ContentionGuidedRepairOptimizer,
    EstimatorRepairCandidate,
    RepairSearchConfig,
    estimator_repair_candidates,
    solve_contention_guided_repair,
)
from .nearest_protection_baseline import (
    NearestProtectionBaselineResult,
    NearestProtectionBaselineSolver,
    solve_nearest_protection_baseline,
)
from .optimized_single_failover import (
    OptimizedSingleFailoverResult,
    OptimizedSingleFailoverSolver,
    solve_optimized_single_failover,
)

__all__ = [
    "ContentionGuidedRepairOptimizer",
    "EstimatorRepairCandidate",
    "NearestProtectionBaselineResult",
    "NearestProtectionBaselineSolver",
    "OptimizedSingleFailoverResult",
    "OptimizedSingleFailoverSolver",
    "RepairSearchConfig",
    "estimator_repair_candidates",
    "solve_contention_guided_repair",
    "solve_nearest_protection_baseline",
    "solve_optimized_single_failover",
]
