from __future__ import annotations

from .mapping_time_expanded_optimizer import MappingTimeExpandedEstimatorOptimizer


class MappingEstimatorBlackBoxOptimizer(MappingTimeExpandedEstimatorOptimizer):
    """Compatibility entry point for the standalone time-expanded optimizer.

    The implementation lives in ``mapping_time_expanded_optimizer.py`` so the
    optimizer/search policy is separated from both the estimator and the legacy
    collapsed hybrid mapper.
    """

    def __init__(self, *args, **kwargs):
        for legacy_key in (
            "collapsed_proposal_time_limit",
            "use_collapsed_proposal",
            "surrogate_mode",
            "validate_with_simulator",
            "miqp_pricing_time_limit",
        ):
            kwargs.pop(legacy_key, None)
        super().__init__(*args, **kwargs)
