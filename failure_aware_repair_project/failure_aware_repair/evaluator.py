from __future__ import annotations

from dataclasses import dataclass

from multitenant.simulator import simulate_collective
from multitenant.solvers import TimeExpandedContentionEstimator

from .models import Mapping, RepairObjective, RepairScenario
from .protection import build_failover_mapping, copy_mapping, count_switches


@dataclass
class RepairEvaluator:
    """Estimator and simulator bridge for repair mappings."""

    datacenter: object
    scenario: RepairScenario
    single_flow_size_bits: int | None = None
    collective: str | None = None
    tenant_collective_specs: dict[int, dict[str, object]] | None = None
    tenant_collective_programs: dict[int, list[dict[str, object]]] | None = None
    tenant_start_times: dict[int, float] | None = None
    slot_duration: float | None = None
    horizon_slots: int | None = None
    failover_policy: str = "same_leaf_or_nearest"
    failover_mapping: Mapping | None = None

    def __post_init__(self) -> None:
        if self.failover_mapping is None:
            self.failover_mapping = build_failover_mapping(
                self.scenario,
                datacenter=self.datacenter,
                policy=self.failover_policy,
            )
        else:
            self.failover_mapping = copy_mapping(self.failover_mapping)
        workload = self.scenario.workload
        estimator_specs = self.tenant_collective_specs
        estimator_programs = self.tenant_collective_programs
        tenant_start_times = self.tenant_start_times
        collective = self.collective
        single_flow_size_bits = self.single_flow_size_bits
        if workload is not None:
            if estimator_specs is None:
                estimator_specs = workload.tenant_collective_specs
            if estimator_programs is None:
                estimator_programs = workload.tenant_collective_programs
            if tenant_start_times is None:
                tenant_start_times = workload.tenant_start_times
            if collective is None:
                collective = workload.collective
            if single_flow_size_bits is None:
                single_flow_size_bits = workload.single_flow_size_bits

        if estimator_specs is None and estimator_programs is None:
            if collective is None or single_flow_size_bits is None:
                raise ValueError(
                    "RepairEvaluator needs either tenant_collective_specs, "
                    "tenant_collective_programs, or both collective and single_flow_size_bits"
                )
            estimator_specs = {
                int(tenant): {
                    "collective": str(collective),
                    "single_flow_size_bits": int(single_flow_size_bits),
                }
                for tenant in self.scenario.pre_failure_mapping
            }
        self.tenant_collective_specs = estimator_specs
        self.tenant_collective_programs = estimator_programs
        self.tenant_start_times = tenant_start_times
        self.collective = collective
        self.single_flow_size_bits = single_flow_size_bits
        # Build paths for all servers and all tenants.  This keeps routing fixed
        # while allowing candidate mappings to include protection nodes.
        self.path_table = self.datacenter.build_tenant_ecmp_path_table(
            sorted(self.scenario.pre_failure_mapping)
        )
        self.estimator = TimeExpandedContentionEstimator(
            self.datacenter,
            tenant_mapping=self.failover_mapping,
            tenant_collective_specs=estimator_specs,
            tenant_collective_programs=estimator_programs,
            tenant_start_times=tenant_start_times,
            path_table=self.path_table,
            slot_duration=self.slot_duration,
            horizon_slots=self.horizon_slots,
            name="failure_aware_repair_estimator",
        )
        self._estimator_specs = estimator_specs
        self.compiled_dag_data = self.estimator._backbone.data
        self._pipeline_score_cache: dict[tuple, tuple[float, float]] = {}
        self._simulation_score_cache: dict[tuple, tuple[float, float]] = {}

    def estimate(self, mapping: Mapping) -> RepairObjective:
        estimate = self.estimator.evaluate(mapping)
        switches = count_switches(
            pre_failure_mapping=self.scenario.pre_failure_mapping,
            failover_mapping=self.failover_mapping,
            repaired_mapping=mapping,
            failure=self.scenario.failure,
        )
        return RepairObjective(
            avg_jct=float(estimate.avg_jct),
            makespan=float(estimate.makespan),
            extra_switches=int(switches.extra_vs_failover),
        )

    def analyze(self, mapping: Mapping):
        return self.estimator.analyze_for_search(mapping)

    def pipeline_score(self, mapping: Mapping) -> tuple[float, float]:
        normalized = self.estimator.normalize_mapping(mapping)
        signature = self.estimator.signature(normalized)
        cached = self._pipeline_score_cache.get(signature)
        if cached is not None:
            return cached
        score = tuple(float(value) for value in self.estimator.pipeline_score(normalized))
        self._pipeline_score_cache[signature] = score
        if len(self._pipeline_score_cache) > 2048:
            self._pipeline_score_cache.clear()
            self._pipeline_score_cache[signature] = score
        return score

    def simulate(self, mapping: Mapping) -> tuple[float, float]:
        normalized = self.estimator.normalize_mapping(mapping)
        signature = self.estimator.signature(normalized)
        cached = self._simulation_score_cache.get(signature)
        if cached is not None:
            return cached
        makespan, avg_jct = simulate_collective(
            self.datacenter.topology,
            normalized,
            self.path_table,
            self.single_flow_size_bits,
            self.collective,
            tenant_start_times=self.tenant_start_times,
            tenant_collective_specs=self._estimator_specs,
            tenant_collective_programs=self.tenant_collective_programs,
        )
        score = (float(makespan), float(avg_jct))
        self._simulation_score_cache[signature] = score
        if len(self._simulation_score_cache) > 512:
            self._simulation_score_cache.clear()
            self._simulation_score_cache[signature] = score
        return score

    def switch_counts(self, mapping: Mapping):
        return count_switches(
            pre_failure_mapping=self.scenario.pre_failure_mapping,
            failover_mapping=self.failover_mapping,
            repaired_mapping=mapping,
            failure=self.scenario.failure,
        )

    def dag_summary(self) -> dict[str, object]:
        data = self.compiled_dag_data
        per_tenant = data.get("compiled_schedule", {}).get("per_tenant", {})
        task_surrogate = data.get("task_surrogate", {})
        return {
            "source": "multitenant.solvers.DAG_generation.build_collective_dag_data",
            "tenants": [int(tenant) for tenant in data.get("M", [])],
            "task_count_by_tenant": {
                str(tenant): len(meta.get("task_info", {}))
                for tenant, meta in task_surrogate.items()
            },
            "max_epoch_by_tenant": {
                str(tenant): int(meta.get("max_epoch", -1))
                for tenant, meta in per_tenant.items()
            },
            "global_max_epoch": int(data.get("compiled_schedule", {}).get("global_max_epoch", -1)),
            "has_task_surrogate": bool(task_surrogate),
        }
