"""Input-only adaptation of the existing Harmonics heuristic to QP policies.

The inherited offset-search algorithm is unchanged. Its stage metadata and
simulator callbacks observe the same per-message QP paths and source queues.
No global monkey patches or changes to the simulator are used.
"""
from collections import defaultdict, deque
from dataclasses import replace

from multitenant.baselines.harmonics import HarmonicsBaselineHeuristic, _build_program_metadata
from multitenant.simulator.adapter import _run_simulation_worker_result
from multitenant.workloads import build_collective_program_schedule


class _MetadataPaths:
    """Supply task-specific paths to the existing one-pass metadata compiler."""

    def __init__(self, mapping, programs, policy):
        paths_by_name = {str(entry.chunk_id): entry.path for entry in policy}
        self.paths = defaultdict(deque)
        schedule = build_collective_program_schedule(mapping, programs)
        for tenant in sorted(mapping):
            for task in schedule[tenant]["tasks"]:
                key = (int(tenant), mapping[tenant][int(task["src_rank"])],
                       mapping[tenant][int(task["dst_rank"])])
                self.paths[key].append(paths_by_name[str(task["name"])])

    def get(self, key, default=None):
        if not self.paths[key]:
            raise ValueError(f"Metadata compiler requested an unexpected task path: {key}")
        return self.paths[key].popleft()


class QPHarmonicsHeuristic(HarmonicsBaselineHeuristic):
    def __init__(self, datacenter, mapping, specs, sim_topology, policy, initial_result=None,
                 program_time_limit_s=None):
        kwargs = {} if program_time_limit_s is None else {"program_time_limit_s": program_time_limit_s}
        super().__init__(datacenter, mapping, tenant_collective_specs=specs, verbose=False, **kwargs)
        if not self._is_single_collective_program():
            raise ValueError("This adapter supports the main experiments' single dominant collective only")
        self.qp_topology = sim_topology
        self.qp_policy = policy
        self.qp_cache = {}
        self.qp_simulation_count = 0
        self.qp_cache_hits = 0
        paths = _MetadataPaths(mapping, self.tenant_collective_programs, policy)
        self._metadata = _build_program_metadata(
            datacenter, mapping, None, paths,
            tenant_collective_programs=self.tenant_collective_programs,
        )
        assert all(not remaining for remaining in paths.paths.values())
        if initial_result is not None:
            self.qp_cache[self._simulation_key(sorted(mapping), {})] = self._compact(initial_result)

    @staticmethod
    def _compact(result):
        return {key: result[key] for key in
                ("avg_tenant_makespan", "global_makespan", "tenant_makespans")}

    @staticmethod
    def _simulation_key(tenants, starts):
        return tuple((int(t), float(starts.get(int(t), {}).get(0, 0.0))) for t in sorted(tenants))

    def _simulate_qp(self, tenants, starts):
        key = self._simulation_key(tenants, starts)
        if key in self.qp_cache:
            self.qp_cache_hits += 1
            return self.qp_cache[key]
        active = set(int(t) for t in tenants)
        policy = [
            replace(entry, time=entry.time + float(starts.get(int(str(entry.chunk_id).split("-", 1)[0]), {}).get(0, 0.0)))
            for entry in self.qp_policy
            if int(str(entry.chunk_id).split("-", 1)[0]) in active
        ]
        result = self._compact(_run_simulation_worker_result(self.qp_topology, policy))
        self.qp_simulation_count += 1
        self.qp_cache[key] = result
        return result

    def _simulate_single_collective_subset(self, tenants):
        return self._simulate_qp(tenants, {})

    def _simulator_single_collective_score(self, starts):
        result = self._simulate_qp(sorted(self.tenant_mapping), starts)
        return (float(result["avg_tenant_makespan"]), float(result["global_makespan"]),
                {int(t): float(v) for t, v in result["tenant_makespans"].items()})

    def final_result(self):
        return self._simulate_qp(sorted(self.tenant_mapping), self.collective_start_times)
