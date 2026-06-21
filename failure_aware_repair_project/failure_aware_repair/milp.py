from __future__ import annotations

import itertools
import time
from dataclasses import dataclass

from multitenant.workloads import build_collective_program_stage_flows, build_collective_stage_flows
from .mapping_ilp_base import MappingILPSolver

from .evaluator import RepairEvaluator
from .models import Mapping, RepairResult, RepairScenario
from .objectives import repair_better, repair_sort_key
from .protection import (
    build_candidate_server_sets,
    check_repair_feasible,
    copy_mapping,
    count_switches,
    participating_tenants,
)


@dataclass
class ExactEnumerationConfig:
    max_assignments: int = 200_000
    time_limit_seconds: float | None = None


class ExactRepairEnumerator:
    """Small-case exact search under the true estimator-backed objective."""

    name = "exact_repair_enumerator"

    def __init__(
        self,
        scenario: RepairScenario,
        evaluator: RepairEvaluator,
        *,
        config: ExactEnumerationConfig | None = None,
    ):
        self.scenario = scenario
        self.evaluator = evaluator
        self.config = config or ExactEnumerationConfig()
        self.failover_mapping = copy_mapping(evaluator.failover_mapping)
        self.candidate_sets = build_candidate_server_sets(scenario)

    def solve(self) -> RepairResult:
        start = time.time()
        deadline = (
            float("inf")
            if self.config.time_limit_seconds is None
            else start + float(self.config.time_limit_seconds)
        )
        participants = set(participating_tenants(self.scenario))
        tenant_options = []
        for tenant in sorted(self.scenario.pre_failure_mapping):
            ranks = sorted(self.scenario.pre_failure_mapping[tenant])
            if int(tenant) not in participants:
                tenant_options.append([(int(tenant), dict(self.failover_mapping[int(tenant)]))])
                continue
            options = []
            for servers in itertools.permutations(self.candidate_sets[int(tenant)], len(ranks)):
                options.append((int(tenant), {int(rank): int(servers[idx]) for idx, rank in enumerate(ranks)}))
                if len(options) > self.config.max_assignments:
                    break
            tenant_options.append(options)

        best_mapping = copy_mapping(self.failover_mapping)
        best_obj = self.evaluator.estimate(best_mapping)
        evaluated = 1
        stopped = None
        for combination in itertools.product(*tenant_options):
            if time.time() >= deadline:
                stopped = "time_limit"
                break
            if evaluated >= self.config.max_assignments:
                stopped = "assignment_limit"
                break
            mapping = copy_mapping(self.failover_mapping)
            for tenant, tenant_mapping in combination:
                mapping[int(tenant)] = dict(tenant_mapping)
            try:
                check_repair_feasible(self.scenario, mapping, self.failover_mapping)
            except ValueError:
                continue
            obj = self.evaluator.estimate(mapping)
            evaluated += 1
            if repair_better(obj, best_obj):
                best_mapping = mapping
                best_obj = obj

        return RepairResult(
            name=self.name,
            mapping=copy_mapping(best_mapping),
            objective=best_obj,
            switch_counts=self.evaluator.switch_counts(best_mapping),
            runtime_seconds=time.time() - start,
            metadata={
                "evaluated_assignments": evaluated,
                "stopped": stopped,
                "sort_key": repair_sort_key(best_obj),
            },
        )

class FailureAwareRepairTimeExpandedILPSolver(MappingILPSolver):
    """Failure-aware extension of the original time-expanded MappingILPSolver.

    This class lives in the standalone repair project and does not edit
    `multitenant/solvers/mapping_ilp.py`.  It reuses the original solver's
    task-time model, path constraints, endpoint linearization, tenant finish
    variables, and Gurobi solve flow, while overriding only the parts needed for
    repair:

    - per-tenant candidate server sets may include protection nodes,
    - assignment is injective into a larger candidate set,
    - non-participating tenants are fixed to traditional failover,
    - the failed server is excluded,
    - healthy-rank movement variables add a third objective.
    """

    def __init__(
        self,
        scenario: RepairScenario,
        evaluator: RepairEvaluator,
        *,
        verbose: bool = False,
        name: str = "failure_aware_repair_time_expanded_ilp",
        **kwargs,
    ):
        self.repair_scenario = scenario
        self.repair_evaluator = evaluator
        self.repair_failover_mapping = copy_mapping(evaluator.failover_mapping)
        self.repair_candidate_server_sets = build_candidate_server_sets(scenario)
        self.repair_participants = set(participating_tenants(scenario))
        self.repair_move = {}
        self.repair_switch_terms = []
        failed_rank = scenario.failure.failed_rank
        if failed_rank is None:
            from .protection import infer_failed_rank

            failed_rank = infer_failed_rank(scenario.pre_failure_mapping, scenario.failure)
        self.repair_failed_rank = int(failed_rank)
        slot_duration = kwargs.pop("slot_duration", evaluator.slot_duration)
        horizon_slots = kwargs.pop("horizon_slots", evaluator.horizon_slots)

        super().__init__(
            evaluator.datacenter,
            self.repair_failover_mapping,
            verbose=verbose,
            name=name,
            collective=evaluator.collective,
            single_flow_size=evaluator.single_flow_size_bits,
            tenant_collective_specs=evaluator._estimator_specs,
            tenant_collective_programs=evaluator.tenant_collective_programs,
            tenant_start_times=evaluator.tenant_start_times,
            slot_duration=slot_duration,
            horizon_slots=horizon_slots,
            path_table=evaluator.path_table,
            **kwargs,
        )

    def _build_data(self):
        data = super()._build_data()
        data["S"] = {
            int(tenant): tuple(int(server) for server in self.repair_candidate_server_sets[int(tenant)])
            for tenant in data["M"]
        }
        return data

    def _use_ring_pattern_formulation(self, tenant):
        return False

    def _tenant_has_ring_rotation_symmetry(self, tenant):
        # Ring rotations are not symmetry-equivalent once switch count and
        # protection-node use are part of the repair objective.
        return False

    def _add_perm_constraints(self):
        import gurobipy as gp
        from gurobipy import GRB

        tenants, ranks, servers = self.data["M"], self.data["R"], self.data["S"]
        self.repair_move = {}
        self.repair_switch_terms = []
        protection = set(int(server) for server in self.repair_scenario.global_protection_pool)

        for tenant in tenants:
            tenant = int(tenant)
            for rank in ranks[tenant]:
                rank = int(rank)
                self.model.addConstr(
                    gp.quicksum(self.X[(tenant, rank, server)] for server in servers[tenant]) == 1,
                    name=f"repair_rank_one_{tenant}_{rank}",
                )
                failover_server = int(self.repair_failover_mapping[tenant][rank])
                for server in servers[tenant]:
                    server = int(server)
                    if server != failover_server and server not in protection:
                        self.model.addConstr(
                            self.X[(tenant, rank, server)] == 0,
                            name=f"repair_forbid_healthy_to_healthy_{tenant}_{rank}_{server}",
                        )
                if tenant not in self.repair_participants:
                    self.model.addConstr(
                        self.X[(tenant, rank, failover_server)] == 1,
                        name=f"repair_fixed_nonparticipant_{tenant}_{rank}",
                    )

                move_var = self.model.addVar(
                    vtype=GRB.BINARY,
                    name=f"RepairMove_{tenant}_{rank}",
                )
                self.repair_move[(tenant, rank)] = move_var
                if (tenant, rank) == (
                    int(self.repair_scenario.failure.tenant),
                    int(self.repair_failed_rank),
                ):
                    self.model.addConstr(
                        move_var == 0,
                        name=f"repair_ignore_unavoidable_failed_rank_{tenant}_{rank}",
                    )
                elif failover_server in servers[tenant]:
                    self.model.addConstr(
                        move_var == 1 - self.X[(tenant, rank, failover_server)],
                        name=f"repair_move_def_{tenant}_{rank}",
                    )
                    self.repair_switch_terms.append(move_var)
                else:
                    self.model.addConstr(
                        move_var == 1,
                        name=f"repair_move_for_missing_failover_server_{tenant}_{rank}",
                    )
                    self.repair_switch_terms.append(move_var)

        for tenant in tenants:
            tenant = int(tenant)
            for server in servers[tenant]:
                self.model.addConstr(
                    gp.quicksum(self.X[(tenant, rank, int(server))] for rank in ranks[tenant]) <= 1,
                    name=f"repair_srv_at_most_one_{tenant}_{int(server)}",
                )

        failed_tenant = int(self.repair_scenario.failure.tenant)
        failed_rank = int(self.repair_failed_rank)
        self.model.addConstr(
            gp.quicksum(
                self.X[(failed_tenant, failed_rank, int(server))]
                for server in servers[failed_tenant]
                if int(server) in protection
            )
            == 1,
            name="repair_failed_rank_exactly_one_protection",
        )
        all_candidate_servers = sorted(
            {int(server) for tenant in tenants for server in servers[int(tenant)]}
        )
        for server in all_candidate_servers:
            occupants = [
                self.X[(int(tenant), int(rank), server)]
                for tenant in tenants
                for rank in ranks[int(tenant)]
                if server in set(int(candidate) for candidate in servers[int(tenant)])
            ]
            if len(occupants) > 1:
                self.model.addConstr(
                    gp.quicksum(occupants) <= 1,
                    name=f"repair_global_server_at_most_one_{server}",
                )

    def _set_lexicographic_objective(self):
        import gurobipy as gp
        from gurobipy import GRB

        tenant_count = max(len(self.data["M"]), 1)
        avg_completion = gp.quicksum(self.tenant_finish[tenant] for tenant in self.data["M"]) / tenant_count
        self.T_max = self.model.addVar(
            vtype=GRB.CONTINUOUS,
            lb=0.0,
            name="T_max",
        )
        for tenant in self.data["M"]:
            self.model.addConstr(
                self.T_max >= self.tenant_finish[tenant],
                name=f"T_max_ge_{tenant}",
            )

        switch_count = gp.quicksum(self.repair_switch_terms)
        self.model.ModelSense = GRB.MINIMIZE
        self.model.setObjectiveN(avg_completion, index=0, priority=3, name="avg_jct")
        self.model.setObjectiveN(self.T_max, index=1, priority=2, name="makespan")
        self.model.setObjectiveN(switch_count, index=2, priority=1, name="switch_count")

    def solve(self, time_limit=None):
        result = super().solve(time_limit=time_limit)
        check_repair_feasible(
            self.repair_scenario,
            self.get_X_mapping(),
            self.repair_failover_mapping,
        )
        return result

    def to_repair_result(self) -> RepairResult:
        mapping = self.get_X_mapping()
        objective = self.repair_evaluator.estimate(mapping)
        return RepairResult(
            name="time_expanded_repair_ilp",
            mapping=copy_mapping(mapping),
            objective=objective,
            switch_counts=count_switches(
                pre_failure_mapping=self.repair_scenario.pre_failure_mapping,
                failover_mapping=self.repair_failover_mapping,
                repaired_mapping=mapping,
                failure=self.repair_scenario.failure,
            ),
            runtime_seconds=float(getattr(self.model, "Runtime", 0.0)),
            metadata={
                "gurobi_status": int(self.model.Status),
                "gurobi_sol_count": int(self.model.SolCount),
                "slot_duration": float(self.data["slot_duration"]),
                "ilp_final_avg_jct": self.final_avg_jct,
                "ilp_final_makespan": self.final_makespan,
            },
        )


class ProxyRepairMILPSolver:
    """Assignment-level proxy MILP for repair constraints.

    This lightweight model is kept only as a constraint sanity checker.  The
    paper-facing exact path is `FailureAwareRepairTimeExpandedILPSolver`, which
    integrates the repair constraints into the copied time-expanded
    `MappingILPSolver` base in this project directory.
    """

    name = "proxy_repair_milp"

    def __init__(self, scenario: RepairScenario, evaluator: RepairEvaluator):
        self.scenario = scenario
        self.evaluator = evaluator
        self.failover_mapping = copy_mapping(evaluator.failover_mapping)
        self.candidate_sets = build_candidate_server_sets(scenario)

    def solve(self, *, time_limit: float | None = None) -> RepairResult:
        try:
            import gurobipy as gp
            from gurobipy import GRB
        except Exception as exc:  # pragma: no cover - depends on local license/install.
            raise RuntimeError("gurobipy is required for ProxyRepairMILPSolver") from exc

        start = time.time()
        model = gp.Model("proxy_failure_aware_repair")
        self.model = model
        model.Params.OutputFlag = 0
        model.Params.Seed = 0
        model.Params.MIPFocus = 1
        model.Params.Heuristics = 0.8
        if time_limit is not None:
            model.Params.TimeLimit = float(time_limit)

        participants = set(participating_tenants(self.scenario))
        failed_rank = self.scenario.failure.failed_rank
        if failed_rank is None:
            from .protection import infer_failed_rank

            failed_rank = infer_failed_rank(
                self.scenario.pre_failure_mapping,
                self.scenario.failure,
            )
        x = {}
        move = {}
        proxy_cost_terms = []
        switch_terms = []
        protection = set(int(server) for server in self.scenario.global_protection_pool)

        # Static path-length proxy: shorter tenant-aware paths and fewer changes
        # are preferred.  Estimator evaluation is still used after solve.
        for tenant, ranks in self.scenario.pre_failure_mapping.items():
            tenant = int(tenant)
            ranks = {int(rank): int(server) for rank, server in ranks.items()}
            for rank in ranks:
                failover_server = int(self.failover_mapping[tenant][rank])
                for server in self.candidate_sets[tenant]:
                    x[(tenant, rank, int(server))] = model.addVar(
                        vtype=GRB.BINARY,
                        name=f"X_{tenant}_{rank}_{int(server)}",
                    )
                    if int(server) != failover_server and int(server) not in protection:
                        model.addConstr(
                            x[(tenant, rank, int(server))] == 0,
                            name=f"forbid_healthy_to_healthy_{tenant}_{rank}_{int(server)}",
                        )
                    path_penalty = abs(int(server) - failover_server)
                    proxy_cost_terms.append(path_penalty * x[(tenant, rank, int(server))])

                model.addConstr(
                    gp.quicksum(x[(tenant, rank, int(server))] for server in self.candidate_sets[tenant]) == 1,
                    name=f"rank_one_{tenant}_{rank}",
                )
                if tenant not in participants:
                    fixed = int(self.failover_mapping[tenant][rank])
                    model.addConstr(x[(tenant, rank, fixed)] == 1, name=f"fixed_{tenant}_{rank}")

                mv = model.addVar(vtype=GRB.BINARY, name=f"Move_{tenant}_{rank}")
                move[(tenant, rank)] = mv
                if (tenant, rank) == (int(self.scenario.failure.tenant), int(failed_rank)):
                    # The failed rank's unavoidable failover should not dominate
                    # the disruption objective.
                    model.addConstr(mv == 0, name=f"ignore_failed_move_{tenant}_{rank}")
                elif failover_server in self.candidate_sets[tenant]:
                    model.addConstr(mv == 1 - x[(tenant, rank, failover_server)], name=f"move_def_{tenant}_{rank}")
                    switch_terms.append(mv)

            for server in self.candidate_sets[tenant]:
                model.addConstr(
                    gp.quicksum(x[(tenant, rank, int(server))] for rank in ranks) <= 1,
                    name=f"server_at_most_one_{tenant}_{int(server)}",
                )

        failed_tenant = int(self.scenario.failure.tenant)
        failed_rank = int(failed_rank)
        model.addConstr(
            gp.quicksum(
                x[(failed_tenant, failed_rank, int(server))]
                for server in self.candidate_sets[failed_tenant]
                if int(server) in protection
            )
            == 1,
            name="failed_rank_exactly_one_protection",
        )
        all_candidate_servers = sorted(
            {int(server) for tenant in tenants for server in self.candidate_sets[int(tenant)]}
        )
        for server in all_candidate_servers:
            occupants = [
                x[(int(tenant), int(rank), server)]
                for tenant in tenants
                for rank in sorted(self.scenario.pre_failure_mapping[int(tenant)])
                if server in set(int(candidate) for candidate in self.candidate_sets[int(tenant)])
            ]
            if len(occupants) > 1:
                model.addConstr(
                    gp.quicksum(occupants) <= 1,
                    name=f"global_server_at_most_one_{server}",
                )

        model.ModelSense = GRB.MINIMIZE
        model.setObjectiveN(gp.quicksum(proxy_cost_terms), index=0, priority=2, name="path_proxy")
        model.setObjectiveN(gp.quicksum(switch_terms), index=1, priority=1, name="switch_count")
        model.optimize()

        mapping = copy_mapping(self.failover_mapping)
        if model.SolCount > 0:
            for tenant, ranks in self.scenario.pre_failure_mapping.items():
                tenant = int(tenant)
                for rank in ranks:
                    for server in self.candidate_sets[tenant]:
                        if x[(tenant, int(rank), int(server))].X > 0.5:
                            mapping[tenant][int(rank)] = int(server)
                            break

        check_repair_feasible(self.scenario, mapping, self.failover_mapping)
        objective = self.evaluator.estimate(mapping)
        return RepairResult(
            name=self.name,
            mapping=mapping,
            objective=objective,
            switch_counts=count_switches(
                pre_failure_mapping=self.scenario.pre_failure_mapping,
                failover_mapping=self.failover_mapping,
                repaired_mapping=mapping,
                failure=self.scenario.failure,
            ),
            runtime_seconds=time.time() - start,
            metadata={
                "gurobi_status": int(model.Status),
                "gurobi_sol_count": int(model.SolCount),
            },
        )


class CollapsedRepairMILPSolver:
    """Exact MILP for a collapsed collective-stage repair surrogate.

    The model is independent from the original codebase.  It keeps the same
    repair constraints as the heuristic and optimizes a linearized stage
    bottleneck surrogate:

    - rank-to-server assignment is binary,
    - per-flow endpoint products are linearized,
    - each global stage duration is the max normalized load across links,
      senders, and receivers,
    - tenant completion is the cumulative stage duration through that tenant's
      final collective stage,
    - objectives are Avg JCT, makespan, then extra healthy-rank switches.

    Final reported objectives are still evaluated with the existing
    time-expanded estimator through `RepairEvaluator`.
    """

    name = "collapsed_repair_milp"

    def __init__(self, scenario: RepairScenario, evaluator: RepairEvaluator):
        self.scenario = scenario
        self.evaluator = evaluator
        self.failover_mapping = copy_mapping(evaluator.failover_mapping)
        self.candidate_sets = build_candidate_server_sets(scenario)

    def solve(self, *, time_limit: float | None = None, verbose: bool = False) -> RepairResult:
        try:
            import gurobipy as gp
            from gurobipy import GRB
        except Exception as exc:  # pragma: no cover - depends on local license/install.
            raise RuntimeError("gurobipy is required for CollapsedRepairMILPSolver") from exc

        start = time.time()
        model = gp.Model("collapsed_failure_aware_repair")
        self.model = model
        model.Params.OutputFlag = 1 if verbose else 0
        model.Params.Seed = 0
        model.Params.MIPFocus = 1
        model.Params.Heuristics = 0.8
        if time_limit is not None:
            model.Params.TimeLimit = float(time_limit)

        tenants = sorted(self.scenario.pre_failure_mapping)
        participants = set(participating_tenants(self.scenario))
        failed_rank = self.scenario.failure.failed_rank
        if failed_rank is None:
            from .protection import infer_failed_rank

            failed_rank = infer_failed_rank(
                self.scenario.pre_failure_mapping,
                self.scenario.failure,
            )

        stage_flows = self._stage_flows()
        max_epoch = max((len(stages) for stages in stage_flows.values()), default=0)
        capacities = {
            (int(src), int(dst)): float(attrs["capacity"])
            for src, dst, attrs in self.evaluator.datacenter.topology.edges(data=True)
        }
        server_send_capacity = {}
        server_recv_capacity = {}
        for server in self.evaluator.datacenter.get_all_servers():
            leaf = int(self.evaluator.datacenter.get_server_leaf(int(server)))
            server_send_capacity[int(server)] = float(capacities[(int(server), leaf)])
            server_recv_capacity[int(server)] = float(capacities[(leaf, int(server))])

        x = {}
        y = {}
        z = {}
        move = {}
        u = {}
        switch_terms = []
        protection = set(int(server) for server in self.scenario.global_protection_pool)

        for tenant in tenants:
            tenant = int(tenant)
            ranks = sorted(int(rank) for rank in self.scenario.pre_failure_mapping[tenant])
            for rank in ranks:
                original_server = int(self.scenario.pre_failure_mapping[tenant][rank])
                failover_server = int(self.failover_mapping[tenant][rank])
                y[(tenant, rank)] = model.addVar(
                    vtype=GRB.BINARY,
                    name=f"Y_SelectProtection_{tenant}_{rank}",
                )
                for server in self.candidate_sets[tenant]:
                    server = int(server)
                    x[(tenant, rank, int(server))] = model.addVar(
                        vtype=GRB.BINARY,
                        name=f"X_{tenant}_{rank}_{int(server)}",
                    )
                    if server in protection:
                        z[(tenant, rank, server)] = model.addVar(
                            vtype=GRB.BINARY,
                            name=f"Z_AssignProtection_{tenant}_{rank}_{server}",
                        )
                        model.addConstr(
                            x[(tenant, rank, server)] == z[(tenant, rank, server)],
                            name=f"x_equals_z_{tenant}_{rank}_{server}",
                        )
                    elif server != original_server:
                        model.addConstr(
                            x[(tenant, rank, server)] == 0,
                            name=f"forbid_healthy_to_healthy_{tenant}_{rank}_{int(server)}",
                        )
                model.addConstr(
                    gp.quicksum(x[(tenant, rank, int(server))] for server in self.candidate_sets[tenant]) == 1,
                    name=f"rank_one_{tenant}_{rank}",
                )
                protection_options = [
                    int(server)
                    for server in self.candidate_sets[tenant]
                    if int(server) in protection
                ]
                model.addConstr(
                    gp.quicksum(z[(tenant, rank, server)] for server in protection_options)
                    == y[(tenant, rank)],
                    name=f"select_then_assign_protection_{tenant}_{rank}",
                )
                if tenant not in participants:
                    fixed = int(self.failover_mapping[tenant][rank])
                    model.addConstr(x[(tenant, rank, fixed)] == 1, name=f"fixed_{tenant}_{rank}")
                    model.addConstr(y[(tenant, rank)] == 0, name=f"nonparticipant_not_selected_{tenant}_{rank}")
                elif original_server in self.candidate_sets[tenant]:
                    model.addConstr(
                        x[(tenant, rank, original_server)] == 1 - y[(tenant, rank)],
                        name=f"stay_or_select_protection_{tenant}_{rank}",
                    )

                mv = model.addVar(vtype=GRB.BINARY, name=f"Move_{tenant}_{rank}")
                move[(tenant, rank)] = mv
                if (tenant, rank) == (int(self.scenario.failure.tenant), int(failed_rank)):
                    model.addConstr(y[(tenant, rank)] == 1, name=f"failed_rank_selected_{tenant}_{rank}")
                    model.addConstr(mv == 0, name=f"ignore_failed_move_{tenant}_{rank}")
                else:
                    model.addConstr(mv == y[(tenant, rank)], name=f"move_def_{tenant}_{rank}")
                    switch_terms.append(mv)

            for server in self.candidate_sets[tenant]:
                model.addConstr(
                    gp.quicksum(x[(tenant, rank, int(server))] for rank in ranks) <= 1,
                    name=f"server_at_most_one_{tenant}_{int(server)}",
                )

        failed_tenant = int(self.scenario.failure.tenant)
        failed_rank = int(failed_rank)
        model.addConstr(y[(failed_tenant, failed_rank)] == 1, name="failed_rank_exactly_one_protection")
        all_candidate_servers = sorted(
            {int(server) for tenant in tenants for server in self.candidate_sets[int(tenant)]}
        )
        for server in sorted(protection):
            protection_occupants = [
                z[(int(tenant), int(rank), int(server))]
                for tenant in tenants
                for rank in sorted(self.scenario.pre_failure_mapping[int(tenant)])
                if (int(tenant), int(rank), int(server)) in z
            ]
            if len(protection_occupants) > 1:
                model.addConstr(
                    gp.quicksum(protection_occupants) <= 1,
                    name=f"protection_slot_at_most_one_{server}",
                )
        for server in all_candidate_servers:
            occupants = [
                x[(int(tenant), int(rank), server)]
                for tenant in tenants
                for rank in sorted(self.scenario.pre_failure_mapping[int(tenant)])
                if server in set(int(candidate) for candidate in self.candidate_sets[int(tenant)])
            ]
            if len(occupants) > 1:
                model.addConstr(
                    gp.quicksum(occupants) <= 1,
                    name=f"global_server_at_most_one_{server}",
                )

        # Linearize endpoint products for each logical flow and possible server pair.
        flow_records = []
        for tenant in tenants:
            for epoch, flows in enumerate(stage_flows.get(int(tenant), [])):
                for flow_idx, (src_rank, dst_rank, volume_bits) in enumerate(flows):
                    src_rank = int(src_rank)
                    dst_rank = int(dst_rank)
                    volume_bits = float(volume_bits)
                    for src_server in self.candidate_sets[int(tenant)]:
                        for dst_server in self.candidate_sets[int(tenant)]:
                            src_server = int(src_server)
                            dst_server = int(dst_server)
                            if src_server == dst_server:
                                continue
                            key = (int(tenant), int(epoch), int(flow_idx), src_server, dst_server)
                            u[key] = model.addVar(vtype=GRB.BINARY, name=f"U_{'_'.join(map(str, key))}")
                            model.addConstr(u[key] <= x[(int(tenant), src_rank, src_server)])
                            model.addConstr(u[key] <= x[(int(tenant), dst_rank, dst_server)])
                            model.addConstr(
                                u[key] >= x[(int(tenant), src_rank, src_server)] + x[(int(tenant), dst_rank, dst_server)] - 1
                            )
                            flow_records.append(
                                {
                                    "tenant": int(tenant),
                                    "epoch": int(epoch),
                                    "flow_idx": int(flow_idx),
                                    "src_server": src_server,
                                    "dst_server": dst_server,
                                    "volume_bits": volume_bits,
                                    "var": u[key],
                                    "path_edges": tuple(
                                        zip(
                                            self.evaluator.path_table[(int(tenant), src_server, dst_server)][:-1],
                                            self.evaluator.path_table[(int(tenant), src_server, dst_server)][1:],
                                        )
                                    ),
                                }
                            )

        delta = {
            epoch: model.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name=f"Delta_{epoch}")
            for epoch in range(max_epoch)
        }
        for epoch in range(max_epoch):
            epoch_records = [record for record in flow_records if int(record["epoch"]) == epoch]
            for edge, capacity in capacities.items():
                expr = gp.quicksum(
                    (float(record["volume_bits"]) / capacity) * record["var"]
                    for record in epoch_records
                    if edge in record["path_edges"]
                )
                model.addConstr(expr <= delta[epoch], name=f"edge_load_{epoch}_{edge[0]}_{edge[1]}")
            active_servers = sorted(
                {
                    int(record["src_server"])
                    for record in epoch_records
                }
                | {
                    int(record["dst_server"])
                    for record in epoch_records
                }
            )
            for server in active_servers:
                send_expr = gp.quicksum(
                    (float(record["volume_bits"]) / server_send_capacity[server]) * record["var"]
                    for record in epoch_records
                    if int(record["src_server"]) == server
                )
                recv_expr = gp.quicksum(
                    (float(record["volume_bits"]) / server_recv_capacity[server]) * record["var"]
                    for record in epoch_records
                    if int(record["dst_server"]) == server
                )
                model.addConstr(send_expr <= delta[epoch], name=f"sender_load_{epoch}_{server}")
                model.addConstr(recv_expr <= delta[epoch], name=f"receiver_load_{epoch}_{server}")

        tenant_finish = {}
        for tenant in tenants:
            stage_count = len(stage_flows.get(int(tenant), []))
            tenant_finish[int(tenant)] = model.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name=f"TenantFinish_{tenant}")
            model.addConstr(
                tenant_finish[int(tenant)] == gp.quicksum(delta[epoch] for epoch in range(stage_count)),
                name=f"tenant_finish_def_{tenant}",
            )
        makespan = model.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name="Makespan")
        for tenant in tenants:
            model.addConstr(makespan >= tenant_finish[int(tenant)], name=f"makespan_ge_{tenant}")

        avg_jct = gp.quicksum(tenant_finish[int(tenant)] for tenant in tenants) / max(len(tenants), 1)
        switch_count = gp.quicksum(switch_terms)
        model.ModelSense = GRB.MINIMIZE
        model.setObjectiveN(avg_jct, index=0, priority=3, name="avg_jct")
        model.setObjectiveN(makespan, index=1, priority=2, name="makespan")
        model.setObjectiveN(switch_count, index=2, priority=1, name="switch_count")
        model.optimize()

        mapping = copy_mapping(self.failover_mapping)
        if model.SolCount > 0:
            for tenant in tenants:
                tenant = int(tenant)
                for rank in sorted(self.scenario.pre_failure_mapping[tenant]):
                    for server in self.candidate_sets[tenant]:
                        if x[(tenant, int(rank), int(server))].X > 0.5:
                            mapping[tenant][int(rank)] = int(server)
                            break

        check_repair_feasible(self.scenario, mapping, self.failover_mapping)
        objective = self.evaluator.estimate(mapping)
        return RepairResult(
            name=self.name,
            mapping=mapping,
            objective=objective,
            switch_counts=count_switches(
                pre_failure_mapping=self.scenario.pre_failure_mapping,
                failover_mapping=self.failover_mapping,
                repaired_mapping=mapping,
                failure=self.scenario.failure,
            ),
            runtime_seconds=time.time() - start,
            metadata={
                "gurobi_status": int(model.Status),
                "gurobi_sol_count": int(model.SolCount),
                "decomposition_model": "two_layer_select_ranks_then_assign_protection_slots",
                "lexicographic_objective_order": [
                    "avg_jct",
                    "makespan",
                    "extra_servers_switched_to_protection_set",
                ],
                "selection_variables": len(y),
                "assignment_variables": len(z),
                "collapsed_avg_jct": float(avg_jct.getValue()) if model.SolCount > 0 else None,
                "collapsed_makespan": float(makespan.X) if model.SolCount > 0 else None,
                "collapsed_switch_count": float(switch_count.getValue()) if model.SolCount > 0 else None,
            },
        )

    def _stage_flows(self):
        if self.evaluator.tenant_collective_programs is not None:
            return build_collective_program_stage_flows(
                self.failover_mapping,
                self.evaluator.tenant_collective_programs,
                scale=1.0,
            )
        specs = self.evaluator._estimator_specs
        if specs is None:
            raise ValueError("CollapsedRepairMILPSolver requires collective specs or programs")
        return build_collective_stage_flows(
            self.failover_mapping,
            tenant_collective_specs=specs,
            scale=1.0,
        )
