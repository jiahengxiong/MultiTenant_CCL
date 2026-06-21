# Failure-Aware Collective Repair Plan

## Goal

Extend the original multi-tenant collective rank-mapping project to handle one
server failure after the working-set mapping has been computed.

The extension must keep the original mapping project as the reference pipeline:

- working-set mapping comes from `experiment/Low_contension.py::run_mapping`,
- workload specs come from the original Low Contention dominant trace pipeline,
- collective DAG/resource data comes from the original time-expanded estimator
  and `multitenant.solvers.DAG_generation.build_collective_dag_data`,
- tenant-aware ECMP paths and bottleneck modeling stay unchanged.

Original source files under `multitenant/`, `experiment/`, and
`CCL_Simulator/` are not edited.  Repair-specific extensions live under
`failure_aware_repair_project/`.

## Post-Failure Problem

Inputs:

- pre-failure mapping `x0[tenant][rank] -> server`,
- one `FailureEvent(tenant, failed_rank, failed_server)`,
- one shared `global_protection_pool`,
- original `RepairWorkload` carrying `tenant_collective_specs` or collective
  programs,
- fixed datacenter topology and tenant-aware path table.

There is no per-tenant protection-node ownership in the current design.  The
global protection pool is shared by all repair strategies and is disjoint from
the working servers.

Every feasible post-failure mapping must:

- keep the failed server unused,
- map every logical rank to exactly one active server,
- keep each tenant injective over its ranks,
- use only healthy working servers plus the global protection pool,
- obey each strategy's participant/fixed-tenant constraints.

Failure selection has two experiment modes:

- `random`: uniform random working-server failure, useful for average-case
  paper-facing multi-trial averages.
- `critical_failover`: enumerate every working server as a single failure and
  select the one whose traditional failover has the highest estimated Avg JCT.
  This is kept as a supplemental stress test, not the main result.

## Strategies

The comparison has exactly three strategy specs:

1. `repair_failed_server_only`
   - fixed traditional failover baseline,
   - move only the failed rank to one global protection server,
   - keep every healthy rank fixed.

2. `tenant_local_repair`
   - only the failed tenant may switch ranks,
   - all other tenants remain fixed to the failover mapping,
   - each switched rank can only move to a server in the global protection
     pool.

3. `cooperative_repair`
   - all tenants may switch ranks,
   - each switched rank can only move to a server in the global protection
     pool,
   - objective is still shared-system performance, not independent per-tenant
     repair.

This is not a full remapping problem.  The fixed failover mapping is the repair
baseline; a rank may stay on its failover server or switch to the shared
protection pool.  A rank may not move to a different healthy working server, so
the post-failure solver does not reconstruct the whole original mapping.

## Objective

The repair objective is lexicographic:

1. minimize post-repair Avg JCT,
2. minimize post-repair makespan,
3. minimize additional healthy-rank switches relative to the fixed failover
   baseline.

The failed rank's unavoidable movement is not counted as an extra repair switch.
Reports still include both total switches versus the pre-failure mapping and
extra switches versus failover.

## Solver Architecture

`FailureAwareMappingProblem` is the fixed post-failure input.  It creates one
`RepairScenario` per strategy and exposes constraints as JSON payloads.

`RepairEvaluator` is the bridge to the original project.  It constructs
`TimeExpandedContentionEstimator` with the original workload specs and then
exposes `compiled_dag_data = estimator._backbone.data`.  That data is produced
by the original `build_collective_dag_data` path and contains the compiled task
DAG, task surrogate data, tenant schedules, physical paths, and link/resource
capacity tables.

`FailureAwareMappingStrategySolver` is the mapping-style facade:

- `solve()`,
- `get_X_mapping()`,
- `get_objective()`,
- `to_strategy_result()`.

The scalable repair path is the structured heuristic in
`failure_aware_repair/heuristic.py`.  It uses the time-expanded estimator's
price/pressure signals to generate switch-only protection candidates, then
scores full mappings with the original estimator objective.

The exact repair path is `FailureAwareRepairTimeExpandedILPSolver` in
`failure_aware_repair/milp.py`.  It extends the copied base
`failure_aware_repair/mapping_ilp_base.py` with:

- candidate server sets that include global protection nodes,
- failed-server exclusion,
- fixed non-participant constraints,
- movement variables,
- third objective for switch count.

## Random Experiment Pipeline

The canonical script is:

```bash
python3.12 failure_aware_repair_project/scripts/run_low_contention_random_repair.py
```

For original Low Contention style large experiments, report 10 random-failure
trials:

```bash
python3.12 failure_aware_repair_project/scripts/run_low_contention_random_repair_trials.py \
  --seed 10 \
  --trials 10 \
  --num-spine 4 \
  --num-leaf 8 \
  --per-leaf-server 8 \
  --num-tenants 7 \
  --working-allocation-mode balanced_remaining \
  --protection-pool-size-mode max \
  --working-mapping-time-limit 3 \
  --repair-time-limit 20 \
  --failover-policy first \
  --failure-selection random \
  --beam-width 6 \
  --max-rounds 4 \
  --max-candidates-per-tenant 80 \
  --max-joint-tenants 3 \
  --joint-candidates-per-tenant 6 \
  --max-joint-candidates 240 \
  --max-block-ranks 4 \
  --block-extra-servers 6 \
  --max-block-candidates 512
```

`balanced_remaining` reserves the global protection pool first and balances all
remaining servers across tenants.  With 4 spines, 8 leaves, and 8 servers per
leaf, the topology has 64 servers; the protection pool size is capped at
`floor(0.10 * 64) = 6`.

Before large runs, compile the original native accelerators from the repository
root:

```bash
python3.12 setup.py build_ext --inplace
python3.12 debug/audits/build_te_accel.py
python3.12 failure_aware_repair_project/scripts/check_native_dependencies.py
```

## Regression Evidence To Keep

The tests should continue to verify:

- working-set mapping uses `experiment/Low_contension.py::run_mapping`,
- default workload mode uses Low Contention dominant trace-derived specs,
- balanced allocation covers all non-protection servers,
- repair scenarios carry the original workload into `RepairEvaluator`,
- `repair_dag.source` is
  `multitenant.solvers.DAG_generation.build_collective_dag_data`,
- all three strategy constraints are emitted,
- failover, tenant-local repair, and cooperative repair are feasible,
- the exact time-expanded repair MILP consumes the compiled DAG input on small
  cases.
