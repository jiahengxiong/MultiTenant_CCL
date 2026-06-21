# Failure-Aware Collective Repair

This directory implements a failure-aware extension of the original
multi-tenant collective rank-mapping project.

It intentionally does not edit the original `multitenant/`, `experiment/`, or
`CCL_Simulator/` files.  Instead, this package imports those modules as the
reference implementation for working-set mapping, collective workload
construction, tenant-aware paths, time-expanded DAG compilation, and objective
evaluation.  The repair-specific models, constraints, solvers, tests, and debug
scripts live here.

## What Is Implemented

- Failure, working-set, and one shared global protection-pool data model.
- A fixed traditional failover strategy that only replaces the failed rank.
- Feasibility and switch-count checks against the post-failure constraints.
- Estimator-backed repair objective evaluation using the original
  time-expanded collective DAG compiler.
- Tenant-local repair, where only the failed tenant may switch ranks between
  its failover placement and the shared protection pool.
- Cooperative repair, where all tenants may switch ranks through the same
  global protection pool.
- Small-case exact enumerator for estimator-grounded validation.
- Copied base of the original time-expanded `MappingILPSolver` in
  `failure_aware_repair/mapping_ilp_base.py`, with failure-aware extensions kept
  in this standalone directory.
- Collapsed-stage repair MILP with Avg JCT, makespan, and switch-count
  objectives.
- Proxy MILP scaffold for assignment-level repair formulation.
- Pytest tests and a debug runner.

The exact enumerator is the current ground-truth checker for small instances
under the existing time-expanded estimator.  `FailureAwareRepairTimeExpandedILPSolver`
is the main INFOCOM-style exact formulation path: it extends a copied base of
the original time-expanded MILP inside this new project directory, so the
original solver file remains untouched.  The collapsed-stage MILP remains a
lighter independent surrogate for method development and sanity checks.

## Canonical Input Pipeline

The paper-style experiment path is deliberately aligned with the original
mapping project:

1. Build a `LeafSpineDatacenter` with the requested spine, leaf, and server
   counts.
2. Sample low-contention working sets.  In `balanced_remaining` mode, one
   global protection pool is reserved first, then every remaining server is
   balanced across tenants, matching the spirit of `experiment/Low_contension.py`.
3. Build random initial rank-to-server mappings on those working sets.
4. Load the original `experiment/Low_contension.py` module and construct its
   dominant trace-derived `tenant_collective_specs`.
5. Call the original `experiment/Low_contension.py::run_mapping` function to
   compute each tenant's working-set mapping through
   `MappingEstimatorBlackBoxOptimizer`.  This pre-failure step is not MILP.
6. Create a `FailureAwareMappingProblem` from the solved working mapping, one
   random server failure, the global protection pool, and the original workload
   specs.
7. For each repair strategy, `RepairEvaluator` constructs the original
   `TimeExpandedContentionEstimator`.  Its estimator backbone calls
   `multitenant.solvers.DAG_generation.build_collective_dag_data`, producing the
   compiled task DAG and resource data consumed by the repair search/MILP.
8. Evaluate the three post-failure strategies under the same compiled
   collective DAG, tenant-aware paths, link/sender/receiver bottlenecks, and
   lexicographic Avg JCT / makespan / switch-count objective.

The default failover baseline uses `--failover-policy first`, which represents
a static traffic-oblivious backup choice from the global protection pool.  The
stronger `--failover-policy same_leaf_or_nearest` option is available for
topology-aware baseline stress tests.

The paper-facing failure selection is uniform random and should be reported as
multi-trial averages.  `--failure-selection critical_failover` is kept only as a
supplemental stress test: it enumerates every working server as a possible
single failure and selects the one whose traditional failover has the highest
estimated Avg JCT.

The repair search is switch-only.  Starting from the fixed failover mapping, a
rank may stay where it is or move to a server in the shared global protection
pool.  A rank may not move to another healthy working server, so the repair does
not recompute the original rank mapping after a failure.  Cooperative repair
uses joint candidate composition: after generating switch candidates for
individual tenants, it combines candidates from multiple tenants and evaluates
the full mapping with the time-expanded estimator.  This prevents cooperative
repair from degenerating into tenant-local repair repeated one tenant at a time.

The JSON payload reports this explicitly through:

- `working_set_best_mappings[*].solver_source`
- `workload.source`
- `failure_aware_mapping_problem`
- `repair_dag.source`
- `strategy_constraints`
- `results[*].metadata`

## Quick Run

From the repository root:

```bash
python3.12 -m pytest failure_aware_repair_project/tests
python3.12 failure_aware_repair_project/scripts/run_debug_repair.py
python3.12 failure_aware_repair_project/scripts/audit_infocom_repair_case.py
python3.12 failure_aware_repair_project/scripts/run_time_expanded_ilp_debug.py
python3.12 failure_aware_repair_project/scripts/run_low_contention_random_repair.py
```

For the current paper-style 10-trial random-failure average:

```bash
python3.12 failure_aware_repair_project/scripts/run_low_contention_random_repair_trials.py \
  --seed 10 \
  --trials 10 \
  --num-spine 4 \
  --num-leaf 8 \
  --per-leaf-server 8 \
  --num-tenants 7 \
  --protection-pool-size-mode high_resource \
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

## Random Low-Contention Pipeline

`scripts/run_low_contention_random_repair.py` is the current paper-style input
pipeline:

1. Randomly sample every tenant's working set under low-contention placement.
   The recommended `balanced_remaining` mode reserves the global protection
   pool and uses all other servers as tenant working servers.
2. Reserve one shared global protection pool from topology-defined resource
   modes before tenant working sets are assigned.  `high_resource` picks one
   protection server per leaf; `low_resource` picks one protection server per
   two leaves.
3. Build one multi-tenant initial mapping from all sampled working sets.
4. Call the original `experiment/Low_contension.py::run_mapping` entry point to
   compute the working-set-only proposed mapping.  This uses the original
   `MappingEstimatorBlackBoxOptimizer` path and does not use MILP for
   pre-failure mapping.
5. Randomly choose a tenant and one of its working servers to fail.
6. Evaluate repair-failed-server-only, tenant-local repair, and cooperative
   repair on that failure.

The repair MILP is separate from the pre-failure mapping solver.  It lives in
`failure_aware_repair/milp.py` and extends the copied time-expanded
`MappingILPSolver` with failure, global-protection-pool, participant, and
switch-count constraints.  The structured heuristic in
`failure_aware_repair/heuristic.py` follows the same repair problem structure:
switch-only protection candidates, participant tenants, joint multi-tenant
candidate composition, estimator bottleneck pressure, and lexicographic Avg JCT
/ makespan / switch-count ordering.

## Native Accelerator Notes

The original repository documents two optional C++/pybind11 build steps for
large experiments:

```bash
python3.12 setup.py build_ext --inplace
python3.12 debug/audits/build_te_accel.py
```

This standalone repair project does not run those build commands automatically
because they create `.so` artifacts in the original project tree.  The tests and
debug scripts here run through the pure-Python estimator/simulator path.  Before
large-scale INFOCOM experiments, compile the native accelerators from the
repository root and confirm their status with:

```bash
python3.12 failure_aware_repair_project/scripts/check_native_dependencies.py
```

## INFOCOM Audit Case

`scripts/audit_infocom_repair_case.py` contains a fixed shared-contention
failure scenario where repair improves post-failure Avg JCT relative to
traditional failover.  It reports:

- pre-failure mapping,
- global protection pool,
- failure event,
- failover objective and simulator result,
- tenant-local repair objective and switch count,
- cooperative repair objective and switch count.

This is a regression audit for the paper-quality claim that structure repair can
recover contention lost by traditional failover.
