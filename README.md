# MultiTenant CCL

This repository contains the code, experiments, and paper draft for
communication-program-aware rank mapping in multi-tenant collective
communication.  The current implementation uses a time-expanded contention
estimator to guide rank mapping under fixed server allocation and tenant-aware
ECMP paths.

## Repository Layout

- `multitenant/`: Python package for topology construction, workload parsing,
  DAG generation, simulation adapters, baselines, MILP models, and rank mapping
  solvers.
- `multitenant/solvers/contention_estimator.py`: time-expanded contention
  estimator used by the current mapping path.
- `multitenant/solvers/mapping_time_expanded_optimizer.py`: optimizer driven by
  the time-expanded contention estimator.
- `multitenant/solvers/mapping_hybrid.py`: collapsed hybrid mapping baseline and
  compatibility classes.
- `multitenant/solvers/mapping_ilp.py`: exact MILP model for small instances.
- `experiment/`: experiment entry points and saved JSON results.
- `figures/`: plotting scripts and generated figures/tables.
- `paper/`: LaTeX source and figures for the paper draft.
- `debug/`: audit/debug scripts. Generated JSON audit outputs are ignored.

## Environment Setup

Use Python 3.11 or newer.  The current development environment uses Python 3.12.

```bash
cd /Users/xiongjiaheng/COCA/MultiTenant
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

The exact MILP experiments require a working Gurobi installation and license
because `gurobipy` is used by `multitenant/solvers/mapping_ilp.py`.

## Compile Native Accelerators

The simulator and the time-expanded estimator both have optional C++/pybind11
accelerators.  Build them before running large experiments.

```bash
cd /Users/xiongjiaheng/COCA/MultiTenant
python setup.py build_ext --inplace
python debug/audits/build_te_accel.py
```

The first command builds the simulator extension.  The second command builds
`multitenant/solvers/_te_accel*.so`, which accelerates the time-expanded
contention estimator and price-guided candidate generation.  The `.so` files are
local build artifacts and are ignored by git.

## Run Experiments

Run the full experiment suite from the repository root:

```bash
cd /Users/xiongjiaheng/COCA/MultiTenant
python experiment/run_experiments.py
```

By default, this executes:

- `experiment/Low_contension.py`
- `experiment/High_contension.py`
- `experiment/Low_contension_homo.py`
- `experiment/High_contension_homo.py`
- `experiment/dominant vs full.py`
- `experiment/mapping_vs_ilp_single.py`
- `experiment/mapping_vs_ilp_multi.py`

The low/high contention scripts use the current time-expanded-estimator mapping
path for the main mapping result.  The dominant-vs-full and MILP comparison
scripts are validation experiments: they keep the corresponding hybrid/MILP
paths needed for those comparisons.

To run only selected scripts:

```bash
python experiment/run_experiments.py --scripts Low_contension.py High_contension.py
```

Experiment outputs are written as JSON files under `experiment/`, for example
`Low_contension.json`, `High_contension.json`, `mapping_vs_ilp_single.json`, and
`dominant vs full.json`.

## Generate Figures and Tables

### Add QP Baselines to Saved Main Experiments

`scripts/augment_qp_results.py` reads every saved trial in the four low/high
contention JSONs and evaluates only `qp` and `qp_plus_harmonics`. It preserves
the saved server allocation, default mapping, workload, and all old results.
The QP baseline uses four QPs per connection by default, message round robin,
deterministic QP-aware ECMP, and separate source queues across tenants.
QP+Harmonics uses the existing offset-search algorithm with QP-aware inputs.
The simulator and existing solvers are not modified.

```bash
python scripts/augment_qp_results.py --input-dir experiment/qp_full_20261011/originals \
  --output-dir experiment/qp_full_output --workers 44 --qp-count 4 --hash-seed 0
```

The output directory contains resumable per-trial checkpoints, a progress
manifest, and augmented copies of the four JSONs. Existing source files are
never overwritten by this runner. Do not use an already augmented JSON as its
input; use the original snapshots instead. The completed CloudLab run's
original snapshots and audit artifacts are retained in
`experiment/qp_full_20261011/`.

### Fresh randomized main experiments

Run all eight methods on 50 paired cases per scenario and tenant count:

```bash
python scripts/run_main_randomized.py --output-dir experiment/main50_output \
  --workers 48 --trials 50 --seed 20261011
python scripts/finalize_main_randomized.py --run-dir experiment/main50_output
```

This protocol generates 1,400 cases across the four main scenarios and tenant
counts 2--8. Independent, recorded seeds control balanced server allocation,
each tenant's rank permutation, and each heterogeneous trial's workload
assignment. Heterogeneous assignments sample the three existing profiles with
equal probability, conditioned on at least two distinct profiles; homogeneous
trials remain GPT-only. Topology, bandwidth, occupancy strength, collective
size rules, algorithm settings, and QP count remain fixed. Each of the eight
methods receives the same case input. Input fingerprints are checked for
duplicates before execution. This is a new protocol, not an extension of the
legacy ten cases, which held workload assignments fixed and repeated the
two-tenant high-contention input.

The runner saves the complete case plan before execution and checkpoints each
finished method. Its output contains the means and standard errors from every
trial. The finalizer verifies the inputs and means, exports paired comparisons,
generates figures in the run directory, and archives the completed run. On
CloudLab, the current run directory is
`/users/jxiong/main50_20261011/results/main50`; logs are in
`/users/jxiong/main50_20261011/main50.log`. Existing ten-trial results are retained.

After experiment JSON files are available, regenerate the main result figures:

```bash
cd /Users/xiongjiaheng/COCA/MultiTenant
python figures/plot_main_results.py
python figures/plot_dominant_proxy_results.py
```

Generated figures are written under `figures/`, including normalized Avg. JCT,
makespan summaries, MILP-vs-heuristic validation tables, and dominant-vs-full
validation tables.

## Compile the Paper

The paper source is in `paper/main.tex`.

```bash
cd /Users/xiongjiaheng/COCA/MultiTenant/paper
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

The compiled PDF is `paper/main.pdf`.

## Notes

- The main experimental objective is lexicographic: minimize average tenant
  completion time first, then makespan.
- The mapping experiments assume fixed server allocation and tenant-aware ECMP
  paths.
- Local logs, generated audit JSON files, LaTeX aux/log files, `.DS_Store`, and
  native `.so` build artifacts are intentionally ignored.
