"""Run all eight main methods on preregistered, paired randomized cases."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
from statistics import mean, stdev
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "experiment")]
import Low_contension as legacy
from multitenant.baselines import build_leaf_local_mapping
from multitenant.topology import LeafSpineDatacenter
from scripts.augment_qp_results import atomic_json, run_trial as run_qp

SCENARIOS = ("Low_contension", "Low_contension_homo", "High_contension", "High_contension_homo")
METHODS = (*legacy.METHOD_NAMES, "qp", "qp_plus_harmonics")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def seed_for(master, *parts):
    return int.from_bytes(hashlib.sha256(json.dumps([master, *parts]).encode()).digest()[:16], "big")


def random_mapping(datacenter, tenants, high, occupancy_seed, rank_seed):
    """Random quota recipients, feasible weighted tenant pairs, independent ranks."""
    rng = random.Random(occupancy_seed)
    servers = list(datacenter.get_all_servers())
    rng.shuffle(servers)
    slots = len(servers) * (2 if high else 1)
    quotas = {t: slots // tenants for t in range(tenants)}
    for t in rng.sample(list(quotas), slots % tenants):
        quotas[t] += 1
    owned = {t: [] for t in quotas}
    if high:
        remaining = dict(quotas)
        for i, server in enumerate(servers):
            after = len(servers) - i - 1
            pairs, weights = [], []
            for a in range(tenants):
                for b in range(a + 1, tenants):
                    if remaining[a] and remaining[b] and all(
                        q - int(t == a or t == b) <= after for t, q in remaining.items()
                    ):
                        pairs.append((a, b))
                        weights.append(remaining[a] * remaining[b])
            assert pairs, "Balanced two-tenant occupancy must remain feasible"
            for t in rng.choices(pairs, weights=weights, k=1)[0]:
                owned[t].append(server)
                remaining[t] -= 1
        assert not any(remaining.values())
    else:
        cursor = 0
        for t, quota in quotas.items():
            owned[t] = servers[cursor:cursor + quota]
            cursor += quota
    mapping = {}
    for t, physical_servers in owned.items():
        random.Random(seed_for(rank_seed, "tenant", t)).shuffle(physical_servers)
        mapping[t] = dict(enumerate(physical_servers))
    assert all(len(mapping[t]) == quotas[t] for t in quotas)
    counts = {s: 0 for s in servers}
    for ranks in mapping.values():
        assert len(set(ranks.values())) == len(ranks)
        for s in ranks.values():
            counts[s] += 1
    assert set(counts.values()) == {2 if high else 1}
    return mapping


def prepare(args):
    profiles = legacy.load_dominant_profiles(legacy.TASK_SIZE_MULTIPLIER)
    datacenter = LeafSpineDatacenter(**legacy.TOPOLOGY)
    cases = []
    uniqueness = {}
    for scenario in SCENARIOS:
        for count in args.tenant_counts:
            fingerprints = set()
            for index in range(args.trials):
                seeds = {kind: seed_for(args.seed, scenario, count, index, kind)
                         for kind in ("case", "occupancy", "ranks", "workloads")}
                mapping = random_mapping(datacenter, count, scenario.startswith("High"), seeds["occupancy"], seeds["ranks"])
                if scenario.endswith("homo"):
                    assignment = {t: dict(profiles["GPT13B"]) for t in range(count)}
                else:
                    rng = random.Random(seeds["workloads"])
                    names = list(profiles)
                    # IID equal-probability profile draws, conditioned on at least
                    # two distinct profiles so the heterogeneous label is true.
                    while True:
                        drawn = [rng.choice(names) for _ in range(count)]
                        if len(set(drawn)) >= 2:
                            break
                    assignment = {t: dict(profiles[name]) for t, name in enumerate(drawn)}
                specs = legacy.build_tenant_collective_specs(assignment, mapping)
                canonical_specs = {str(t): {k: v for k, v in s.items() if k != "trace_path"} for t, s in specs.items()}
                fingerprint = digest([legacy.normalize_mapping(mapping), canonical_specs])
                assert fingerprint not in fingerprints, f"Duplicate case: {scenario}, {count}, {index}"
                fingerprints.add(fingerprint)
                cases.append(dict(source_name=scenario, tenant_count=count, trial_index=index,
                                  trial_seed=seeds["case"], random_seeds=seeds, input_sha256=fingerprint,
                                  initial_mapping=legacy.normalize_mapping(mapping),
                                  tenant_collective_specs=legacy.normalize_specs(specs)))
            uniqueness[f"{scenario}:t{count}"] = len(fingerprints)
    source_paths = sorted(set([Path(__file__), *ROOT.glob("multitenant/**/*.py"),
                              *ROOT.glob("multitenant/**/*.cpp"), *ROOT.glob("CCL_Simulator/**/*.py"),
                              *ROOT.glob("CCL_Simulator/**/*.hpp"), *ROOT.glob("CCL_Simulator/**/*.cpp"),
                              ROOT / "experiment/Low_contension.py", ROOT / "scripts/augment_qp_results.py",
                              ROOT / "scripts/run_qp_scaling_baseline.py"]))
    metadata = dict(topology=legacy.TOPOLOGY, seed=args.seed, trials=args.trials,
                    tenant_counts=args.tenant_counts, harmonics_time_limit_seconds=None,
                    mapping_solver="MappingEstimatorBlackBoxOptimizer", mapping_solver_time_limit_seconds=None,
                    dominant_profiles=profiles, methods=list(METHODS), qp_count=4, qp_hash_seed=0,
                    protocol="randomized-v1", task_size_rule="same dominant collective and size rule as legacy main experiments",
                    randomness={"occupancy": "balanced quotas; random remainder recipients; random servers; weighted feasible pairs for high contention",
                                "ranks": "independent server permutation for every tenant",
                                "workloads": "GPT-only for homogeneous; IID equal-probability profile draws conditioned on >=2 profiles for heterogeneous",
                                "ecmp": "unchanged deterministic hash; routes vary with case input; QP hash seed=0",
                                "fixed": "topology, link bandwidth, collective traces, occupancy strength, QP count, method parameters"},
                    provenance=dict(server=platform.node(), python=sys.version, backend="simcore_cpp", workers=args.workers,
                                    code_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths},
                                    trace_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in legacy.WORKLOAD_TRACES.values()}))
    return dict(metadata=metadata, uniqueness=uniqueness, cases=cases)


def run_case(job):
    case, metadata, output = job
    checkpoint = Path(output) / "checkpoints" / f"{case['source_name']}_t{case['tenant_count']}_trial{case['trial_index']}.json"
    signature = digest([case, metadata])
    if checkpoint.exists():
        result = json.loads(checkpoint.read_text())
        assert result["run_signature"] == signature, "Checkpoint does not match this protocol/code"
    else:
        result = dict(case, results={}, run_signature=signature)
    start = time.time()
    name = f"{case['source_name']} t={case['tenant_count']} trial={case['trial_index']}"
    print(f"START {name} pid={os.getpid()}", flush=True)
    datacenter = LeafSpineDatacenter(**metadata["topology"])
    initial = {int(t): {int(r): int(s) for r, s in ranks.items()} for t, ranks in case["initial_mapping"].items()}
    specs = {int(t): dict(s) for t, s in case["tenant_collective_specs"].items()}
    if "locality_mapping" not in result:
        result["locality_mapping"] = legacy.normalize_mapping(build_leaf_local_mapping(initial))
        atomic_json(checkpoint, result)
    if "proposed_mapping" not in result:
        proposed, runtime = legacy.run_mapping(datacenter, initial, specs)
        result["proposed_mapping"] = legacy.normalize_mapping(proposed)
        result["mapping_runtime_seconds"] = runtime
        atomic_json(checkpoint, result)
        print(f"MAPPING_DONE {name} elapsed={runtime:.1f}s", flush=True)
    for method, key in [("default", "initial_mapping"), ("locality", "locality_mapping"), ("mapping", "proposed_mapping")]:
        mapping = {int(t): {int(r): int(s) for r, s in ranks.items()} for t, ranks in result[key].items()}
        if method not in result["results"]:
            mk, jct = legacy.evaluate_collective(datacenter, mapping, specs)
            result["results"][method] = dict(avg_jct=jct, makespan=mk)
            atomic_json(checkpoint, result)
        harm = method + "_plus_harmonics"
        if harm not in result["results"]:
            metrics, runtime = legacy.run_harmonics(datacenter, mapping, specs, time_limit_s=None)
            result["results"][harm] = dict(metrics, runtime_seconds=runtime)
            atomic_json(checkpoint, result)
        print(f"METHOD_DONE {name} method={harm}", flush=True)
    qp_checkpoint = Path(output) / "qp_checkpoints" / checkpoint.name
    qp = run_qp(dict(name=case["source_name"], tenant_count=case["tenant_count"], trial=result,
                     metadata=metadata, qp_count=4, hash_seed=0, source_sha256=case["input_sha256"],
                     signature=signature, checkpoint=str(qp_checkpoint)))
    for method in ("qp", "qp_plus_harmonics"):
        result["results"][method] = qp["results"][method]
    result["qp_diagnostics"] = qp["qp_diagnostics"]
    assert set(result["results"]) == set(METHODS)
    for metrics in result["results"].values():
        assert all(math.isfinite(metrics[m]) and metrics[m] > 0 for m in ("avg_jct", "makespan"))
    result["complete"] = True
    result["last_attempt_runtime_seconds"] = time.time() - start
    atomic_json(checkpoint, result)
    print(f"CASE_DONE {name} elapsed={time.time()-start:.1f}s", flush=True)
    return result


def export(plan, completed, output):
    for scenario in SCENARIOS:
        groups = []
        for count in plan["metadata"]["tenant_counts"]:
            trials = sorted([c for c in completed.values() if c["source_name"] == scenario and c["tenant_count"] == count],
                            key=lambda t: t["trial_index"])
            if not trials:
                continue
            averages = {m: {metric: mean(t["results"][m][metric] for t in trials)
                            for metric in ("avg_jct", "makespan")} for m in METHODS}
            stderr = {m: {metric: stdev(t["results"][m][metric] for t in trials) / math.sqrt(len(trials)) if len(trials)>1 else None
                          for metric in ("avg_jct", "makespan")} for m in METHODS}
            groups.append(dict(tenant_count=count, trial_count=len(trials), trial_results=trials, averages=averages, standard_errors=stderr))
        atomic_json(Path(output) / f"{scenario}.json", dict(metadata=dict(plan["metadata"], scene=scenario), results=groups,
                                                           complete=len([c for c in completed.values() if c["source_name"]==scenario]) == len(plan["metadata"]["tenant_counts"])*plan["metadata"]["trials"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261011)
    parser.add_argument("--trials", type=int, default=50)
    parser.add_argument("--tenant-counts", type=lambda s: [int(t) for t in s.split(",")], default=list(legacy.TENANT_COUNTS))
    parser.add_argument("--workers", type=int, default=40)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    assert args.trials >= 2 and args.workers >= 1 and all(2 <= t <= 8 for t in args.tenant_counts)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    plan = prepare(args)
    path = output / "case_plan.json"
    if path.exists():
        assert json.loads(path.read_text()) == plan, "Existing case plan differs: use a new output directory"
    else:
        atomic_json(path, plan)
    print(f"PREPARED {len(plan['cases'])} unique cases; {len(METHODS)} paired methods; {plan['uniqueness']}", flush=True)
    if args.prepare_only:
        return
    completed, failures = {}, []
    jobs = []
    for case in plan["cases"]:
        checkpoint = output / "checkpoints" / f"{case['source_name']}_t{case['tenant_count']}_trial{case['trial_index']}.json"
        if checkpoint.exists():
            saved = json.loads(checkpoint.read_text())
            assert saved["run_signature"] == digest([case, plan["metadata"]])
            if saved.get("complete"):
                completed[case["input_sha256"]] = saved
                continue
        jobs.append((case, plan["metadata"], str(output)))
    # Interleave scene and tenant count to avoid leaving the largest cases last.
    jobs.sort(key=lambda job: (job[0]["trial_index"], job[0]["tenant_count"], job[0]["source_name"]))
    start = time.time()
    def progress():
        atomic_json(output / "manifest.json", dict(total_cases=len(plan["cases"]), completed=len(completed),
                                                   failures=failures, elapsed_seconds=time.time()-start,
                                                   complete=len(completed)==len(plan["cases"]) and not failures))
    progress()
    with ProcessPoolExecutor(max_workers=args.workers, max_tasks_per_child=1) as pool:
        futures = {pool.submit(run_case, job): job[0] for job in jobs}
        for future in as_completed(futures):
            case = futures[future]
            try:
                completed[case["input_sha256"]] = future.result()
            except Exception as exc:
                failures.append(dict(scenario=case["source_name"], tenant_count=case["tenant_count"], trial_index=case["trial_index"], error=repr(exc)))
                print(f"ERROR {failures[-1]}", flush=True)
            progress()
            if len(completed) % 20 == 0:
                export(plan, completed, output)
            print(f"PROGRESS {len(completed)}/{len(plan['cases'])} failures={len(failures)}", flush=True)
    export(plan, completed, output)
    progress()
    if failures:
        raise SystemExit("Some cases failed; checkpoints preserved for resume")


if __name__ == "__main__":
    main()
