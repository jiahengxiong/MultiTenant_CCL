"""Replay saved trials with a Meta-inspired message-round-robin QP baseline.

Only simulator inputs change. K=1 is the selected legacy mapping control, not a
per-connection single-QP model. K>1 uses independent source-local QP queues
and QP-aware deterministic hashing; collisions between paths are allowed.
Each existing collective task is treated as one message. No message splitting,
RDMA ACKs, NIC resource costs, or enhanced switch hardware are simulated.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from multitenant.collectives import normalize_collective_programs
from multitenant.simulator.adapter import (
    build_simulator_topology,
    collective_program_policy,
    _run_simulation_worker_result,
)
from multitenant.topology import LeafSpineDatacenter


MAPPING_KEYS = {"default": "initial_mapping", "locality": "locality_mapping", "cape": "proposed_mapping"}
RESULT_KEYS = {"default": "default", "locality": "locality", "cape": "mapping"}


def build_inputs(metadata, trial, k, hash_seed, mapping_method="default"):
    topo = metadata["topology"]
    dc = LeafSpineDatacenter(**topo)
    mapping = {
        int(t): {int(r): int(s) for r, s in ranks.items()}
        for t, ranks in trial[MAPPING_KEYS[mapping_method]].items()
    }
    specs = {int(t): dict(spec) for t, spec in trial["tenant_collective_specs"].items()}
    programs = normalize_collective_programs(mapping, tenant_collective_specs=specs)
    policy = collective_program_policy(mapping, dc.build_tenant_ecmp_path_table(mapping), programs)
    sim_topo = build_simulator_topology(dc.topology)
    original = [(e.chunk_id, e.src, e.dst, e.chunk_size_bytes, e.time,
                 tuple(e.dependency), e.dependency_scope, e.dependency_delay, e.rate) for e in policy]
    counters = Counter()
    queues = defaultdict(dict)
    paths = {}
    cross_leaf_paths = defaultdict(set)
    used = set()
    transformed = []
    for entry in policy:
        tenant = int(str(entry.chunk_id).split("-", 1)[0])
        connection = (tenant, int(entry.src), int(entry.dst))
        qp_index = counters[connection] % k
        counters[connection] += 1
        identity = (*connection, qp_index)
        used.add(identity)
        if k == 1:
            transformed.append(entry)
            continue
        source_queues = queues[int(entry.src)]
        if identity not in source_queues:
            source_queues[identity] = len(source_queues)
        if identity not in paths:
            src, dst = int(entry.src), int(entry.dst)
            src_leaf, dst_leaf = dc.get_server_leaf(src), dc.get_server_leaf(dst)
            if src_leaf == dst_leaf:
                paths[identity] = [src, src_leaf, dst]
            else:
                # Stable QP-aware hash, not least-loaded routing or ideal striping.
                key = f"{hash_seed}:{tenant}:{src}->{dst}:qp={qp_index}".encode()
                value = int.from_bytes(hashlib.md5(key).digest(), "big")
                spine = dc.spine_index[value % dc.num_spine]
                paths[identity] = [src, src_leaf, spine, dst_leaf, dst]
                cross_leaf_paths[connection].add(spine)
        transformed.append(replace(entry, qpid=source_queues[identity], path=paths[identity]))
    if k > 1:
        for src, source_queues in queues.items():
            sim_topo.nodes[src]["num_qps"] = max(1, len(source_queues))
        # Unique identities include tenant; no modulo aliasing at shared servers.
        for src, source_queues in queues.items():
            assert len(set(source_queues.values())) == len(source_queues)
            assert max(source_queues.values()) < sim_topo.nodes[src]["num_qps"]
    after = [(e.chunk_id, e.src, e.dst, e.chunk_size_bytes, e.time,
              tuple(e.dependency), e.dependency_scope, e.dependency_delay, e.rate) for e in transformed]
    assert original == after, "QP scaling must preserve messages and dependencies"
    assert all(dc.topology.has_edge(u, v) for e in transformed for u, v in zip(e.path, e.path[1:]))
    diagnostics = {
        "message_count": len(policy),
        "total_bytes": sum(e.chunk_size_bytes for e in policy),
        "connection_count": len(counters),
        "used_logical_qps": len(used),
        "messages_per_connection_min": min(counters.values(), default=0),
        "messages_per_connection_max": max(counters.values(), default=0),
        "max_source_queue_count": max((len(q) for q in queues.values()), default=0) if k > 1 else 8,
        "mean_used_spines_per_cross_leaf_connection": (
            sum(map(len, cross_leaf_paths.values())) / len(cross_leaf_paths) if cross_leaf_paths else None
        ),
        "input_sha256": hashlib.sha256(json.dumps([
            [str(e.chunk_id), e.qpid, e.path] for e in transformed
        ], sort_keys=True).encode()).hexdigest(),
    }
    return sim_topo, transformed, diagnostics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", type=Path, default=[ROOT / "experiment/Low_contension.json"])
    parser.add_argument("--tenant-counts", default="2,4,8")
    parser.add_argument("--trials", type=int, default=1, help="First N saved trials per tenant count")
    parser.add_argument("--qp-counts", default="1,2,4,8")
    parser.add_argument("--hash-seed", type=int, default=0)
    parser.add_argument("--mapping-method", choices=tuple(MAPPING_KEYS), default="default",
                        help="Use the saved mapping; no mapping optimizer is run")
    parser.add_argument("--output", type=Path, default=ROOT / "experiment/qp_scaling_baseline.json")
    args = parser.parse_args()
    counts = {int(x) for x in args.tenant_counts.split(",")}
    ks = sorted({1, *(int(x) for x in args.qp_counts.split(","))})
    if args.trials < 1 or min(ks) < 1:
        parser.error("trials and QP counts must be positive")
    output = args.output.resolve()
    if output in {p.resolve() for p in args.inputs}:
        parser.error("output must not overwrite a source experiment")
    try:
        import simcore_cpp
        backend = "simcore_cpp"
    except ImportError:
        backend = "python"
    payload = {
        "metadata": {
            "baseline": "Meta-inspired QP scaling with message round robin and QP-aware ECMP",
            "reference": "https://engineering.fb.com/2024/08/05/data-center-engineering/roce-network-distributed-ai-training-at-scale/",
            "k1_control": "unchanged selected legacy mapping policy and queue configuration",
            "mapping_method": args.mapping_method,
            "message_granularity": "one existing collective task",
            "k_gt_1_queues": "independent source-local logical QPs, packet round robin; switch FIFO unchanged",
            "limitations": "not a full Meta implementation; no ACK or NIC resource costs; K=1 legacy queues differ from per-connection queues",
            "hash_seed": args.hash_seed, "qp_counts": ks,
            "python_executable": sys.executable, "python_version": sys.version,
            "simulator_backend": backend,
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        },
        "cases": [],
    }
    def save():
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(output.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, indent=2) + "\n")
        temporary.replace(output)

    for path in args.inputs:
        source_bytes = path.read_bytes()
        source = json.loads(source_bytes)
        for group in source["results"]:
            if int(group["tenant_count"]) not in counts:
                continue
            trials = sorted(group["trial_results"], key=lambda x: int(x["trial_index"]))[:args.trials]
            for trial in trials:
                case = {
                    "source": str(path.resolve()), "source_sha256": hashlib.sha256(source_bytes).hexdigest(),
                    "scene": source["metadata"]["scene"], "tenant_count": int(group["tenant_count"]),
                    "trial_index": trial["trial_index"], "trial_seed": trial["trial_seed"],
                    "saved_default": trial["results"]["default"],
                    "saved_locality": trial["results"]["locality"],
                    "saved_cape": trial["results"]["mapping"], "qp_results": {},
                    "mapping_method": args.mapping_method,
                    "saved_mapping_base": trial["results"][RESULT_KEYS[args.mapping_method]],
                }
                payload["cases"].append(case)
                for k in ks:
                    start = time.time()
                    topo, policy, diagnostics = build_inputs(source["metadata"], trial, k, args.hash_seed, args.mapping_method)
                    result = _run_simulation_worker_result(topo, policy)
                    metrics = {"avg_jct": float(result["avg_tenant_makespan"]),
                               "makespan": float(result["global_makespan"])}
                    saved_default, saved_cape = case["saved_default"], case["saved_cape"]
                    metrics.update({
                        "runtime_seconds": time.time() - start, "diagnostics": diagnostics,
                        "avg_jct_improvement_vs_default_pct": 100 * (1 - metrics["avg_jct"] / saved_default["avg_jct"]),
                        "avg_jct_gap_vs_cape_pct": 100 * (metrics["avg_jct"] / saved_cape["avg_jct"] - 1),
                        "avg_jct_improvement_vs_mapping_base_pct": 100 * (1 - metrics["avg_jct"] / case["saved_mapping_base"]["avg_jct"]),
                    })
                    if k == 1:
                        control = case["saved_mapping_base"]
                        errors = {name: abs(metrics[name] - control[name]) / max(abs(control[name]), 1e-15)
                                  for name in ("avg_jct", "makespan")}
                        metrics["replay_relative_errors"] = errors
                        metrics["replay_matches"] = max(errors.values()) <= 1e-6
                    case["qp_results"][str(k)] = metrics
                    save()
                    print(f"{path.stem} mapping={args.mapping_method} tenants={case['tenant_count']} trial={case['trial_index']} K={k} "
                          f"JCT={metrics['avg_jct']:.9g} default_improvement={metrics['avg_jct_improvement_vs_default_pct']:+.2f}% "
                          f"CAPE_gap={metrics['avg_jct_gap_vs_cape_pct']:+.2f}% runtime={metrics['runtime_seconds']:.1f}s", flush=True)
                    if k == 1 and not metrics["replay_matches"]:
                        raise RuntimeError(f"Legacy {args.mapping_method} replay mismatch: {errors}; saved comparison is not validated")
    if not payload["cases"]:
        raise ValueError("No saved trials matched the requested tenant counts")
    summary = []
    for name in sorted({Path(c["source"]).stem for c in payload["cases"]}):
        cases = [c for c in payload["cases"] if Path(c["source"]).stem == name]
        for k in ks:
            avg = lambda values: sum(values) / len(values)
            row = {"source_name": name, "qp_count": k, "case_count": len(cases),
                   "mapping_method": args.mapping_method,
                   "mean_paired_avg_jct_improvement_vs_mapping_base_pct": avg([
                       c["qp_results"][str(k)]["avg_jct_improvement_vs_mapping_base_pct"] for c in cases]),
                   "mean_paired_avg_jct_improvement_vs_default_pct": avg([
                       c["qp_results"][str(k)]["avg_jct_improvement_vs_default_pct"] for c in cases]),
                   "mean_paired_avg_jct_gap_vs_cape_pct": avg([
                       c["qp_results"][str(k)]["avg_jct_gap_vs_cape_pct"] for c in cases]),
                   "wins_vs_default": sum(c["qp_results"][str(k)]["avg_jct"] < c["saved_default"]["avg_jct"] * (1-1e-6) for c in cases),
                   "wins_vs_cape": sum(c["qp_results"][str(k)]["avg_jct"] < c["saved_cape"]["avg_jct"] * (1-1e-6) for c in cases)}
            summary.append(row)
    payload["summary"] = summary
    save()
    print(f"Saved {len(payload['cases'])} cases to {output}", flush=True)


if __name__ == "__main__":
    main()
