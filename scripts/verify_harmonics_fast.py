from __future__ import annotations

import sys
from pathlib import Path
from statistics import mean, stdev

repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from multitenant.config import BITS_PER_MB

from scripts.run_default_vs_harmonics import run_once


def main():
    num_tenants = 8
    servers_per_tenant = 4
    op_size_bits = int(8.0 * BITS_PER_MB)
    seeds = [20240, 20241, 20242, 20243, 20244]

    D_ms = []
    D_avg = []
    H_ms = []
    H_avg = []
    H_wall = []

    for seed in seeds:
        r = run_once(
            num_tenants=num_tenants,
            seed=seed,
            servers_per_tenant=servers_per_tenant,
            op_size_bits=op_size_bits,
        )
        d_ms, d_avg, _ = r["default"]
        h_ms, h_avg, h_wall = r["default+harmonics"]
        D_ms.append(d_ms)
        D_avg.append(d_avg)
        H_ms.append(h_ms)
        H_avg.append(h_avg)
        H_wall.append(h_wall)
        print(
            f"seed={seed} default(ms={d_ms:.6f}, avg={d_avg:.6f}) "
            f"+harmonics(ms={h_ms:.6f}, avg={h_avg:.6f}, wall={h_wall:.3f}s)"
        )

    print()
    print(f"summary: op_size=8MiB ops=2 tenants={num_tenants} nseeds={len(seeds)}")
    print(f"default           ms={mean(D_ms):.6f}±{stdev(D_ms):.6f}  avg={mean(D_avg):.6f}±{stdev(D_avg):.6f}")
    print(f"default+harmonics  ms={mean(H_ms):.6f}±{stdev(H_ms):.6f}  avg={mean(H_avg):.6f}±{stdev(H_avg):.6f}")
    print(f"delta avg mean: {mean([h - d for h, d in zip(H_avg, D_avg)]):+.6f}")
    print(f"harmonics wall mean: {mean(H_wall):.3f}s")


if __name__ == "__main__":
    main()
