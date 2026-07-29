"""Benchmark all-reduce implementations across message sizes and world sizes.

    python benchmarks/bench_allreduce.py --world-sizes 2,4 --gloo

Measures mini-nccl's ring, tree, and naive algorithms — and optionally
torch.distributed's gloo backend on identical processes — and writes a CSV
of algorithm bandwidth (bytes moved / time) and NCCL-convention bus
bandwidth (algbw * 2(W-1)/W for all-reduce).

Per-config time is the max across ranks of (total loop time / iters): the
slowest rank defines collective latency, exactly as in nccl-tests.
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import torch

import mini_nccl as mn
from mini_nccl import collectives as c

ALGORITHMS = ("ring", "tree", "naive")


def _bench_worker(pg, numels, iters, warmup, gloo_port) -> list[tuple[str, int, float]]:
    gloo = None
    if gloo_port is not None:
        try:
            import torch.distributed as dist

            dist.init_process_group(
                "gloo",
                init_method=f"tcp://127.0.0.1:{gloo_port}",
                rank=pg.rank,
                world_size=pg.world_size,
            )
            gloo = dist
        except Exception as exc:  # gloo is a nice-to-have comparison, not required
            if pg.rank == 0:
                print(f"  (gloo unavailable: {exc})")

    rows = []
    for numel in numels:
        tensor = torch.randn(numel)
        contenders: list[tuple[str, object]] = [
            (algo, lambda a=algo: c.all_reduce(pg, tensor, algorithm=a))
            for algo in ALGORITHMS
        ]
        if gloo is not None:
            contenders.append(("gloo", lambda: gloo.all_reduce(tensor)))

        for name, fn in contenders:
            for _ in range(warmup):
                fn()
            c.barrier(pg)
            start = time.perf_counter()
            for _ in range(iters):
                fn()
            elapsed = (time.perf_counter() - start) / iters
            rows.append((name, numel, elapsed))
        if pg.rank == 0:
            print(f"  done numel={numel:>10,} ({numel * 4 / 1e6:.1f} MB)")

    if gloo is not None:
        gloo.destroy_process_group()
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--world-sizes", default="2,4")
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--gloo", action="store_true", help="also benchmark torch.distributed gloo")
    ap.add_argument("--output", default="benchmarks/results/allreduce.csv")
    args = ap.parse_args()

    # 4 KiB .. 64 MiB in factor-of-4 steps (float32 elements)
    numels = [1024 * 4**i for i in range(8)]
    world_sizes = [int(w) for w in args.world_sizes.split(",")]

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["world", "algorithm", "numel", "bytes", "seconds", "algbw_gbps", "busbw_gbps"]
        )
        for world in world_sizes:
            print(f"world_size={world}")
            gloo_port = None
            if args.gloo:
                from mini_nccl.launcher import _free_ports

                gloo_port = _free_ports(1)[0]
            per_rank = mn.run(
                _bench_worker, world, numels, args.iters, args.warmup, gloo_port,
                timeout=3600.0,
            )
            # The slowest rank defines the time for each configuration.
            for i in range(len(per_rank[0])):
                name, numel, _ = per_rank[0][i]
                seconds = max(rank_rows[i][2] for rank_rows in per_rank)
                nbytes = numel * 4
                algbw = nbytes / seconds / 1e9
                busbw = algbw * 2 * (world - 1) / world
                writer.writerow(
                    [world, name, numel, nbytes, f"{seconds:.6e}", f"{algbw:.4f}", f"{busbw:.4f}"]
                )

    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
