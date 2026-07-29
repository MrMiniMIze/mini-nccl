"""What a bfloat16 wire buys in time, and what it costs in accuracy.

    python benchmarks/bench_low_precision.py --world-size 4

Reduces float32 tensors two ways, natively and with a bfloat16 wire, and
reports both the speedup and the resulting error against an exact reference.
Halving the bytes should approach a 2x speedup once the payload is large
enough for bandwidth rather than per-message latency to dominate.

The accuracy column is the point of pairing them: the win is only worth
having if the error is acceptable, and the error depends on the algorithm
(each hop rounds the partial sum back to the wire dtype, and ring has O(W)
hops against tree's O(log W)).
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import torch

import mini_nccl as mn
from mini_nccl import collectives as c

SIZES_MB = [1, 4, 16, 64]


def _worker(pg, numels, iters, warmup) -> list[tuple]:
    W = pg.world_size
    rows = []
    for numel in numels:
        # 0.1 is not representable in bfloat16, so the error column measures
        # something real rather than a payload that survives by luck.
        contribution = (pg.rank + 1) * 0.1
        exact = sum((r + 1) * 0.1 for r in range(W))
        for algorithm in ("ring", "tree"):
            for wire in (None, torch.bfloat16):
                tensor = torch.full((numel,), contribution)
                for _ in range(warmup):
                    scratch = tensor.clone()
                    c.all_reduce(pg, scratch, algorithm=algorithm, wire_dtype=wire)
                c.barrier(pg)
                start = time.perf_counter()
                for _ in range(iters):
                    scratch = tensor.clone()
                    c.all_reduce(pg, scratch, algorithm=algorithm, wire_dtype=wire)
                seconds = (time.perf_counter() - start) / iters
                error = float((scratch - exact).abs().max())
                rows.append((numel, algorithm, "bf16" if wire else "fp32", seconds, error))
        if pg.rank == 0:
            print(f"  done {numel * 4 // 2**20} MiB", flush=True)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--world-size", type=int, default=4)
    ap.add_argument("--iters", type=int, default=8)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--output", default="benchmarks/results/low_precision.csv")
    args = ap.parse_args()

    numels = [mb * 2**20 // 4 for mb in SIZES_MB]
    per_rank = mn.run(_worker, args.world_size, numels, args.iters, args.warmup, timeout=3600.0)

    # Slowest rank defines the time, as everywhere else in these benchmarks.
    merged = []
    for i, (numel, algorithm, wire, _, error) in enumerate(per_rank[0]):
        seconds = max(rank_rows[i][3] for rank_rows in per_rank)
        merged.append((numel, algorithm, wire, seconds, error))

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["world", "bytes", "algorithm", "wire", "seconds", "max_error"])
        for numel, algorithm, wire, seconds, error in merged:
            writer.writerow(
                [args.world_size, numel * 4, algorithm, wire, f"{seconds:.6e}", f"{error:.6f}"]
            )

    lookup = {(n, a, w): (s, e) for n, a, w, s, e in merged}
    print(f"\nfloat32 payload, world_size={args.world_size}\n")
    print("| size | algorithm | fp32 wire | bf16 wire | speedup | bf16 max error |")
    print("|---|---|---|---|---|---|")
    for numel in numels:
        for algorithm in ("ring", "tree"):
            base, _ = lookup[(numel, algorithm, "fp32")]
            narrow, error = lookup[(numel, algorithm, "bf16")]
            print(
                f"| {numel * 4 // 2**20} MiB | {algorithm} | {base * 1e3:.1f} ms | "
                f"{narrow * 1e3:.1f} ms | {base / narrow:.2f}x | {error:.4f} |"
            )
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
