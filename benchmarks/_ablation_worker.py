"""One ablation configuration, in its own interpreter.

Invoked by bench_ablation.py so that ``MINI_NCCL_MAX_SLICES`` takes effect at
import time. Prints ``RESULT <numel> <seconds>`` lines.
"""

from __future__ import annotations

import argparse
import time

import torch

import mini_nccl as mn
from mini_nccl import collectives as c


def _worker(pg, numels, iters) -> list[tuple[int, float]]:
    rows = []
    for numel in numels:
        tensor = torch.randn(numel)
        for _ in range(2):  # warmup
            c.all_reduce(pg, tensor, algorithm="ring")
        c.barrier(pg)
        start = time.perf_counter()
        for _ in range(iters):
            c.all_reduce(pg, tensor, algorithm="ring")
        rows.append((numel, (time.perf_counter() - start) / iters))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--world-size", type=int, required=True)
    ap.add_argument("--channels", type=int, required=True)
    ap.add_argument("--iters", type=int, required=True)
    ap.add_argument("--numels", required=True)
    args = ap.parse_args()
    numels = [int(n) for n in args.numels.split(",")]

    per_rank = mn.run(
        _worker, args.world_size, numels, args.iters,
        timeout=3600.0, n_channels=args.channels,
    )
    for i, (numel, _) in enumerate(per_rank[0]):
        slowest = max(rank_rows[i][1] for rank_rows in per_rank)
        print(f"RESULT {numel} {slowest:.9f}")


if __name__ == "__main__":
    main()
