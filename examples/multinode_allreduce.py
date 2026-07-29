"""All-reduce benchmark that runs across machines, one process per rank.

Unlike the local benchmarks, this does not spawn anything: each rank is its
own process, told who it is through the environment. That is what makes it
work over ssh, Slurm, torchrun-style launchers, or containers.

    # on every host, with the same host list:
    MINI_NCCL_HOSTS=10.0.0.1:29500,10.0.0.2:29500 \
    WORLD_SIZE=2 RANK=0 python examples/multinode_allreduce.py

See docs/multinode.md, or use scripts/launch_multinode.sh to do the fan-out.
"""

from __future__ import annotations

import argparse
import os
import time

import torch

import mini_nccl as mn
from mini_nccl import collectives as c
from mini_nccl.collectives import ALGORITHMS

SIZES_MB = [0.001, 0.016, 0.25, 1, 4, 16, 64]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--channels", type=int, default=None,
                    help="defaults to MINI_NCCL_CHANNELS or the built-in default")
    ap.add_argument("--algorithms", default=",".join(ALGORITHMS))
    args = ap.parse_args()

    pg = mn.init_process_group(n_channels=args.channels)
    rank, world = pg.rank, pg.world_size
    algorithms = [a.strip() for a in args.algorithms.split(",") if a.strip()]

    if rank == 0:
        hosts = os.environ.get("MINI_NCCL_HOSTS", "(loopback defaults)")
        print(f"world_size={world} channels={pg.n_channels} hosts={hosts}")
        print(f"{'size':>10} " + " ".join(f"{a:>12}" for a in algorithms))

    try:
        for mb in SIZES_MB:
            numel = max(1, int(mb * 2**20) // 4)
            tensor = torch.randn(numel)
            nbytes = numel * 4
            cells = []
            for algorithm in algorithms:
                for _ in range(args.warmup):
                    c.all_reduce(pg, tensor, algorithm=algorithm)
                c.barrier(pg)
                start = time.perf_counter()
                for _ in range(args.iters):
                    c.all_reduce(pg, tensor, algorithm=algorithm)
                seconds = (time.perf_counter() - start) / args.iters
                # NCCL-convention bus bandwidth for all-reduce.
                busbw = nbytes / seconds / 1e9 * 2 * (world - 1) / world
                cells.append(f"{seconds * 1e3:7.2f}ms {busbw:5.3f}")
            if rank == 0:
                label = (
                    f"{nbytes / 2**20:g} MiB" if nbytes >= 2**20 else f"{nbytes / 1024:g} KiB"
                )
                print(f"{label:>10} " + " ".join(f"{cell:>13}" for cell in cells), flush=True)
        c.barrier(pg)
        if rank == 0:
            print("\ncolumns are: time per all-reduce, then bus bandwidth in GB/s")
    finally:
        mn.destroy_process_group()


if __name__ == "__main__":
    main()
