"""Does overlapping device copies with the network actually pay?

    python benchmarks/bench_device.py --world-size 2

Compares three ways to all-reduce a tensor that lives on an accelerator:

- **naive**: ``tensor.cpu()``, reduce, copy back. Pageable host memory, so the
  driver stages through its own pinned buffer and the copies are synchronous.
- **staged**: the same shape of work, but through a pinned host buffer, which
  lets the copy engine DMA directly.
- **pipelined**: a ring run on the device tensor, with the payload chunked so
  chunk *k* is on the wire while chunk *k+1* is being copied off the device.

The interesting comparison is staged against pipelined, because it tests a
claim made elsewhere in this repo. The same pipelining idea measured 5-40%
*slower* on loopback TCP with CPU tensors, since there the copy and the
"network" are both memcpy on the same cores. With a real copy engine the two
should overlap, and this is the measurement that says whether they do.

Runs on CPU tensors too, where all three are expected to be roughly equal:
without distinct hardware there is nothing to overlap, which is the point.
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import torch

import mini_nccl as mn
from mini_nccl import collectives as c
from mini_nccl.device import all_reduce_pipelined, all_reduce_staged, device_report

SIZES_MB = [1, 4, 16, 64]


def _naive_all_reduce(pg, tensor: torch.Tensor) -> None:
    """The obvious implementation, for reference."""
    host = tensor.cpu()
    c.all_reduce(pg, host, algorithm="ring")
    tensor.copy_(host)


def _worker(pg, numels, iters, warmup, use_cuda, chunk_bytes) -> list[tuple]:
    device = torch.device("cuda" if use_cuda else "cpu")
    rows = []
    for numel in numels:
        base = torch.full((numel,), float(pg.rank + 1), device=device)
        variants = {
            "naive": lambda t: _naive_all_reduce(pg, t),
            "staged": lambda t: all_reduce_staged(pg, t, algorithm="ring"),
            "pipelined": lambda t: all_reduce_pipelined(pg, t, chunk_bytes=chunk_bytes),
        }
        for name, fn in variants.items():
            for _ in range(warmup):
                fn(base.clone())
            if device.type == "cuda":
                torch.cuda.synchronize()
            c.barrier(pg)
            start = time.perf_counter()
            for _ in range(iters):
                fn(base.clone())
            if device.type == "cuda":
                torch.cuda.synchronize()
            seconds = (time.perf_counter() - start) / iters

            # Correctness alongside timing: a fast wrong answer is no use.
            check = base.clone()
            fn(check)
            expected = float(pg.world_size * (pg.world_size + 1) // 2)
            error = float((check.cpu() - expected).abs().max())
            rows.append((numel, name, seconds, error))
        if pg.rank == 0:
            print(f"  done {numel * 4 // 2**20} MiB", flush=True)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--world-size", type=int, default=2)
    ap.add_argument("--iters", type=int, default=8)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--chunk-bytes", type=int, default=4 * 1024 * 1024)
    ap.add_argument("--cpu", action="store_true", help="force CPU tensors")
    ap.add_argument("--output", default="benchmarks/results/device.csv")
    args = ap.parse_args()

    report = device_report()
    print("environment:")
    for key, value in report.items():
        print(f"  {key}: {value}")
    use_cuda = bool(report["cuda_available"]) and not args.cpu
    if not use_cuda:
        print(
            "\nrunning on CPU tensors. The pipelined path has no separate copy\n"
            "engine to overlap with here, so expect all three to be close; that\n"
            "is the null result, not a bug. See docs/cuda.md to enable a GPU.\n"
        )
    else:
        print(f"\nrunning on {report['device_name']}, {args.world_size} ranks share it\n")

    numels = [mb * 2**20 // 4 for mb in SIZES_MB]
    per_rank = mn.run(
        _worker, args.world_size, numels, args.iters, args.warmup, use_cuda,
        args.chunk_bytes, timeout=3600.0,
    )

    merged = []
    for i, (numel, name, _, error) in enumerate(per_rank[0]):
        seconds = max(rank_rows[i][2] for rank_rows in per_rank)
        merged.append((numel, name, seconds, error))

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["world", "device", "bytes", "variant", "seconds", "max_error"])
        for numel, name, seconds, error in merged:
            writer.writerow(
                [
                    args.world_size,
                    "cuda" if use_cuda else "cpu",
                    numel * 4,
                    name,
                    f"{seconds:.6e}",
                    f"{error:.6f}",
                ]
            )

    lookup = {(n, v): (s, e) for n, v, s, e in merged}
    label = "cuda" if use_cuda else "cpu"
    print(f"\nall-reduce of {label} tensors, world_size={args.world_size}\n")
    print("| size | naive | staged (pinned) | pipelined | pipelined vs staged |")
    print("|---|---|---|---|---|")
    for numel in numels:
        naive = lookup[(numel, "naive")][0]
        staged = lookup[(numel, "staged")][0]
        piped = lookup[(numel, "pipelined")][0]
        print(
            f"| {numel * 4 // 2**20} MiB | {naive * 1e3:.1f} ms | {staged * 1e3:.1f} ms | "
            f"{piped * 1e3:.1f} ms | **{staged / piped:.2f}x** |"
        )
    worst = max(error for _, _, _, error in merged)
    print(f"\nworst max error across all variants: {worst:.6f}")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
