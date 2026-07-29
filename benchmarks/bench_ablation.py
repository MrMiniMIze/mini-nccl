"""Tuning study: how many channels, and does slice pipelining pay?

    python benchmarks/bench_ablation.py --world-size 4

The ring all-reduce has two independent bandwidth optimizations, and this
measures each rather than assuming either helps:

- **channels**: split the tensor across N connections, one thread each
- **slice pipelining**: consume each ring step's receive in pieces, reducing
  slice i while slice i+1 is still arriving

Study A sweeps the channel count across message sizes, which is what sets
``collectives.CHANNEL_MIN_BYTES``. Study B toggles the pipeline at the best
channel count.

The pipeline is switched off with ``MINI_NCCL_MAX_SLICES=1``, which each
subprocess reads when it imports ``mini_nccl.collectives``.
"""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
from pathlib import Path

RUNNER = Path(__file__).parent / "_ablation_worker.py"
SIZES_MB = [1, 4, 16, 64]


def run_config(world: int, channels: int, slices: int, numels: list[int], iters: int) -> dict:
    """Run one configuration in a fresh interpreter.

    A subprocess is required because ``MINI_NCCL_MAX_SLICES`` is read when
    ``mini_nccl.collectives`` is imported.
    """
    env = dict(os.environ, MINI_NCCL_MAX_SLICES=str(slices))
    cmd = [
        sys.executable,
        str(RUNNER),
        "--world-size", str(world),
        "--channels", str(channels),
        "--iters", str(iters),
        "--numels", ",".join(str(n) for n in numels),
    ]
    out = subprocess.run(cmd, env=env, capture_output=True, text=True, check=True).stdout
    times = {}
    for line in out.splitlines():
        if line.startswith("RESULT"):
            _, numel, seconds = line.split()
            times[int(numel)] = float(seconds)
    return times


def table(title: str, numels: list[int], rows: list[tuple[str, dict]], baseline: dict) -> None:
    print(f"\n{title}\n")
    print("| configuration | " + " | ".join(f"{n * 4 // 2**20} MiB" for n in numels) + " |")
    print("|" + "---|" * (len(numels) + 1))
    for label, times in rows:
        cells = []
        for numel in numels:
            seconds = times[numel]
            cells.append(f"{seconds * 1e3:.0f} ms ({baseline[numel] / seconds:.2f}x)")
        print(f"| {label} | " + " | ".join(cells) + " |")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--world-size", type=int, default=4)
    ap.add_argument("--channel-counts", default="1,2,4,8")
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--output", default="benchmarks/results/ablation.csv")
    args = ap.parse_args()

    numels = [mb * 2**20 // 4 for mb in SIZES_MB]
    channel_counts = [int(c) for c in args.channel_counts.split(",")]

    records: list[tuple[str, int, int, dict]] = []

    for channels in channel_counts:
        label = f"{channels} channel{'s' if channels > 1 else ''}"
        print(f"[A] {label}, pipeline off ...", flush=True)
        records.append((label, channels, 1, run_config(
            args.world_size, channels, 1, numels, args.iters)))

    best_channels = channel_counts[-1]
    pipeline_rows = []
    for slices in (1, 16):
        label = f"{best_channels} channels, pipeline {'on' if slices > 1 else 'off'}"
        print(f"[B] {label} ...", flush=True)
        times = next(
            (t for lbl, ch, sl, t in records if ch == best_channels and sl == slices), None
        )
        if times is None:
            times = run_config(args.world_size, best_channels, slices, numels, args.iters)
            records.append((label, best_channels, slices, times))
        pipeline_rows.append((label, times))

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["world", "config", "channels", "slices", "bytes", "seconds", "gbps"])
        for label, channels, slices, times in records:
            for numel, seconds in sorted(times.items()):
                nbytes = numel * 4
                writer.writerow(
                    [args.world_size, label, channels, slices, nbytes,
                     f"{seconds:.6e}", f"{nbytes / seconds / 1e9:.4f}"]
                )

    single = records[0][3]
    table(
        f"A. channel scaling (ring all-reduce, world_size={args.world_size}, "
        f"pipeline off; speedup vs 1 channel)",
        numels,
        [(lbl, t) for lbl, ch, sl, t in records if sl == 1],
        single,
    )
    table(
        f"B. slice pipelining at {best_channels} channels (speedup vs pipeline off)",
        numels,
        pipeline_rows,
        pipeline_rows[0][1],
    )
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
