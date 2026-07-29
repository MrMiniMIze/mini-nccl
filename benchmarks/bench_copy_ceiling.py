"""How much can overlapping device copies with the network possibly save?

    python benchmarks/bench_copy_ceiling.py

Pipelining the staging copies against the network can only hide the copy, so
the best it can ever do is take the copy off the critical path. This measures
the two halves separately (PCIe bandwidth in both directions, pinned and
pageable, against the transport's measured bandwidth) and reports the resulting
upper bound.

The point is to tell an implementation problem from a structural one. If the
copy is already a small fraction of the total, no amount of tuning will make
overlapping it matter, and the honest move is to say so rather than keep
optimizing.
"""

from __future__ import annotations

import argparse
import time

import torch

SIZES_MB = [4, 16, 64]


def time_copy(fn, iters: int, warmup: int = 3) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) / iters


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument(
        "--transport-gbps",
        type=float,
        default=0.5,
        help="measured all-reduce bus bandwidth of the transport, GB/s "
        "(the loopback TCP ring in this repo measures about 0.4 to 0.7)",
    )
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("needs a CUDA device; see docs/cuda.md")

    print(f"device: {torch.cuda.get_device_name(0)}")
    print(f"torch:  {torch.__version__}\n")
    print("| size | D2H pinned | H2D pinned | D2H pageable | round trip pinned |")
    print("|---|---|---|---|---|")

    round_trips = {}
    for mb in SIZES_MB:
        numel = mb * 2**20 // 4
        device_tensor = torch.empty(numel, device="cuda")
        pinned = torch.empty(numel, pin_memory=True)
        pageable = torch.empty(numel)

        # Bind the buffers as defaults so each closure keeps this iteration's
        # tensors rather than whatever the loop variable ends up pointing at.
        d2h = time_copy(
            lambda p=pinned, d=device_tensor: p.copy_(d, non_blocking=True), args.iters
        )
        h2d = time_copy(
            lambda d=device_tensor, p=pinned: d.copy_(p, non_blocking=True), args.iters
        )
        d2h_page = time_copy(
            lambda g=pageable, d=device_tensor: g.copy_(d), args.iters
        )
        round_trips[mb] = d2h + h2d

        gb = mb / 1024
        print(
            f"| {mb} MiB | {gb / d2h:.1f} GB/s | {gb / h2d:.1f} GB/s | "
            f"{gb / d2h_page:.1f} GB/s | {(d2h + h2d) * 1e3:.2f} ms |"
        )

    print("\nupper bound on what pipelining the copies can save:\n")
    print("| size | copy round trip | est. network time | copy share | max speedup |")
    print("|---|---|---|---|---|")
    for mb in SIZES_MB:
        # A ring all-reduce moves 2(W-1)/W of the payload per rank; at W=2 that
        # is the payload once in each direction.
        network = (mb / 1024) / args.transport_gbps
        copy = round_trips[mb]
        share = copy / (copy + network)
        print(
            f"| {mb} MiB | {copy * 1e3:.1f} ms | {network * 1e3:.0f} ms | "
            f"{share * 100:.1f}% | **{1 / (1 - share):.2f}x** |"
        )
    print(
        "\nPipelining hides the copy behind the network, so the copy's share of\n"
        "the total is the whole prize. A single-digit share means a single-digit\n"
        "ceiling, and per-chunk overhead can easily exceed it."
    )


if __name__ == "__main__":
    main()
