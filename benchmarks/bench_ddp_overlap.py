"""Measure what communication/compute overlap buys in DDP step time.

    python benchmarks/bench_ddp_overlap.py --world-size 4

Trains the example GPT twice — once with the bucketed-overlap reducer,
once reducing all buckets serially after backward — and reports mean step
time for each. The gap is the communication cost hidden behind backward.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "examples"))

import mini_nccl as mn
from model import GPT


def _time_config(pg, overlap: bool, steps: int, cfg: dict) -> float:
    torch.manual_seed(0)
    model = GPT(
        vocab_size=65,
        block_size=cfg["block_size"],
        n_layer=cfg["n_layer"],
        n_head=cfg["n_head"],
        n_embd=cfg["n_embd"],
    )
    ddp = mn.DistributedDataParallel(model, pg, bucket_cap_mb=0.5, overlap=overlap)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)

    g = torch.Generator().manual_seed(pg.rank)
    x = torch.randint(0, 65, (cfg["batch_size"], cfg["block_size"]), generator=g)
    y = torch.randint(0, 65, (cfg["batch_size"], cfg["block_size"]), generator=g)

    def step() -> None:
        ddp.zero_grad()
        _, loss = ddp(x, y)
        loss.backward()
        ddp.sync()
        opt.step()

    for _ in range(3):  # warmup
        step()
    from mini_nccl import collectives

    collectives.barrier(pg)
    start = time.perf_counter()
    for _ in range(steps):
        step()
    return (time.perf_counter() - start) / steps


def _worker(pg, steps: int, cfg: dict) -> tuple[float, float]:
    with_overlap = _time_config(pg, True, steps, cfg)
    without = _time_config(pg, False, steps, cfg)
    return with_overlap, without


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--world-size", type=int, default=4)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--block-size", type=int, default=128)
    ap.add_argument("--n-layer", type=int, default=6)
    ap.add_argument("--n-head", type=int, default=8)
    ap.add_argument("--n-embd", type=int, default=256)
    args = ap.parse_args()
    cfg = vars(args).copy()
    world = cfg.pop("world_size")
    steps = cfg.pop("steps")

    results = mn.run(_worker, world, steps, cfg, timeout=3600.0)
    with_overlap = max(r[0] for r in results)
    without = max(r[1] for r in results)
    saved = (1 - with_overlap / without) * 100
    print(f"world_size={world}  step time with overlap:    {with_overlap * 1e3:8.1f} ms")
    print(f"world_size={world}  step time without overlap: {without * 1e3:8.1f} ms")
    print(f"overlap hides {saved:.1f}% of step time")


if __name__ == "__main__":
    main()
