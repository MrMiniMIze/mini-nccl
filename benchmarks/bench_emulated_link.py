"""Test the predictions in docs/emulated-link.md over an emulated network link.

    python benchmarks/bench_emulated_link.py                  # every experiment
    python benchmarks/bench_emulated_link.py --only naive,bf16
    python benchmarks/bench_emulated_link.py --unshaped-inputs

Loopback TCP is a memory copy performed by the cores that also do the
reduction, so on loopback bytes are cheap and CPU is contended. Every
experiment here runs its ranks under a :class:`mini_nccl.LinkModel` instead,
which makes bytes cost time and leaves the CPU idle while they move, the two
properties a real NIC has and loopback lacks.

The six experiments are the six predictions, in the same order and under the
same names:

1. ``naive``: naive vs ring at 16 MiB as the world grows
2. ``crossover``: where tree stops beating ring
3. ``bf16``: what halving the wire bytes buys
4. ``ddp``: what overlapping gradient reduction with backward buys
5. ``channels``: whether more connections per peer help, under a per-rank cap
   and under a per-flow cap
6. ``slices``: whether reducing slice *i* while slice *i+1* arrives helps

``--unshaped-inputs`` measures the compute-side inputs the DDP prediction
needs (forward, backward and optimizer time on 4 ranks) with no link shaping,
so the predictions could be written before any shaped run.

Per-configuration time is the slowest rank's mean over the timed iterations,
as in bench_allreduce.py. Results go to benchmarks/results/emulated_link.csv.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import os
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "examples"))

from model import GPT

import mini_nccl as mn
from mini_nccl import collectives as c

MiB = 1024 * 1024
WORLD = 4
LARGE = 16 * MiB
RATES = (100, 1000)  # Mb/s
EXPERIMENTS = ("naive", "crossover", "bf16", "ddp", "channels", "slices")
FIELDS = ["experiment", "link", "world", "config", "bytes", "seconds", "iters"]

# The DDP experiment's model: the same GPT and bucket size bench_ddp_overlap.py
# uses, so the loopback "+2%" in the README is the baseline it is compared with.
GPT_CONFIG = {"n_layer": 6, "n_head": 8, "n_embd": 256, "block_size": 128, "batch_size": 8}
BUCKET_CAP_MB = 0.5


@contextlib.contextmanager
def _env(**values: str):
    """Set environment variables for the workers spawned inside the block.

    ``MINI_NCCL_CHANNEL_MIN_BYTES`` and ``MINI_NCCL_MAX_SLICES`` are read when
    a worker imports the library, and spawned workers inherit the parent's
    environment, so this is how one sweep can vary them.
    """
    saved = {key: os.environ.get(key) for key in values}
    os.environ.update(values)
    try:
        yield
    finally:
        for key, old in saved.items():
            if old is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old


def _agreed_iters(pg, seconds_per_iter: float, budget: float, lo: int, hi: int) -> int:
    """Rank 0 picks the iteration count and every rank uses it.

    Each rank deciding for itself would give different counts and desynchronize
    the collective sequence, which is the one thing ranks must never do.
    """
    iters = torch.tensor([float(max(lo, min(hi, round(budget / max(seconds_per_iter, 1e-6)))))])
    c.broadcast(pg, iters, src=0)
    return int(iters.item())


def _time_allreduce(pg, numel, algorithm, wire, budget) -> tuple[float, int]:
    tensor = torch.randn(numel)
    wire_dtype = torch.bfloat16 if wire == "bf16" else None

    def once() -> None:
        c.all_reduce(pg, tensor, algorithm=algorithm, wire_dtype=wire_dtype)

    once()  # warmup
    c.barrier(pg)
    start = time.perf_counter()
    once()
    iters = _agreed_iters(pg, time.perf_counter() - start, budget, lo=2, hi=50)
    c.barrier(pg)
    start = time.perf_counter()
    for _ in range(iters):
        once()
    return (time.perf_counter() - start) / iters, iters


def _allreduce_worker(pg, cases, budget) -> list[tuple[float, int]]:
    return [_time_allreduce(pg, numel, algo, wire, budget) for numel, algo, wire in cases]


def _run_allreduce(world, link, cases, budget=1.0, n_channels=mn.DEFAULT_N_CHANNELS):
    per_rank = mn.run(
        _allreduce_worker, world, cases, budget,
        link=link, n_channels=n_channels, timeout=3600.0,
    )
    out = []
    for i, (numel, algo, wire) in enumerate(cases):
        seconds = max(rank[i][0] for rank in per_rank)
        out.append((algo if wire is None else f"{algo}+{wire}", numel * 4, seconds, per_rank[0][i][1]))
    return out


def _link(mbps: float | None = None, flow_mbps: float | None = None) -> mn.LinkModel:
    return mn.LinkModel(mbps=mbps, flow_mbps=flow_mbps)


def _label(mbps: float | None) -> str:
    return "loopback" if mbps is None else f"{mbps:g} Mb/s"


# ---- the six experiments ----------------------------------------------------------


def exp_naive(emit) -> None:
    plan = [(1000, (2, 4, 8)), (100, (4,)), (None, (2, 4, 8))]
    for mbps, worlds in plan:
        for world in worlds:
            cases = [(LARGE // 4, "ring", None), (LARGE // 4, "naive", None)]
            for config, nbytes, seconds, iters in _run_allreduce(world, _link(mbps), cases):
                emit("naive", _label(mbps), world, config, nbytes, seconds, iters)


def exp_crossover(emit) -> None:
    sizes = [1024 * 4**i for i in range(6)]  # 1 KiB .. 1 MiB
    for mbps in (*RATES, None):
        cases = [(n // 4, algo, None) for n in sizes for algo in ("ring", "tree")]
        for config, nbytes, seconds, iters in _run_allreduce(WORLD, _link(mbps), cases, budget=0.5):
            emit("crossover", _label(mbps), WORLD, config, nbytes, seconds, iters)


def exp_bf16(emit) -> None:
    for mbps in (*RATES, None):
        cases = [(LARGE // 4, "ring", None), (LARGE // 4, "ring", "bf16")]
        for config, nbytes, seconds, iters in _run_allreduce(WORLD, _link(mbps), cases):
            emit("bf16", _label(mbps), WORLD, config, nbytes, seconds, iters)


def exp_channels(emit) -> None:
    # Lower the split threshold so a 16 MiB ring can use all four channels;
    # with the default 8 MiB it would never split past two.
    with _env(MINI_NCCL_CHANNEL_MIN_BYTES=str(1 * MiB)):
        for cap, link in (("per-rank", _link(mbps=1000)), ("per-flow", _link(flow_mbps=1000))):
            for channels in (1, 2, 4):
                cases = [(LARGE // 4, "ring", None)]
                (_, nbytes, seconds, iters), = _run_allreduce(
                    WORLD, link, cases, n_channels=channels
                )
                emit("channels", f"1000 Mb/s {cap}", WORLD, f"{channels}ch", nbytes, seconds, iters)


def exp_slices(emit) -> None:
    for slices in ("1", "16"):
        with _env(MINI_NCCL_MAX_SLICES=slices):
            for mbps in (*RATES, None):
                cases = [(LARGE // 4, "ring", None)]
                (_, nbytes, seconds, iters), = _run_allreduce(WORLD, _link(mbps), cases)
                emit("slices", _label(mbps), WORLD, f"slices={slices}", nbytes, seconds, iters)


def _ddp_worker(pg, overlap: bool, steps: int) -> dict[str, float]:
    """Mean seconds per step, and its forward / backward / sync / optimizer parts."""
    torch.manual_seed(0)
    cfg = GPT_CONFIG
    model = GPT(
        vocab_size=65, block_size=cfg["block_size"], n_layer=cfg["n_layer"],
        n_head=cfg["n_head"], n_embd=cfg["n_embd"],
    )
    ddp = mn.DistributedDataParallel(model, pg, bucket_cap_mb=BUCKET_CAP_MB, overlap=overlap)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    g = torch.Generator().manual_seed(pg.rank)
    x = torch.randint(0, 65, (cfg["batch_size"], cfg["block_size"]), generator=g)
    y = torch.randint(0, 65, (cfg["batch_size"], cfg["block_size"]), generator=g)

    parts = dict.fromkeys(("forward", "backward", "sync", "optimizer"), 0.0)

    def step(record: bool) -> None:
        marks = [time.perf_counter()]
        ddp.zero_grad()
        _, loss = ddp(x, y)
        marks.append(time.perf_counter())
        loss.backward()
        marks.append(time.perf_counter())
        ddp.sync()
        marks.append(time.perf_counter())
        opt.step()
        marks.append(time.perf_counter())
        if record:
            for key, a, b in zip(parts, marks, marks[1:], strict=False):
                parts[key] += b - a

    for _ in range(2):
        step(record=False)
    c.barrier(pg)
    start = time.perf_counter()
    for _ in range(steps):
        step(record=True)
    total = (time.perf_counter() - start) / steps
    return {"step": total, **{k: v / steps for k, v in parts.items()}}


def _run_ddp(link, overlap: bool, steps: int) -> dict[str, float]:
    per_rank = mn.run(_ddp_worker, WORLD, overlap, steps, link=link, timeout=3600.0)
    # The slowest rank sets the step time; report every part from that rank.
    return max(per_rank, key=lambda r: r["step"])


def exp_ddp(emit, extra_rates: tuple[float, ...] = ()) -> None:
    for mbps in (*RATES, *extra_rates, None):
        steps = 4 if mbps == 100 else 10
        for overlap in (False, True):
            result = _run_ddp(_link(mbps), overlap, steps)
            config = "overlap" if overlap else "no-overlap"
            for part, seconds in result.items():
                emit("ddp", _label(mbps), WORLD, f"{config}:{part}", 0, seconds, steps)


def unshaped_inputs() -> None:
    """Compute times the DDP prediction depends on, measured with no shaping."""
    result = _run_ddp(_link(), overlap=False, steps=10)
    print(f"DDP on {WORLD} ranks, unshaped, overlap off (slowest rank, mean of 10 steps):")
    for part, seconds in result.items():
        print(f"  {part:10s} {seconds * 1e3:8.1f} ms")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--only", help=f"comma-separated subset of {','.join(EXPERIMENTS)}")
    ap.add_argument(
        "--ddp-extra-rate", type=float, action="append", default=[],
        help="an additional link rate (Mb/s) for the DDP experiment",
    )
    ap.add_argument("--unshaped-inputs", action="store_true")
    ap.add_argument("--output", default="benchmarks/results/emulated_link.csv")
    args = ap.parse_args()

    if args.unshaped_inputs:
        unshaped_inputs()
        return

    chosen = args.only.split(",") if args.only else list(EXPERIMENTS)
    unknown = set(chosen) - set(EXPERIMENTS)
    if unknown:
        ap.error(f"unknown experiments {sorted(unknown)}")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[list] = []
    if out_path.exists() and args.only:
        # A partial rerun replaces only the experiments it ran.
        with out_path.open() as f:
            rows = [
                [r[k] for k in FIELDS] for r in csv.DictReader(f) if r["experiment"] not in chosen
            ]

    def emit(experiment, link, world, config, nbytes, seconds, iters) -> None:
        rows.append([experiment, link, world, config, nbytes, f"{seconds:.6e}", iters])
        print(f"  {experiment:9s} {link:22s} W={world} {config:22s} "
              f"{nbytes / MiB:8.3f} MiB {seconds * 1e3:10.2f} ms  (x{iters})", flush=True)

    runners = {
        "naive": exp_naive,
        "crossover": exp_crossover,
        "bf16": exp_bf16,
        "ddp": lambda emit: exp_ddp(emit, tuple(args.ddp_extra_rate)),
        "channels": exp_channels,
        "slices": exp_slices,
    }
    started = time.perf_counter()
    for name in EXPERIMENTS:
        if name in chosen:
            print(f"== {name}", flush=True)
            t0 = time.perf_counter()
            runners[name](emit)
            print(f"   ({time.perf_counter() - t0:.0f}s)", flush=True)

    with out_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(FIELDS)
        writer.writerows(rows)
    print(f"wrote {out_path} in {time.perf_counter() - started:.0f}s")


if __name__ == "__main__":
    main()
