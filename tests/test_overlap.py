"""Proving the claims the README makes about timing.

Two claims elsewhere in this project rest on wall-clock behaviour rather than
arithmetic, so they are asserted here from the flight recorder's own timestamps
instead of being taken on trust:

1. DDP really does start reducing a bucket while backward is still running. The
   "+2% from overlap" measurement says the payoff is small on this hardware; it
   does not say whether the mechanism works at all, and those are different
   questions. If a refactor accidentally serialized the reducer, the throughput
   change would be within noise while the design claim quietly became false.
2. A consistently slow rank is identifiable from a trace, which is what makes a
   straggler diagnosable rather than merely suspected.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time

import torch
import torch.nn.functional as F
from torch import nn

from mini_nccl import collectives as c
from mini_nccl.ddp import DistributedDataParallel
from mini_nccl.launcher import run


def _deep_model() -> nn.Module:
    """Deep enough that backward takes long enough to overlap with."""
    torch.manual_seed(5)
    layers: list[nn.Module] = []
    for _ in range(8):
        layers += [nn.Linear(256, 256), nn.Tanh()]
    layers.append(nn.Linear(256, 1))
    return nn.Sequential(*layers)


def _overlap_worker(pg, overlap: bool) -> dict:
    pg.recorder.enabled = True
    model = _deep_model()
    # Small buckets so there are several reductions to interleave.
    ddp = DistributedDataParallel(model, pg, bucket_cap_mb=0.05, overlap=overlap)
    x = torch.randn(64, 256)
    y = torch.randn(64, 1)

    # Warm up so allocation and thread startup are not part of the measurement.
    for _ in range(2):
        ddp.zero_grad()
        F.mse_loss(ddp(x), y).backward()
        ddp.sync()

    first_start = None
    ddp.zero_grad()
    loss = F.mse_loss(ddp(x), y)
    backward_begin = time.perf_counter_ns()
    loss.backward()
    backward_end = time.perf_counter_ns()
    ddp.sync()

    # The recorder stamps events with time.perf_counter_ns(), the same clock.
    bucket_events = [
        ev
        for ev in pg.recorder._events
        if ev.op.startswith("ddp_bucket") and ev.start_ns >= backward_begin
    ]
    if bucket_events:
        first_start = min(ev.start_ns for ev in bucket_events)
    return {
        "rank": pg.rank,
        "n_buckets": len(ddp._buckets),
        "n_events": len(bucket_events),
        # Negative means a reduction began before backward returned.
        "first_reduce_minus_backward_end_us": (
            None if first_start is None else (first_start - backward_end) / 1e3
        ),
        "backward_us": (backward_end - backward_begin) / 1e3,
    }


def test_overlap_starts_reducing_before_backward_finishes() -> None:
    reports = run(_overlap_worker, 2, True)
    for report in reports:
        assert report["n_buckets"] > 1, report
        assert report["n_events"] > 0, report
        offset = report["first_reduce_minus_backward_end_us"]
        assert offset is not None and offset < 0, (
            f"rank {report['rank']}: first bucket reduction began "
            f"{offset:.0f}us after backward returned, so nothing overlapped: {report}"
        )


def test_without_overlap_reduction_waits_for_backward() -> None:
    """The control: with overlap off, every reduction happens after backward."""
    reports = run(_overlap_worker, 2, False)
    for report in reports:
        offset = report["first_reduce_minus_backward_end_us"]
        assert offset is not None and offset > 0, (
            f"rank {report['rank']}: reduction began before backward returned "
            f"with overlap disabled: {report}"
        )


def _straggler_worker(pg, slow_rank: int) -> None:
    """One rank is deliberately slow to enter each collective."""
    for _ in range(12):
        if pg.rank == slow_rank:
            time.sleep(0.02)  # stand in for a slower host or an unbalanced shard
        c.all_reduce(pg, torch.ones(4096))
    c.barrier(pg)


def test_diagnose_names_a_straggler(tmp_path) -> None:
    trace_dir = tmp_path / "trace"
    run(_straggler_worker, 4, 2, trace_dir=trace_dir, timeout=180.0)

    proc = subprocess.run(
        [sys.executable, "-m", "mini_nccl.diagnose", str(trace_dir)],
        capture_output=True,
        text=True,
    )
    # The job succeeded, so there is nothing to fail on: only a note.
    assert proc.returncode == 0, proc.stdout
    assert "all ranks agree" in proc.stdout, proc.stdout
    assert "STRAGGLER: rank 2" in proc.stdout, proc.stdout

    # The trace should back that up: the healthy ranks wait inside the
    # collective for the slow one, so their recorded durations are the long ones.
    durations = {}
    for path in sorted(trace_dir.glob("rank*.json")):
        state = json.loads(path.read_text(encoding="utf-8"))
        completed = [
            ev["dur_us"]
            for ev in state["events"]
            if ev["op"] == "all_reduce" and ev["dur_us"] is not None
        ]
        durations[state["rank"]] = sorted(completed)[len(completed) // 2]
    assert durations[2] < max(durations.values()), durations
