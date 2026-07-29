"""Failure behavior: a broken job must fail loudly and quickly.

A hang is the worst outcome in distributed training, because it burns the
whole allocation while telling you nothing. These tests assert the two
common failures surface as errors with a diagnosis instead:

- a rank dies mid-collective
- a rank issues a different collective than everyone else (desync)
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time

import pytest
import torch

from mini_nccl import collectives as c
from mini_nccl.launcher import run


def _dying_worker(pg) -> None:
    if pg.rank == 1:
        os._exit(9)  # hard exit: no traceback, no queue message, like a segfault
    for _ in range(50):
        c.all_reduce(pg, torch.ones(50_000))


def test_dead_rank_fails_fast() -> None:
    start = time.perf_counter()
    with pytest.raises(RuntimeError) as excinfo:
        run(_dying_worker, 3, timeout=60.0, op_timeout=20.0)
    elapsed = time.perf_counter() - start

    message = str(excinfo.value)
    assert "rank 1 exited with code" in message, message
    # Survivors must notice the dropped connection, not wait out the timeout.
    assert elapsed < 30.0, f"took {elapsed:.1f}s to notice a dead rank"


def _desync_worker(pg) -> None:
    """Rank 1 skips a collective the others perform."""
    tensor = torch.ones(1000)
    c.all_reduce(pg, tensor)
    if pg.rank != 1:
        c.all_reduce(pg, tensor)  # rank 1 never joins this one
    c.barrier(pg)


def test_desync_times_out_with_diagnosis(tmp_path) -> None:
    trace_dir = tmp_path / "trace"
    start = time.perf_counter()
    with pytest.raises(RuntimeError) as excinfo:
        run(_desync_worker, 3, timeout=90.0, op_timeout=3.0, trace_dir=trace_dir)
    elapsed = time.perf_counter() - start

    message = str(excinfo.value)
    assert "CollectiveTimeoutError" in message, message
    assert "collectives in identical order" in message, message
    assert elapsed < 45.0, f"desync took {elapsed:.1f}s to surface"

    # The flight recorder must have captured each rank's collective stream.
    dumps = sorted(trace_dir.glob("rank*.json"))
    assert len(dumps) == 3, f"expected 3 dumps, got {[p.name for p in dumps]}"
    second_op = {}
    for path in dumps:
        state = json.loads(path.read_text(encoding="utf-8"))
        order = sorted(
            (ev for ev in state["events"] if ev["channel"] == -1), key=lambda ev: ev["seq"]
        )
        second_op[state["rank"]] = order[1]["op"] if len(order) > 1 else None
    # Rank 1 moved on to the barrier while the others were still all-reducing.
    assert second_op[1] == "barrier", second_op
    assert second_op[0] == "all_reduce", second_op

    # And the diagnose CLI must name the divergence.
    proc = subprocess.run(
        [sys.executable, "-m", "mini_nccl.diagnose", str(trace_dir),
         "--trace", str(tmp_path / "merged.json")],
        capture_output=True, text=True,
    )
    assert "DESYNC at collective #1" in proc.stdout, proc.stdout
    assert "rank 1: barrier" in proc.stdout, proc.stdout
    assert (tmp_path / "merged.json").exists()


def _healthy_traced_worker(pg) -> None:
    for _ in range(3):
        c.all_reduce(pg, torch.ones(4096))
    c.barrier(pg)


def test_healthy_run_traces_clean(tmp_path) -> None:
    trace_dir = tmp_path / "trace"
    run(_healthy_traced_worker, 2, trace_dir=trace_dir)

    proc = subprocess.run(
        [sys.executable, "-m", "mini_nccl.diagnose", str(trace_dir)],
        capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stdout
    assert "all ranks agree" in proc.stdout, proc.stdout

    state = json.loads((trace_dir / "rank0.json").read_text(encoding="utf-8"))
    ops = [ev["op"] for ev in state["events"]]
    assert ops.count("all_reduce") == 3, ops
    assert "barrier" in ops
    assert all(ev["dur_us"] is not None for ev in state["events"])
