"""Show what a desynchronized job looks like when it is diagnosable.

    python examples/desync_demo.py

A classic production failure: one rank takes a different code path (an
uneven data loader, a rank-dependent early return, a conditional log step)
and issues a different collective than everyone else. Without instrumentation
the job simply stops making progress, and there is nothing in the logs to say
why.

Here the same failure is injected deliberately. Because every collective is
sequence-numbered and recorded, the timeout says which rank is out of step,
and the diagnose tool points at the exact collective where the streams
diverged.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import torch

import mini_nccl as mn
from mini_nccl import collectives as c

TRACE_DIR = Path(__file__).parent.parent / "data" / "desync_trace"
STRAGGLER = 2


def worker(pg) -> None:
    """Every rank runs three all-reduces. Rank 2 "forgets" the second one."""
    tensor = torch.ones(4096)
    for step in range(3):
        if step == 1 and pg.rank == STRAGGLER:
            continue  # the bug: a rank-dependent branch around a collective
        c.all_reduce(pg, tensor)
    c.barrier(pg)


def main() -> None:
    if TRACE_DIR.exists():
        shutil.rmtree(TRACE_DIR)

    print(f"running 3 ranks; rank {STRAGGLER} will skip one all-reduce\n", flush=True)
    try:
        mn.run(worker, 3, timeout=90.0, op_timeout=5.0, trace_dir=TRACE_DIR)
    except RuntimeError as exc:
        lines = str(exc).splitlines()
        start = next(
            (i for i, line in enumerate(lines) if "CollectiveTimeoutError" in line), None
        )
        print("=" * 72, flush=True)
        print("the job raised instead of hanging. What one rank reported:", flush=True)
        print("=" * 72, flush=True)
        print("\n".join(lines[start : start + 5] if start is not None else lines[-6:]),
              flush=True)
    else:
        print("unexpectedly succeeded (nothing to diagnose)", flush=True)
        return

    print(flush=True)
    print("=" * 72, flush=True)
    print(f"python -m mini_nccl.diagnose {TRACE_DIR.name}", flush=True)
    print("=" * 72, flush=True)
    subprocess.run(
        [sys.executable, "-m", "mini_nccl.diagnose", str(TRACE_DIR)],
        check=False,
    )


if __name__ == "__main__":
    main()
