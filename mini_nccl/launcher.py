"""Spawn-based local launcher for multi-process runs.

``run(fn, world_size)`` starts one process per rank, builds a
``ProcessGroup`` in each, calls ``fn(pg, *args)``, and returns the per-rank
results. Ports are allocated fresh for every run so parallel test sessions
don't collide. Worker exceptions are captured with their tracebacks and
re-raised in the parent; a worker that dies without reporting (or hangs)
turns into a timeout error rather than a silent deadlock.

Passing ``trace_dir`` turns on the flight recorder in every worker and dumps
each rank's event log there, even if the worker fails, which is exactly when
those logs matter. See ``python -m mini_nccl.diagnose``.
"""

from __future__ import annotations

import multiprocessing as mp
import socket
import time
import traceback
from pathlib import Path
from queue import Empty

from .process_group import DEFAULT_N_CHANNELS, ProcessGroup
from .transport import DEFAULT_OP_TIMEOUT


def _free_ports(count: int) -> list[int]:
    socks, ports = [], []
    for _ in range(count):
        s = socket.socket()
        s.bind(("127.0.0.1", 0))
        socks.append(s)
        ports.append(s.getsockname()[1])
    for s in socks:
        s.close()
    return ports


def _worker(fn, rank, world_size, addrs, args, result_q, options) -> None:
    trace_dir = options.get("trace_dir")
    pg = None
    try:
        pg = ProcessGroup(
            rank,
            world_size,
            addrs,
            n_channels=options.get("n_channels", DEFAULT_N_CHANNELS),
            op_timeout=options.get("op_timeout", DEFAULT_OP_TIMEOUT),
            trace=True if trace_dir else None,
        )
        try:
            result = fn(pg, *args)
        finally:
            if trace_dir:
                pg.recorder.dump(trace_dir)
            pg.close()
        result_q.put((rank, "ok", result))
    except BaseException:
        if pg is not None and trace_dir:
            try:
                pg.recorder.dump(trace_dir)
            except OSError:
                pass
        result_q.put((rank, "err", traceback.format_exc()))
        raise SystemExit(1)


def run(
    fn,
    world_size: int,
    *args,
    timeout: float = 300.0,
    n_channels: int = DEFAULT_N_CHANNELS,
    op_timeout: float = DEFAULT_OP_TIMEOUT,
    trace_dir: str | Path | None = None,
) -> list:
    """Run ``fn(pg, *args)`` on ``world_size`` local processes.

    Returns a list of per-rank return values, indexed by rank. ``fn`` must
    be picklable (defined at module level) because workers are spawned.
    """
    ctx = mp.get_context("spawn")
    addrs = [("127.0.0.1", p) for p in _free_ports(world_size)]
    options = {
        "n_channels": n_channels,
        "op_timeout": op_timeout,
        "trace_dir": str(trace_dir) if trace_dir else None,
    }
    result_q = ctx.Queue()
    procs = [
        ctx.Process(
            target=_worker,
            args=(fn, r, world_size, addrs, args, result_q, options),
            daemon=True,
        )
        for r in range(world_size)
    ]
    for p in procs:
        p.start()

    results: dict[int, object] = {}
    errors: dict[int | str, str] = {}
    reported: set[int] = set()
    deadline = time.monotonic() + timeout
    while len(reported) < world_size:
        try:
            rank, status, payload = result_q.get(timeout=0.2)
        except Empty:
            # A rank killed hard (segfault, os._exit, OOM killer) never
            # reports. Notice that rather than waiting out the full timeout:
            # a hung job and a dead job need different answers.
            silent = [
                r
                for r, p in enumerate(procs)
                if p.exitcode is not None and r not in reported
            ]
            if len(reported) + len(silent) == world_size:
                for r in silent:
                    errors[r] = (
                        f"rank {r} exited with code {procs[r].exitcode} without "
                        f"reporting (killed or crashed hard)"
                    )
                break
            if time.monotonic() > deadline:
                exitcodes = [p.exitcode for p in procs]
                errors["timeout"] = (
                    f"timed out after {timeout}s waiting for ranks "
                    f"{sorted(set(range(world_size)) - reported)}; exitcodes={exitcodes}"
                )
                break
            continue
        reported.add(rank)
        if status == "ok":
            results[rank] = payload
        else:
            errors[rank] = payload

    for p in procs:
        p.join(5.0)
    for p in procs:
        if p.is_alive():
            p.terminate()

    if errors:
        detail = "\n\n".join(f"[{key}]\n{tb}" for key, tb in errors.items())
        hint = ""
        if trace_dir:
            hint = f"\n\nFlight recorder logs: python -m mini_nccl.diagnose {trace_dir}"
        raise RuntimeError(f"{len(errors)} worker failure(s):\n\n{detail}{hint}")
    return [results[r] for r in range(world_size)]
