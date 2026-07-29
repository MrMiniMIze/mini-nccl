"""Spawn-based local launcher for multi-process runs.

``run(fn, world_size)`` starts one process per rank, builds a
``ProcessGroup`` in each, calls ``fn(pg, *args)``, and returns the per-rank
results. Ports are allocated fresh for every run so parallel test sessions
don't collide. Worker exceptions are captured with their tracebacks and
re-raised in the parent; a worker that dies without reporting (or hangs)
turns into a timeout error rather than a silent deadlock.
"""

from __future__ import annotations

import multiprocessing as mp
import socket
import time
import traceback
from queue import Empty

from .process_group import ProcessGroup


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


def _worker(fn, rank, world_size, addrs, args, result_q) -> None:
    try:
        pg = ProcessGroup(rank, world_size, addrs)
        try:
            result = fn(pg, *args)
        finally:
            pg.close()
        result_q.put((rank, "ok", result))
    except BaseException:
        result_q.put((rank, "err", traceback.format_exc()))
        raise SystemExit(1)


def run(fn, world_size: int, *args, timeout: float = 300.0) -> list:
    """Run ``fn(pg, *args)`` on ``world_size`` local processes.

    Returns a list of per-rank return values, indexed by rank. ``fn`` must
    be picklable (defined at module level) because workers are spawned.
    """
    ctx = mp.get_context("spawn")
    addrs = [("127.0.0.1", p) for p in _free_ports(world_size)]
    result_q = ctx.Queue()
    procs = [
        ctx.Process(target=_worker, args=(fn, r, world_size, addrs, args, result_q), daemon=True)
        for r in range(world_size)
    ]
    for p in procs:
        p.start()

    results: dict[int, object] = {}
    errors: dict[int | str, str] = {}
    deadline = time.monotonic() + timeout
    for _ in range(world_size):
        remaining = max(0.1, deadline - time.monotonic())
        try:
            rank, status, payload = result_q.get(timeout=remaining)
        except Empty:
            exitcodes = [p.exitcode for p in procs]
            errors["timeout"] = (
                f"timed out after {timeout}s waiting for workers; exitcodes={exitcodes}"
            )
            break
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
        raise RuntimeError(f"{len(errors)} worker failure(s):\n\n{detail}")
    return [results[r] for r in range(world_size)]
