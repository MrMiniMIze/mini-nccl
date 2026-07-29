"""Correctness tests for every collective, checked against locally
recomputed references.

Each rank derives its input from ``torch.manual_seed(rank)``, so every rank
can independently reconstruct all inputs and compute the exact expected
result: no golden files, no cross-process oracles.

To keep process-spawn overhead sane, one worker runs the whole battery of
(algorithm x op x size x dtype) cases inside a single process group, and the
test is parametrized only over world size.
"""

from __future__ import annotations

import pytest
import torch

from mini_nccl import collectives as c
from mini_nccl.launcher import run

ALGORITHMS = ("ring", "tree", "halving", "naive")
# Sizes chosen to hit the edge cases: single element, smaller than the
# world size, non-divisible by world size, and large enough to span many
# socket buffers.
NUMELS = (1, 3, 1024, 262_147)


def _rank_tensor(rank: int, numel: int, dtype: torch.dtype) -> torch.Tensor:
    g = torch.Generator().manual_seed(1000 + rank)
    if dtype.is_floating_point:
        return torch.randn(numel, generator=g, dtype=dtype)
    return torch.randint(-50, 50, (numel,), generator=g, dtype=dtype)


def _expected_reduce(world: int, numel: int, dtype: torch.dtype, op: str) -> torch.Tensor:
    tensors = [_rank_tensor(r, numel, dtype) for r in range(world)]
    acc = tensors[0].clone()
    for t in tensors[1:]:
        if op == "sum":
            acc += t
        elif op == "max":
            acc = torch.maximum(acc, t)
        elif op == "min":
            acc = torch.minimum(acc, t)
        elif op == "prod":
            acc *= t
    return acc


def _battery_worker(pg) -> None:
    W = pg.world_size

    # all_reduce: every algorithm, several ops/dtypes/sizes
    for algorithm in ALGORITHMS:
        for op in ("sum", "max", "min", "prod"):
            for dtype in (torch.float32, torch.int64):
                for numel in NUMELS:
                    if op == "prod" and numel > 1024:
                        continue  # avoid float overflow noise on huge products
                    t = _rank_tensor(pg.rank, numel, dtype)
                    c.all_reduce(pg, t, op=op, algorithm=algorithm)
                    expected = _expected_reduce(W, numel, dtype, op)
                    case = f"{algorithm}/{op}/{dtype}/{numel}"
                    torch.testing.assert_close(
                        t, expected, rtol=1e-4, atol=1e-4,
                        msg=lambda m, case=case: f"all_reduce {case}: {m}",
                    )

    # auto dispatch: small -> tree, large -> ring; both must be correct
    for numel in (16, 1_000_000):
        t = _rank_tensor(pg.rank, numel, torch.float32)
        c.all_reduce(pg, t, algorithm="auto")
        torch.testing.assert_close(
            t, _expected_reduce(W, numel, torch.float32, "sum"), rtol=1e-4, atol=1e-4
        )

    # broadcast from every possible root
    for src in range(W):
        t = _rank_tensor(pg.rank, 4097, torch.float32)
        c.broadcast(pg, t, src=src)
        torch.testing.assert_close(t, _rank_tensor(src, 4097, torch.float32))

    # all_gather returns every rank's tensor, in rank order
    t = _rank_tensor(pg.rank, 517, torch.float32).view(11, 47)
    gathered = c.all_gather(pg, t)
    assert len(gathered) == W
    for r, g in enumerate(gathered):
        torch.testing.assert_close(g, _rank_tensor(r, 517, torch.float32).view(11, 47))

    # reduce_scatter: rank r gets the r-th chunk of the summed tensor
    numel = 12 * W
    t = _rank_tensor(pg.rank, numel, torch.float32)
    original = t.clone()
    mine = c.reduce_scatter(pg, t, op="sum")
    expected = _expected_reduce(W, numel, torch.float32, "sum").view(W, -1)[pg.rank]
    torch.testing.assert_close(mine, expected, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(t, original)  # input must be untouched

    # all_to_all: out chunk i is what rank i sent us, i.e. its chunk r
    numel = 9 * W
    t = _rank_tensor(pg.rank, numel, torch.float32)
    out = c.all_to_all(pg, t)
    for src in range(W):
        expected = _rank_tensor(src, numel, torch.float32).view(W, -1)[pg.rank]
        torch.testing.assert_close(out.view(W, -1)[src], expected)

    # barrier completes without deadlock
    c.barrier(pg)


@pytest.mark.parametrize("world_size", [2, 3, 4])
def test_collective_battery(world_size: int) -> None:
    run(_battery_worker, world_size)


def _multichannel_worker(pg) -> None:
    """A payload the ring must split across every channel."""
    assert pg.n_channels == 4
    numel = 1_000_003  # deliberately divisible by neither channels nor ranks
    assert c._n_channels_for(pg, numel * 4) == 4, "test must exercise all channels"
    for algorithm in ("ring", "halving"):
        t = _rank_tensor(pg.rank, numel, torch.float32)
        c.all_reduce(pg, t, algorithm=algorithm)
        torch.testing.assert_close(
            t, _expected_reduce(pg.world_size, numel, torch.float32, "sum"),
            rtol=1e-4, atol=1e-3,
        )


def test_multichannel_large_allreduce(monkeypatch) -> None:
    # Lower the split threshold so a small, fast tensor still spans channels;
    # spawned workers inherit this environment.
    monkeypatch.setenv("MINI_NCCL_CHANNEL_MIN_BYTES", str(64 * 1024))
    run(_multichannel_worker, 4, n_channels=4)


def _single_channel_worker(pg) -> None:
    assert pg.n_channels == 1
    numel = 1_000_001
    t = _rank_tensor(pg.rank, numel, torch.float32)
    c.all_reduce(pg, t, algorithm="ring")
    torch.testing.assert_close(
        t, _expected_reduce(pg.world_size, numel, torch.float32, "sum"),
        rtol=1e-4, atol=1e-3,
    )


def test_single_channel_still_correct() -> None:
    run(_single_channel_worker, 2, n_channels=1)


def _failing_worker(pg) -> None:
    if pg.rank == 1:
        raise ValueError("intentional failure from rank 1")
    c.barrier(pg)


def test_worker_errors_propagate() -> None:
    with pytest.raises(RuntimeError, match="intentional failure from rank 1"):
        run(_failing_worker, 2, timeout=60.0)


def test_world_size_one_is_identity() -> None:
    def check(pg):
        t = torch.arange(8, dtype=torch.float32)
        c.all_reduce(pg, t)
        torch.testing.assert_close(t, torch.arange(8, dtype=torch.float32))

    # world_size 1 needs no sockets at all; run inline
    from mini_nccl.process_group import ProcessGroup

    pg = ProcessGroup(0, 1)
    try:
        check(pg)
        c.barrier(pg)
    finally:
        pg.close()
