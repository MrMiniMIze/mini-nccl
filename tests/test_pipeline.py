"""Pipeline parallel correctness and the memory claim behind 1F1B.

Splitting a model by depth must not change what it learns, so the bar is the
same as for DDP and FSDP: gradients and parameters after a step must match
single-process training of the whole model on the whole batch. A pipeline
schedule that is subtly wrong (a gradient matched to the wrong microbatch, a
queue popped in the wrong order) still runs and still converges to something,
which is exactly why this needs a numeric reference rather than a smoke test.

The second claim under test is the reason 1F1B exists at all: it reaches the
same bubble as GPipe while holding at most ``W-s`` microbatches on stage ``s``
instead of all ``M``.
"""

from __future__ import annotations

import json

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from mini_nccl.launcher import run
from mini_nccl.pipeline import PIPELINE_CHANNEL, PipelineParallel

WIDTH = 8
LAYERS_PER_STAGE = 2


def _build_layers(n_stages: int) -> list[nn.Module]:
    """The full model, identical on every rank thanks to the seed."""
    torch.manual_seed(4)
    return [
        nn.Sequential(nn.Linear(WIDTH, WIDTH), nn.Tanh())
        for _ in range(n_stages * LAYERS_PER_STAGE)
    ]


def _my_slice(layers: list[nn.Module], rank: int) -> nn.Module:
    start = rank * LAYERS_PER_STAGE
    return nn.Sequential(*layers[start : start + LAYERS_PER_STAGE])


def _parity_worker(pg, schedule: str, n_microbatches: int, batch: int) -> None:
    layers = _build_layers(pg.world_size)
    reference = nn.Sequential(*layers)  # shares parameters with the stages
    stage = _my_slice(layers, pg.rank)

    pipeline = PipelineParallel(
        stage,
        pg,
        activation_shape=(batch // n_microbatches, WIDTH),
        loss_fn=F.mse_loss,
        schedule=schedule,
    )

    gen = torch.Generator().manual_seed(19)
    x = torch.randn(batch, WIDTH, generator=gen)
    y = torch.randn(batch, WIDTH, generator=gen)

    # Pipeline pass: gradients accumulate into this stage's parameters.
    for param in reference.parameters():
        param.grad = None
    loss = pipeline.step(
        inputs=x if pg.rank == 0 else None,
        targets=y if pg.rank == pg.world_size - 1 else None,
        n_microbatches=n_microbatches,
    )
    pipeline_grads = [
        None if p.grad is None else p.grad.clone() for p in stage.parameters()
    ]

    # Reference pass: the whole model, the whole batch, one process.
    for param in reference.parameters():
        param.grad = None
    reference_loss = F.mse_loss(reference(x), y)
    reference_loss.backward()

    for i, param in enumerate(stage.parameters()):
        assert pipeline_grads[i] is not None, f"stage param {i} got no gradient"
        torch.testing.assert_close(
            pipeline_grads[i], param.grad, rtol=1e-4, atol=1e-6,
            msg=lambda m, i=i: f"{schedule} stage {pg.rank} param {i}: {m}",
        )

    if pg.rank == pg.world_size - 1:
        torch.testing.assert_close(
            torch.tensor(loss), reference_loss.detach(), rtol=1e-5, atol=1e-6
        )


def test_1f1b_matches_single_process_two_stages() -> None:
    run(_parity_worker, 2, "1f1b", 4, 16)


def test_1f1b_matches_single_process_four_stages() -> None:
    run(_parity_worker, 4, "1f1b", 8, 16)


def test_gpipe_matches_single_process() -> None:
    run(_parity_worker, 4, "gpipe", 8, 16)


def test_1f1b_with_one_microbatch_per_stage() -> None:
    """Fewer microbatches than stages: warmup gets clamped."""
    run(_parity_worker, 4, "1f1b", 2, 16)


def test_single_microbatch_degenerates_cleanly() -> None:
    run(_parity_worker, 3, "1f1b", 1, 12)


def _depth_worker(pg, schedule: str, n_microbatches: int) -> dict:
    layers = _build_layers(pg.world_size)
    stage = _my_slice(layers, pg.rank)
    pipeline = PipelineParallel(
        stage, pg, activation_shape=(2, WIDTH), loss_fn=F.mse_loss, schedule=schedule
    )
    batch = 2 * n_microbatches
    gen = torch.Generator().manual_seed(23)
    pipeline.step(
        inputs=torch.randn(batch, WIDTH, generator=gen) if pg.rank == 0 else None,
        targets=torch.randn(batch, WIDTH, generator=gen)
        if pg.rank == pg.world_size - 1
        else None,
        n_microbatches=n_microbatches,
    )
    return {"rank": pg.rank, "peak": pipeline.in_flight_peak}


def test_1f1b_bounds_in_flight_microbatches() -> None:
    """The memory claim: depth is bounded by stages, not by microbatch count."""
    world, n_micro = 4, 12
    reports = run(_depth_worker, world, "1f1b", n_micro)
    peaks = {r["rank"]: r["peak"] for r in reports}

    for rank, peak in peaks.items():
        bound = world - rank
        assert peak <= bound, f"rank {rank} held {peak}, expected at most {bound}"
        assert peak < n_micro, f"rank {rank} held {peak}, no better than GPipe"

    # The first stage carries the deepest queue, the last the shallowest.
    assert peaks[0] > peaks[world - 1], peaks
    assert peaks[world - 1] == 1, peaks


def test_gpipe_holds_every_microbatch() -> None:
    """The contrast: GPipe's depth grows with the microbatch count."""
    world, n_micro = 4, 12
    reports = run(_depth_worker, world, "gpipe", n_micro)
    for report in reports:
        assert report["peak"] == n_micro, report


def _traced_worker(pg) -> None:
    stage = nn.Sequential(nn.Linear(WIDTH, WIDTH), nn.Tanh())
    pipeline = PipelineParallel(stage, pg, (2, WIDTH), loss_fn=F.mse_loss)
    for _ in range(2):
        for param in stage.parameters():
            param.grad = None
        pipeline.step(
            inputs=torch.randn(8, WIDTH) if pg.rank == 0 else None,
            targets=torch.randn(8, WIDTH) if pg.rank == pg.world_size - 1 else None,
            n_microbatches=4,
        )


def test_pipeline_is_traced_without_faking_a_desync(tmp_path) -> None:
    """Pipeline events must be visible, and on a channel of their own.

    Stages legitimately issue different numbers of forwards and backwards, since
    warmup depends on the stage index. Recording them on the collective-order
    channel would therefore make every healthy pipeline look diverged, so they
    get their own channel and desync detection ignores them.
    """
    import subprocess
    import sys

    trace_dir = tmp_path / "trace"
    run(_traced_worker, 4, trace_dir=trace_dir, timeout=180.0)

    proc = subprocess.run(
        [sys.executable, "-m", "mini_nccl.diagnose", str(trace_dir)],
        capture_output=True, text=True,
    )
    assert proc.returncode == 0, f"a healthy pipeline was reported as broken:\n{proc.stdout}"
    assert "all ranks agree" in proc.stdout, proc.stdout
    # Present in the trace, under its own label.
    assert "'pipeline':" in proc.stdout, proc.stdout

    state = json.loads((trace_dir / "rank0.json").read_text(encoding="utf-8"))
    ops = [ev["op"] for ev in state["events"]]
    assert "pp_forward" in ops and "pp_backward" in ops, ops
    assert all(
        ev["channel"] == PIPELINE_CHANNEL
        for ev in state["events"]
        if ev["op"].startswith("pp_")
    )


def _rejects_worker(pg) -> None:
    """Argument validation, on a single stage so nothing is communicated.

    Deliberately world_size 1: a stage that raises mid-schedule would leave its
    neighbors waiting on activations that never arrive, so these checks all have
    to fire before any traffic, and testing them without a peer proves it.
    """
    stage = nn.Linear(WIDTH, WIDTH)

    with pytest.raises(ValueError, match="unknown schedule"):
        PipelineParallel(stage, pg, (2, WIDTH), loss_fn=F.mse_loss, schedule="nope")
    with pytest.raises(ValueError, match="needs loss_fn"):
        PipelineParallel(stage, pg, (2, WIDTH), loss_fn=None)

    pipeline = PipelineParallel(stage, pg, (2, WIDTH), loss_fn=F.mse_loss)
    with pytest.raises(ValueError, match="needs inputs"):
        pipeline.step(inputs=None, targets=torch.zeros(4, WIDTH), n_microbatches=2)
    with pytest.raises(ValueError, match="needs targets"):
        pipeline.step(inputs=torch.zeros(4, WIDTH), targets=None, n_microbatches=2)
    with pytest.raises(ValueError, match="not divisible"):
        pipeline.step(
            inputs=torch.zeros(5, WIDTH), targets=torch.zeros(5, WIDTH), n_microbatches=2
        )
    with pytest.raises(ValueError, match="at least 1"):
        pipeline.step(
            inputs=torch.zeros(4, WIDTH), targets=torch.zeros(4, WIDTH), n_microbatches=0
        )


def test_argument_errors_are_clear() -> None:
    run(_rejects_worker, 1)
