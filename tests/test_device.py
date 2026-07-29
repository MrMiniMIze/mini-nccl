"""Staged and pipelined collectives.

The chunking and double-buffering logic is device-independent, so it is tested
on CPU tensors here, where the staging copies are ordinary memory copies. That
keeps the pipeline covered on machines without a GPU, which includes CI.

The CUDA-specific parts (pinned allocation, copy stream, events) are exercised
by the same tests when a GPU is present; they skip otherwise. If you have just
installed a CUDA build of torch, this file is what tells you the device path
works.
"""

from __future__ import annotations

import itertools

import pytest
import torch

from mini_nccl import collectives as c
from mini_nccl.device import (
    Staging,
    all_reduce_pipelined,
    all_reduce_staged,
    device_report,
)
from mini_nccl.launcher import run

CUDA = torch.cuda.is_available()
requires_cuda = pytest.mark.skipif(not CUDA, reason="no CUDA device available")


def _expected(world: int, numel: int) -> torch.Tensor:
    return torch.full((numel,), float(world * (world + 1) // 2))


def _contribution(rank: int, numel: int) -> torch.Tensor:
    return torch.full((numel,), float(rank + 1))


# ---- chunking, without any communication --------------------------------


def test_staging_covers_the_payload_exactly() -> None:
    staging = Staging(1000, torch.float32, torch.device("cpu"), chunk_bytes=400)
    ranges = staging.ranges(1000)
    assert ranges[0][0] == 0
    assert ranges[-1][1] == 1000
    for (_, end), (start, _) in itertools.pairwise(ranges):
        assert end == start, ranges  # contiguous, no gaps or overlaps
    assert sum(hi - lo for lo, hi in ranges) == 1000


def test_staging_handles_payloads_smaller_than_a_chunk() -> None:
    staging = Staging(3, torch.float32, torch.device("cpu"), chunk_bytes=1 << 20)
    assert staging.ranges(3) == [(0, 3)]
    assert staging.n_chunks == 1
    assert len(staging.slots) == 1


# ---- pipelined ring on CPU tensors --------------------------------------


def _pipelined_worker(pg, chunk_bytes: int) -> None:
    for numel in (1, 7, 1000, 100_003):
        x = _contribution(pg.rank, numel)
        all_reduce_pipelined(pg, x, chunk_bytes=chunk_bytes)
        torch.testing.assert_close(x, _expected(pg.world_size, numel))


@pytest.mark.parametrize("chunk_bytes", [64, 4096, 1 << 20])
def test_pipelined_matches_expected_for_any_chunk_size(chunk_bytes: int) -> None:
    """Small chunk sizes force many chunks through the double buffer."""
    run(_pipelined_worker, 4, chunk_bytes)


def _pipelined_vs_plain_worker(pg) -> None:
    gen = torch.Generator().manual_seed(400 + pg.rank)
    for numel in (999, 65_536):
        x = torch.randn(numel, generator=gen)
        reference = x.clone()
        all_reduce_pipelined(pg, x, chunk_bytes=8192)
        c.all_reduce(pg, reference, algorithm="ring")
        torch.testing.assert_close(x, reference, rtol=1e-5, atol=1e-6)


def test_pipelined_agrees_with_the_plain_ring() -> None:
    run(_pipelined_vs_plain_worker, 3)


def _staged_worker(pg) -> None:
    """On CPU tensors, all_reduce_staged is the ordinary collective."""
    for numel in (16, 5000):
        x = _contribution(pg.rank, numel)
        all_reduce_staged(pg, x, algorithm="ring")
        torch.testing.assert_close(x, _expected(pg.world_size, numel))


def test_staged_on_cpu_tensors() -> None:
    run(_staged_worker, 2)


def _rejects_worker(pg) -> None:
    non_contiguous = torch.zeros(4, 4).t()
    with pytest.raises(ValueError, match="contiguous"):
        all_reduce_pipelined(pg, non_contiguous)


def test_pipelined_rejects_non_contiguous() -> None:
    run(_rejects_worker, 2)


# ---- the real device path, when there is one ---------------------------


def test_device_report_is_honest() -> None:
    report = device_report()
    assert report["cuda_available"] == CUDA
    if CUDA:
        assert report["device_count"] >= 1
        assert "compute_capability" in report


def test_cuda_api_surface_is_what_the_code_calls() -> None:
    """Shallow but useful: the CUDA branch cannot run here, so at least check
    that every name and keyword it uses exists in this torch build.

    This catches the failure mode that actually bites untested code (a renamed
    method, a keyword that moved) without pretending to verify behavior.
    """
    assert hasattr(torch.cuda, "Stream")
    assert hasattr(torch.cuda, "Event")
    for method in ("record", "synchronize"):
        assert hasattr(torch.cuda.Event, method), method
    assert hasattr(torch.cuda.Stream, "wait_event")
    assert hasattr(torch.cuda, "stream") and hasattr(torch.cuda, "current_stream")

    # The keywords the staging path passes, validated on CPU where they are
    # accepted and ignored.
    assert torch.empty(4, dtype=torch.float32, pin_memory=False).numel() == 4
    assert torch.empty(4).copy_(torch.zeros(4), non_blocking=True).numel() == 4
    assert hasattr(torch.Tensor, "is_pinned")


def _cuda_worker(pg) -> None:
    device = torch.device("cuda")
    for numel in (1, 4096, 250_003):
        pipelined = _contribution(pg.rank, numel).to(device)
        all_reduce_pipelined(pg, pipelined, chunk_bytes=1 << 16)
        torch.testing.assert_close(pipelined.cpu(), _expected(pg.world_size, numel))

        staged = _contribution(pg.rank, numel).to(device)
        all_reduce_staged(pg, staged, algorithm="ring")
        torch.testing.assert_close(staged.cpu(), _expected(pg.world_size, numel))


@requires_cuda
def test_cuda_tensors_all_reduce() -> None:
    """Both device paths must produce the same answer as the CPU ones.

    Every rank shares one GPU here, which is fine for correctness even though
    it says nothing about performance.
    """
    run(_cuda_worker, 2)


def _cuda_staging_worker(pg) -> None:
    """Pinned staging and the copy stream must actually be in use."""
    staging = Staging(1 << 14, torch.float32, torch.device("cuda"), chunk_bytes=1 << 12)
    assert staging.accelerated
    assert staging.stream is not None
    assert all(slot.send.is_pinned() for slot in staging.slots)
    assert all(slot.recv.is_pinned() for slot in staging.slots)
    assert len(staging.slots) == 2, "double buffering is what enables overlap"
    c.barrier(pg)


@requires_cuda
def test_cuda_staging_uses_pinned_memory_and_a_stream() -> None:
    run(_cuda_staging_worker, 2)
