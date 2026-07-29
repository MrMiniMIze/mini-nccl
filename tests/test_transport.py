"""Point-to-point transport tests."""

from __future__ import annotations

import pytest
import torch

from mini_nccl.launcher import run
from mini_nccl.transport import byte_view


def _pingpong_worker(pg) -> None:
    payload = torch.arange(10_000, dtype=torch.float64)
    if pg.rank == 0:
        pg.send(payload, dst=1)
        echo = torch.empty_like(payload)
        pg.recv(echo, src=1)
        torch.testing.assert_close(echo, payload * 2)
    else:
        buf = torch.empty_like(payload)
        pg.recv(buf, src=0)
        pg.send(buf * 2, dst=0)


def test_send_recv_pingpong() -> None:
    run(_pingpong_worker, 2)


def _sendrecv_ring_worker(pg) -> None:
    """Every rank simultaneously passes a token right; no deadlock allowed."""
    W, r = pg.world_size, pg.rank
    token = torch.full((4096,), float(r))
    incoming = torch.empty_like(token)
    pg.send_recv(token, (r + 1) % W, incoming, (r - 1) % W)
    torch.testing.assert_close(incoming, torch.full((4096,), float((r - 1) % W)))


def test_full_duplex_ring_rotation() -> None:
    run(_sendrecv_ring_worker, 3)


def test_byte_view_rejects_non_contiguous() -> None:
    t = torch.zeros(4, 4).t()
    assert not t.is_contiguous()
    with pytest.raises(ValueError, match="contiguous"):
        byte_view(t)


def test_byte_view_is_zero_copy() -> None:
    t = torch.zeros(4, dtype=torch.float32)
    view = byte_view(t)
    assert len(view) == 16
    view[0] = 1  # poke one byte; the tensor must see it
    assert t[0] != 0
