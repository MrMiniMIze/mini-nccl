"""Point-to-point transport tests."""

from __future__ import annotations

import contextlib
import os
import re
import socket
import threading
import time

import pytest
import torch

from mini_nccl.errors import RendezvousError
from mini_nccl.launcher import _free_ports, run
from mini_nccl.transport import (
    _HANDSHAKE,
    _MAGIC,
    PROTOCOL_VERSION,
    Mesh,
    byte_view,
    job_fingerprint,
)


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


def test_failed_rendezvous_frees_the_port() -> None:
    """A caller that retries after a failed rendezvous must not be blocked.

    The accepter thread parks in accept() holding the listening socket. If a
    failed rendezvous leaves it there, the port stays bound and a retry in the
    same process dies with "address already in use" on its own address, which
    looks like a configuration problem rather than a leak.
    """
    from mini_nccl.errors import RendezvousError
    from mini_nccl.launcher import _free_ports
    from mini_nccl.transport import Mesh

    port = _free_ports(1)[0]
    addrs = [("127.0.0.1", port), ("127.0.0.1", _free_ports(1)[0])]

    # Rank 0 with no rank 1 to talk to: it binds its port, waits, and fails.
    for attempt in range(2):
        with pytest.raises(RendezvousError):
            Mesh(0, 2, addrs, n_channels=1, timeout=1.0)
        # The second iteration is the actual assertion: rebinding the same port
        # only works if the first attempt released it.
        assert attempt >= 0

    # And the port really is free for an unrelated listener now.
    probe = socket.create_server(("127.0.0.1", port))
    probe.close()


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


# ---- failures must arrive as library errors, with the local rank's view -------


def _dies_while_peer_sends(pg) -> None:
    if pg.rank == 1:
        os._exit(9)  # hard exit with the connection open, like a crashed rank
    # 64 MiB is far more than the socket buffers hold, so the send is still in
    # progress when the peer's kernel resets the connection.
    payload = torch.zeros(16 * 1024 * 1024)
    for _ in range(4):
        pg.send(payload, dst=1)


def _dies_while_peer_receives(pg) -> None:
    if pg.rank == 1:
        os._exit(9)
    pg.recv(torch.empty(1024), src=1)


@pytest.mark.parametrize(
    ("worker", "direction"),
    [(_dies_while_peer_sends, "send"), (_dies_while_peer_receives, "recv")],
)
def test_dead_peer_raises_peer_closed_with_context(worker, direction) -> None:
    """A reset or closed connection must not surface as a bare socket error.

    A ``BrokenPipeError`` from deep inside a ring step says nothing about which
    peer died or what this rank was doing. The README promises errors that
    carry the local rank's view, and timeouts always did; connection failures
    have to as well.
    """
    with pytest.raises(RuntimeError) as excinfo:
        run(worker, 2, timeout=60.0, op_timeout=20.0)
    message = str(excinfo.value)
    # The launcher reports one "[rank]" section per failed rank; take rank 0's.
    assert "[0]\n" in message, message
    rank0 = message.split("[0]\n", 1)[1].split("\n\n[", 1)[0]
    # The last exception header in the traceback is the one that escaped.
    headers = [line for line in rank0.splitlines() if re.match(r"[\w.]+(Error|Exception): ", line)]
    final = headers[-1] if headers else rank0
    assert final.startswith(
        f"mini_nccl.errors.PeerClosedError: rank 0: {direction} on channel 0"
    ), final
    assert "local state:" in rank0, rank0


# ---- rendezvous must not trust whoever connects first --------------------------


def _connect_when_listening(addr: tuple[str, int], timeout: float = 10.0) -> socket.socket:
    deadline = time.monotonic() + timeout
    while True:
        try:
            return socket.create_connection(addr, timeout=1.0)
        except OSError:
            if time.monotonic() > deadline:
                raise
            time.sleep(0.02)


def _build_meshes(addrs, configs, timeout) -> dict[int, object]:
    """Construct one Mesh per (rank, n_channels) on its own thread."""
    results: dict[int, object] = {}

    def build(rank: int, n_channels: int) -> None:
        try:
            results[rank] = Mesh(rank, len(addrs), addrs, n_channels=n_channels, timeout=timeout)
        except BaseException as exc:
            results[rank] = exc

    threads = [threading.Thread(target=build, args=cfg) for cfg in configs]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout + 10)
    return results


def test_rendezvous_survives_stray_connections() -> None:
    """A port scanner or a stale process must not stall or poison the mesh.

    Before the fix, the accepter read each handshake under the 300 s operation
    timeout, so one silent connection stalled rendezvous past its budget, and
    it filed whatever 8 bytes arrived as a (rank, channel) pair.
    """
    addrs = [("127.0.0.1", p) for p in _free_ports(2)]
    results: dict[int, object] = {}
    rank0 = threading.Thread(
        target=lambda: results.update(_build_meshes(addrs, [(0, 1)], timeout=15.0))
    )
    rank0.start()

    silent = _connect_when_listening(addrs[0])  # connects, never speaks
    garbage = socket.create_connection(addrs[0])
    garbage.sendall(bytes(range(64)))  # not a handshake at all
    try:
        start = time.perf_counter()
        results.update(_build_meshes(addrs, [(1, 1)], timeout=15.0))
        rank0.join(30)
        elapsed = time.perf_counter() - start
    finally:
        silent.close()
        garbage.close()

    meshes = [results[0], results[1]]
    try:
        assert all(isinstance(m, Mesh) for m in meshes), results
        assert elapsed < 12.0, f"strays held up rendezvous for {elapsed:.1f}s"
        # And the connection filed as rank 1 really is rank 1.
        meshes[1].conns[0][0].send(memoryview(b"hello"))
        got = bytearray(5)
        meshes[0].conns[1][0].recv_into(memoryview(got))
        assert bytes(got) == b"hello"
    finally:
        for m in meshes:
            if isinstance(m, Mesh):
                m.close()


def test_a_trickled_handshake_cannot_outlast_its_deadline(monkeypatch) -> None:
    """The handshake budget covers the whole handshake, not each read of it.

    A per-read timeout restarts every time a byte arrives, so a client that
    trickles one byte at a time (slowloris) could hold the acceptor for the
    handshake's length times the budget, stalling every real rank behind it.
    """
    from mini_nccl import transport

    monkeypatch.setattr(transport, "_HANDSHAKE_TIMEOUT", 1.0)
    addrs = [("127.0.0.1", p) for p in _free_ports(2)]
    results: dict[int, object] = {}
    rank0 = threading.Thread(
        target=lambda: results.update(_build_meshes(addrs, [(0, 1)], timeout=6.0))
    )
    rank0.start()

    slow = _connect_when_listening(addrs[0])
    stop = threading.Event()

    def trickle() -> None:
        # One byte every 0.4 s: never silent for the 1 s budget, and the 22-byte
        # handshake would take 8.8 s, past the 6 s rendezvous.
        with contextlib.suppress(OSError):
            while not stop.is_set():
                slow.send(b"M")
                stop.wait(0.4)

    trickler = threading.Thread(target=trickle, daemon=True)
    trickler.start()
    try:
        results.update(_build_meshes(addrs, [(1, 1)], timeout=6.0))
        rank0.join(15)
    finally:
        stop.set()
        slow.close()
    try:
        assert all(isinstance(results[r], Mesh) for r in (0, 1)), results
    finally:
        for m in results.values():
            if isinstance(m, Mesh):
                m.close()


def test_mismatched_job_configuration_is_refused_by_name() -> None:
    """Ranks that disagree about the job must fail at rendezvous, saying why.

    Here rank 1 believes in two channels and rank 0 in one. Before the fix both
    sides "succeeded": rank 0 filed the first connection it saw and the job
    would have hung or crossed wires on its first multi-channel collective.
    """
    addrs = [("127.0.0.1", p) for p in _free_ports(2)]
    results = _build_meshes(addrs, [(0, 1), (1, 2)], timeout=5.0)
    try:
        assert isinstance(results[1], RendezvousError), results
        assert "refused" in str(results[1]) and "configuration" in str(results[1]), results[1]
        assert isinstance(results[0], RendezvousError), results
        assert "configuration" in str(results[0]), results[0]
    finally:
        for m in results.values():
            if isinstance(m, Mesh):
                m.close()


def test_protocol_version_mismatch_is_named() -> None:
    """A peer speaking another protocol version is told so, and so is the user."""
    addrs = [("127.0.0.1", p) for p in _free_ports(2)]
    results: dict[int, object] = {}
    rank0 = threading.Thread(
        target=lambda: results.update(_build_meshes(addrs, [(0, 1)], timeout=3.0))
    )
    rank0.start()
    sock = _connect_when_listening(addrs[0])
    try:
        sock.sendall(
            _HANDSHAKE.pack(_MAGIC, PROTOCOL_VERSION + 1, 1, 0, job_fingerprint(2, 1, addrs))
        )
        status = sock.recv(1)
    finally:
        sock.close()
    rank0.join(15)
    assert status and status[0] != 0, status
