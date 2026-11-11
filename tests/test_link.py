"""The emulated link: it must never be faster than the rate it claims.

Every timing assertion here is a **lower bound**. A shaper can always come out
slower than its rate on a loaded CI runner, so an upper bound would be a test
of the machine, not of the code. But a shaper can never legitimately be
*faster* than its rate, so the lower bound is deterministic, and it is the half
that catches a broken shaper. (The same reasoning as test_overlap.py asserting
that a reduction was dispatched, rather than when the OS ran it.)
"""

from __future__ import annotations

import socket
import threading
import time

import pytest
import torch

from mini_nccl import LinkModel
from mini_nccl.launcher import run
from mini_nccl.transport import Connection, LinkClock, ShapedConnection

MB = 1e6


def test_link_model_from_env(monkeypatch) -> None:
    monkeypatch.setenv("MINI_NCCL_LINK_MBPS", "100")
    monkeypatch.setenv("MINI_NCCL_LINK_FLOW_MBPS", " ")
    monkeypatch.setenv("MINI_NCCL_LINK_LATENCY_US", "50")
    assert LinkModel.from_env() == LinkModel(mbps=100.0, flow_mbps=None, latency_us=50.0)
    for name in ("MINI_NCCL_LINK_MBPS", "MINI_NCCL_LINK_FLOW_MBPS", "MINI_NCCL_LINK_LATENCY_US"):
        monkeypatch.delenv(name)
    assert not LinkModel.from_env().shaped


def test_link_model_rejects_nonsense() -> None:
    with pytest.raises(ValueError, match="mbps"):
        LinkModel(mbps=0)
    with pytest.raises(ValueError, match="latency"):
        LinkModel(latency_us=-1)


def _shaped_pair(egress, ingress=None):
    """A loopback TCP pair: the near end shaped, the far end plain."""
    with socket.create_server(("127.0.0.1", 0)) as listener:
        near = socket.create_connection(listener.getsockname())
        far, _ = listener.accept()
    return ShapedConnection(near, 30.0, egress, ingress, 0.0), Connection(far, 30.0)


def _drain(conn, nbytes: int) -> threading.Thread:
    buf = bytearray(nbytes)
    t = threading.Thread(target=conn.recv_into, args=(memoryview(buf),))
    t.start()
    return t


def test_a_link_clock_never_beats_its_rate() -> None:
    rate = 20 * MB  # bytes per second
    total = 4 * 1024 * 1024
    sender, receiver = _shaped_pair((LinkClock(rate),))
    reader = _drain(receiver, total)
    try:
        start = time.perf_counter()
        sender.send(memoryview(bytearray(total)))
        elapsed = time.perf_counter() - start
    finally:
        reader.join(10)
        sender.close()
        receiver.close()
    floor = 0.9 * total / rate
    assert elapsed >= floor, f"{total} bytes at {rate / MB:.0f} MB/s took {elapsed:.3f}s"


def test_an_idle_link_banks_no_credit() -> None:
    """A small message after a pause still pays its own serialization delay.

    A token bucket with a burst allowance would let it through instantly, which
    erases exactly the small-message costs the tree/ring crossover depends on.
    """
    rate = 1 * MB
    small = 16 * 1024  # 16 ms at 1 MB/s
    sender, receiver = _shaped_pair((LinkClock(rate),))
    try:
        time.sleep(0.05)  # idle long enough to bank 50 KB if banking existed
        reader = _drain(receiver, small)
        start = time.perf_counter()
        sender.send(memoryview(bytearray(small)))
        elapsed = time.perf_counter() - start
        reader.join(10)
    finally:
        sender.close()
        receiver.close()
    assert elapsed >= 0.9 * small / rate, f"{elapsed * 1e3:.1f} ms"


def test_one_clock_is_shared_fairly_between_threads() -> None:
    """Two senders on one clock get its rate between them, not each."""
    rate = 20 * MB
    per_thread = 2 * 1024 * 1024
    clock = LinkClock(rate)
    pairs = [_shaped_pair((clock,)) for _ in range(2)]
    readers = [_drain(receiver, per_thread) for _, receiver in pairs]
    senders = [
        threading.Thread(target=sender.send, args=(memoryview(bytearray(per_thread)),))
        for sender, _ in pairs
    ]
    start = time.perf_counter()
    for t in senders:
        t.start()
    for t in senders + readers:
        t.join(20)
    elapsed = time.perf_counter() - start
    for sender, receiver in pairs:
        sender.close()
        receiver.close()
    floor = 0.9 * 2 * per_thread / rate
    assert elapsed >= floor, f"two threads took {elapsed:.3f}s, floor {floor:.3f}s"


# ---- through the transport -------------------------------------------------------

PAYLOAD = 4 * 1024 * 1024  # bytes


def _timed_send(pg) -> float:
    payload = torch.zeros(PAYLOAD // 4)
    if pg.rank == 1:
        pg.recv(payload, src=0)
        return 0.0
    start = time.perf_counter()
    pg.send(payload, dst=1)
    return time.perf_counter() - start


def test_a_shaped_send_takes_at_least_bytes_over_rate() -> None:
    link = LinkModel(mbps=200)  # 25 MB/s
    elapsed = run(_timed_send, 2, link=link, n_channels=1)[0]
    floor = 0.9 * PAYLOAD / _bytes_per_s(link.mbps)
    assert elapsed >= floor, f"{elapsed:.3f}s < {floor:.3f}s"


def test_a_flow_cap_alone_shapes_a_single_connection() -> None:
    link = LinkModel(flow_mbps=200)
    elapsed = run(_timed_send, 2, link=link, n_channels=1)[0]
    floor = 0.9 * PAYLOAD / _bytes_per_s(link.flow_mbps)
    assert elapsed >= floor, f"{elapsed:.3f}s < {floor:.3f}s"


def _send_on_every_channel(pg) -> float:
    """Rank 0 sends PAYLOAD on each channel at once."""
    payloads = [torch.zeros(PAYLOAD // 4) for _ in range(pg.n_channels)]
    if pg.rank == 1:
        pg.run_per_channel(
            [lambda c=c: pg.recv(payloads[c], src=0, channel=c) for c in range(pg.n_channels)]
        )
        return 0.0
    start = time.perf_counter()
    pg.run_per_channel(
        [lambda c=c: pg.send(payloads[c], dst=1, channel=c) for c in range(pg.n_channels)]
    )
    return time.perf_counter() - start


def test_the_rank_cap_is_shared_by_every_channel() -> None:
    """One NIC per host: two channels must not get twice the bandwidth.

    If each connection had its own bucket, two channels would finish in the
    time one takes, and so would rank 0 of the naive algorithm pushing W-1
    copies of the result, hiding exactly the bottleneck the emulator exists to
    show.
    """
    link = LinkModel(mbps=200)
    elapsed = run(_send_on_every_channel, 2, link=link, n_channels=2)[0]
    floor = 0.9 * 2 * PAYLOAD / _bytes_per_s(link.mbps)
    assert elapsed >= floor, f"two channels took {elapsed:.3f}s, floor {floor:.3f}s"


def _send_to_every_peer(pg) -> float:
    """Rank 0 sends PAYLOAD to each other rank at once."""
    payload = torch.zeros(PAYLOAD // 4)
    if pg.rank != 0:
        pg.recv(payload, src=0)
        return 0.0
    threads = [
        threading.Thread(target=pg.send, args=(payload, peer)) for peer in range(1, pg.world_size)
    ]
    start = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return time.perf_counter() - start


def test_the_rank_cap_is_shared_by_every_peer() -> None:
    link = LinkModel(mbps=200)
    elapsed = run(_send_to_every_peer, 3, link=link, n_channels=1)[0]
    floor = 0.9 * 2 * PAYLOAD / _bytes_per_s(link.mbps)
    assert elapsed >= floor, f"two peers took {elapsed:.3f}s, floor {floor:.3f}s"


def _incast(pg) -> float:
    """Every other rank sends PAYLOAD to rank 0 at once."""
    if pg.rank != 0:
        pg.send(torch.zeros(PAYLOAD // 4), dst=0)
        return 0.0
    buffers = {peer: torch.empty(PAYLOAD // 4) for peer in range(1, pg.world_size)}
    threads = [
        threading.Thread(target=pg.recv, args=(buffers[peer], peer)) for peer in buffers
    ]
    start = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return time.perf_counter() - start


def test_the_rank_cap_meters_what_arrives() -> None:
    """Two senders at full rate into one rank must queue on its ingress.

    Shaping only the send side would let rank 0 of the naive algorithm take in
    W-1 tensors at W-1 times its link rate, hiding half of its bottleneck.
    """
    link = LinkModel(mbps=200)
    elapsed = run(_incast, 3, link=link, n_channels=1)[0]
    floor = 0.9 * 2 * PAYLOAD / _bytes_per_s(link.mbps)
    assert elapsed >= floor, f"incast of two took {elapsed:.3f}s, floor {floor:.3f}s"


def _many_small_sends(pg) -> float:
    tiny = torch.zeros(1)
    start = time.perf_counter()
    for _ in range(10):
        if pg.rank == 0:
            pg.send(tiny, dst=1)
        else:
            pg.recv(tiny, src=0)
    return time.perf_counter() - start


def test_latency_is_paid_once_per_send() -> None:
    link = LinkModel(latency_us=20_000)
    elapsed = run(_many_small_sends, 2, link=link, n_channels=1)[0]
    assert elapsed >= 0.9 * 10 * 0.020, f"10 sends took {elapsed:.3f}s"


def _unshaped(pg) -> tuple[bool, str]:
    conn = pg._mesh.conns[1 - pg.rank][0]
    return pg.link.shaped, type(conn).__name__


def test_no_link_means_the_plain_connection(monkeypatch) -> None:
    """Off must mean off: not a shaper with no limits, the original code path."""
    for name in ("MINI_NCCL_LINK_MBPS", "MINI_NCCL_LINK_FLOW_MBPS", "MINI_NCCL_LINK_LATENCY_US"):
        monkeypatch.delenv(name, raising=False)
    assert run(_unshaped, 2) == [(False, "Connection")] * 2


def _bytes_per_s(mbps: float | None) -> float:
    assert mbps is not None
    return mbps * MB / 8
