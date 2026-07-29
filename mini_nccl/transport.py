"""TCP transport: a full mesh of persistent connections between ranks.

Design notes:

- Rank ``r`` listens on ``addrs[r]``; every rank dials all lower-numbered
  ranks, so each pair of ranks shares exactly ``n_channels`` connections.
  An 8-byte handshake identifies the dialing peer and which channel the
  socket belongs to.
- **Channels** are independent connections between the same pair of ranks,
  the same idea NCCL uses to drive one collective over several parallel
  paths. A large collective is split across channels and each channel is
  driven by its own thread, so one collective can keep several sockets
  (and several cores) busy at once.
- Tensors travel as raw bytes with no per-message framing. Both ends of
  every exchange inside a collective already agree on element count and
  dtype (the same contract NCCL uses), so headers would be pure overhead.
- Receives land directly in the destination tensor's storage via
  ``socket.recv_into`` on a ``memoryview``; the hot path does no
  Python-side buffer copies.
- Every socket carries an operation timeout. Blocking forever on a dead or
  desynchronized peer is the failure mode that makes real distributed jobs
  undebuggable, so the default is a bounded wait that raises.
"""

from __future__ import annotations

import contextlib
import socket
import struct
import threading
import time

import torch

from .errors import PeerClosedError, RendezvousError

_HANDSHAKE = struct.Struct("!II")

# 2 MiB socket buffers: large enough that a ring step's chunk usually fits
# in flight without stalling the sender on loopback or a LAN.
_SOCK_BUF_BYTES = 1 << 21

DEFAULT_OP_TIMEOUT = 300.0


def byte_view(tensor: torch.Tensor) -> memoryview:
    """Reinterpret a tensor's storage as a flat, writable uint8 memoryview.

    The tensor must be contiguous and on the CPU. No data is copied.
    """
    if not tensor.is_contiguous():
        raise ValueError("transport requires contiguous tensors")
    if tensor.device.type != "cpu":
        raise ValueError("transport requires CPU tensors (stage device buffers first)")
    return tensor.detach().view(-1).view(torch.uint8).numpy().data


class Connection:
    """A single peer-to-peer TCP link (one channel to one peer)."""

    def __init__(self, sock: socket.socket, op_timeout: float = DEFAULT_OP_TIMEOUT) -> None:
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, _SOCK_BUF_BYTES)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, _SOCK_BUF_BYTES)
        sock.settimeout(op_timeout)
        self._sock = sock

    def set_timeout(self, op_timeout: float | None) -> None:
        self._sock.settimeout(op_timeout)

    def send(self, view: memoryview) -> None:
        self._sock.sendall(view)

    def recv_into(self, view: memoryview) -> None:
        offset = 0
        remaining = len(view)
        while remaining:
            n = self._sock.recv_into(view[offset:], remaining)
            if n == 0:
                raise PeerClosedError("peer closed connection mid-message")
            offset += n
            remaining -= n

    def close(self) -> None:
        with contextlib.suppress(OSError):  # already closed by the peer
            self._sock.shutdown(socket.SHUT_RDWR)
        self._sock.close()


class Mesh:
    """Establishes and owns the full connection mesh for one rank.

    ``conns[peer][channel]`` is the link to ``peer`` on ``channel``.
    """

    def __init__(
        self,
        rank: int,
        world_size: int,
        addrs: list[tuple[str, int]],
        n_channels: int = 1,
        timeout: float = 60.0,
        op_timeout: float = DEFAULT_OP_TIMEOUT,
    ) -> None:
        self.rank = rank
        self.world_size = world_size
        self.n_channels = n_channels
        self.conns: dict[int, list[Connection]] = {}
        self._partial: dict[tuple[int, int], Connection] = {}
        self._lock = threading.Lock()
        self._op_timeout = op_timeout
        if world_size == 1:
            return

        host, port = addrs[rank]
        listener = socket.create_server((host, port), backlog=world_size * n_channels)
        n_incoming = (world_size - 1 - rank) * n_channels
        accepter = threading.Thread(
            target=self._accept_peers, args=(listener, n_incoming), daemon=True
        )
        accepter.start()

        for peer in range(rank):
            for channel in range(n_channels):
                self._dial(peer, channel, addrs[peer], timeout)

        accepter.join(timeout)
        expected = (world_size - 1) * n_channels
        with self._lock:
            got = len(self._partial)
            if got != expected:
                missing = sorted(
                    {p for p in range(world_size) if p != rank}
                    - {p for p, _ in self._partial}
                )
                raise RendezvousError(
                    f"rank {rank}: established {got}/{expected} connections; "
                    f"no contact with ranks {missing}"
                )
            for peer in range(world_size):
                if peer == rank:
                    continue
                self.conns[peer] = [self._partial[(peer, c)] for c in range(n_channels)]
            self._partial.clear()

    def _accept_peers(self, listener: socket.socket, count: int) -> None:
        for _ in range(count):
            sock, _ = listener.accept()
            conn = Connection(sock, self._op_timeout)
            raw = bytearray(_HANDSHAKE.size)
            conn.recv_into(memoryview(raw))
            peer, channel = _HANDSHAKE.unpack(raw)
            with self._lock:
                self._partial[(peer, channel)] = conn
        listener.close()

    def _dial(self, peer: int, channel: int, addr: tuple[str, int], timeout: float) -> None:
        deadline = time.monotonic() + timeout
        while True:
            try:
                sock = socket.create_connection(addr, timeout=5.0)
                break
            except OSError:
                if time.monotonic() > deadline:
                    raise RendezvousError(
                        f"rank {self.rank}: could not reach rank {peer} at {addr}"
                    ) from None
                time.sleep(0.05)
        conn = Connection(sock, self._op_timeout)
        conn.send(memoryview(_HANDSHAKE.pack(self.rank, channel)))
        with self._lock:
            self._partial[(peer, channel)] = conn

    def set_op_timeout(self, op_timeout: float | None) -> None:
        for channels in self.conns.values():
            for conn in channels:
                conn.set_timeout(op_timeout)

    def close(self) -> None:
        for channels in self.conns.values():
            for conn in channels:
                conn.close()
        self.conns.clear()
