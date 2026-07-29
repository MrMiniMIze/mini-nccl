"""TCP transport: a full mesh of persistent connections between ranks.

Design notes:

- Rank ``r`` listens on ``addrs[r]``; every rank dials all lower-numbered
  ranks, so each pair of ranks shares exactly one connection. A 4-byte
  handshake identifies the dialing peer.
- Tensors travel as raw bytes with no per-message framing. Both ends of
  every exchange inside a collective already agree on element count and
  dtype (the same contract NCCL uses), so headers would be pure overhead.
- Receives land directly in the destination tensor's storage via
  ``socket.recv_into`` on a ``memoryview`` — the hot path does no
  Python-side buffer copies.
"""

from __future__ import annotations

import socket
import struct
import threading
import time

import torch

_HANDSHAKE = struct.Struct("!I")

# 2 MiB socket buffers: large enough that a ring step's chunk usually fits
# in flight without stalling the sender on loopback or a LAN.
_SOCK_BUF_BYTES = 1 << 21


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
    """A single peer-to-peer TCP link."""

    def __init__(self, sock: socket.socket) -> None:
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, _SOCK_BUF_BYTES)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, _SOCK_BUF_BYTES)
        sock.settimeout(None)
        self._sock = sock

    def send(self, view: memoryview) -> None:
        self._sock.sendall(view)

    def recv_into(self, view: memoryview) -> None:
        offset = 0
        remaining = len(view)
        while remaining:
            n = self._sock.recv_into(view[offset:], remaining)
            if n == 0:
                raise ConnectionError("peer closed connection mid-message")
            offset += n
            remaining -= n

    def close(self) -> None:
        try:
            self._sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        self._sock.close()


class Mesh:
    """Establishes and owns the full connection mesh for one rank."""

    def __init__(
        self,
        rank: int,
        world_size: int,
        addrs: list[tuple[str, int]],
        timeout: float = 60.0,
    ) -> None:
        self.rank = rank
        self.world_size = world_size
        self.conns: dict[int, Connection] = {}
        self._lock = threading.Lock()
        if world_size == 1:
            return

        host, port = addrs[rank]
        listener = socket.create_server((host, port), backlog=world_size)
        n_incoming = world_size - 1 - rank
        accepter = threading.Thread(
            target=self._accept_peers, args=(listener, n_incoming), daemon=True
        )
        accepter.start()

        for peer in range(rank):
            self._dial(peer, addrs[peer], timeout)

        accepter.join(timeout)
        if len(self.conns) != world_size - 1:
            missing = sorted(set(range(world_size)) - {rank} - set(self.conns))
            raise TimeoutError(f"rank {rank}: no connection to ranks {missing}")

    def _accept_peers(self, listener: socket.socket, count: int) -> None:
        for _ in range(count):
            sock, _ = listener.accept()
            conn = Connection(sock)
            raw = bytearray(_HANDSHAKE.size)
            conn.recv_into(memoryview(raw))
            (peer,) = _HANDSHAKE.unpack(raw)
            with self._lock:
                self.conns[peer] = conn
        listener.close()

    def _dial(self, peer: int, addr: tuple[str, int], timeout: float) -> None:
        deadline = time.monotonic() + timeout
        while True:
            try:
                sock = socket.create_connection(addr, timeout=5.0)
                break
            except OSError:
                if time.monotonic() > deadline:
                    raise TimeoutError(f"rank {self.rank}: could not reach rank {peer} at {addr}")
                time.sleep(0.05)
        conn = Connection(sock)
        conn.send(memoryview(_HANDSHAKE.pack(self.rank)))
        with self._lock:
            self.conns[peer] = conn

    def close(self) -> None:
        for conn in self.conns.values():
            conn.close()
        self.conns.clear()
