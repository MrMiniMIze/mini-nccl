"""TCP transport: a full mesh of persistent connections between ranks.

Design notes:

- Rank ``r`` listens on ``addrs[r]``; every rank dials all lower-numbered
  ranks, so each pair of ranks shares exactly ``n_channels`` connections.
- Every connection opens with a handshake, and nothing is trusted until it
  checks out: a magic number (is this mini-nccl at all?), a protocol version,
  the dialer's rank and channel (in range, and not already connected), and a
  fingerprint of the job's configuration (world size, channels, host list).
  The acceptor answers with one status byte, so a refused dialer can say *why*
  instead of failing later on its first collective. A connection that does not
  produce a handshake promptly is dropped, so a port scanner or a stale process
  from an earlier run cannot stall rendezvous.
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
import hashlib
import socket
import struct
import threading
import time

import torch

from .errors import PeerClosedError, RendezvousError

# magic, protocol version, dialer rank, dialer channel, job fingerprint.
_HANDSHAKE = struct.Struct("!4sHIIQ")
_MAGIC = b"MNCL"
#: Bumped whenever the wire format changes, so mismatched builds refuse each
#: other at rendezvous instead of misreading each other's bytes.
PROTOCOL_VERSION = 1

# The acceptor's one-byte answer to a handshake.
_ACCEPTED = 0
_REFUSALS = {
    1: "it speaks a different protocol version",
    2: "the job configuration differs (world size, channel count, or host list)",
    3: "the claimed rank or channel is not one this rank expects to accept",
    4: "that rank and channel are already connected",
}
_BAD_VERSION, _BAD_JOB, _BAD_IDENTITY, _DUPLICATE = 1, 2, 3, 4

# A real peer sends its handshake immediately after connecting. Anything that
# stays silent longer than this is not a peer, and must not hold up the others.
_HANDSHAKE_TIMEOUT = 5.0

# 2 MiB socket buffers: large enough that a ring step's chunk usually fits
# in flight without stalling the sender on loopback or a LAN.
_SOCK_BUF_BYTES = 1 << 21

DEFAULT_OP_TIMEOUT = 300.0

# How often the accepter checks whether the rendezvous has been abandoned.
_ACCEPT_POLL_SECONDS = 0.5


def job_fingerprint(world_size: int, n_channels: int, addrs: list[tuple[str, int]]) -> int:
    """A 64-bit digest of what every rank must agree on before connecting.

    The host list in particular has to be identical everywhere (docs/multinode.md),
    and a disagreement used to surface as ranks quietly connecting to the wrong
    peers. Hashing it into the handshake turns that into a named refusal.
    """
    canonical = repr((world_size, n_channels, [(str(h), int(p)) for h, p in addrs]))
    return int.from_bytes(hashlib.blake2b(canonical.encode(), digest_size=8).digest(), "big")


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
        self._refused: list[str] = []
        self._lock = threading.Lock()
        self._op_timeout = op_timeout
        self._stop = threading.Event()
        if world_size == 1:
            return
        self._fingerprint = job_fingerprint(world_size, n_channels, addrs)

        host, port = addrs[rank]
        listener = socket.create_server((host, port), backlog=world_size * n_channels)
        n_incoming = (world_size - 1 - rank) * n_channels
        accepter = threading.Thread(
            target=self._accept_peers, args=(listener, n_incoming), daemon=True
        )
        accepter.start()

        # A failed rendezvous has to clean up after itself. Without this, the
        # accepter thread stays parked in accept() holding the listening socket,
        # so a caller that catches RendezvousError and retries in the same
        # process hits "address already in use" on its own port, and the
        # connections that did succeed are never closed.
        try:
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
                    refused = ""
                    if self._refused:
                        refused = "; refused " + "; ".join(self._refused)
                    raise RendezvousError(
                        f"rank {rank}: established {got}/{expected} connections; "
                        f"no contact with ranks {missing}{refused}"
                    )
                for peer in range(world_size):
                    if peer == rank:
                        continue
                    self.conns[peer] = [
                        self._partial[(peer, c)] for c in range(n_channels)
                    ]
                self._partial.clear()
        except BaseException:
            self._stop.set()
            accepter.join(_ACCEPT_POLL_SECONDS * 3)
            with contextlib.suppress(OSError):
                listener.close()
            with self._lock:
                for conn in self._partial.values():
                    conn.close()
                self._partial.clear()
            raise

    def _accept_peers(self, listener: socket.socket, count: int) -> None:
        # Polled rather than blocking indefinitely, so an abandoned rendezvous
        # can tell this thread to stop and release the port.
        listener.settimeout(_ACCEPT_POLL_SECONDS)
        try:
            accepted = 0
            while accepted < count and not self._stop.is_set():
                try:
                    sock, origin = listener.accept()
                except TimeoutError:
                    continue
                if self._admit(Connection(sock, self._op_timeout), origin):
                    accepted += 1
        finally:
            listener.close()

    def _admit(self, conn: Connection, origin: tuple) -> bool:
        """Check one incoming handshake, then file the connection or refuse it."""
        where = f"{origin[0]}:{origin[1]}"
        raw = bytearray(_HANDSHAKE.size)
        conn.set_timeout(_HANDSHAKE_TIMEOUT)
        try:
            conn.recv_into(memoryview(raw))
        except OSError:
            conn.close()
            self._note(f"{where}: no handshake within {_HANDSHAKE_TIMEOUT:.0f}s")
            return False
        magic, version, peer, channel, fingerprint = _HANDSHAKE.unpack(raw)
        if magic != _MAGIC:
            # Not mini-nccl at all, so there is nobody to explain anything to.
            conn.close()
            self._note(f"{where}: not a mini-nccl peer")
            return False

        with self._lock:
            if version != PROTOCOL_VERSION:
                status = _BAD_VERSION
            elif fingerprint != self._fingerprint:
                status = _BAD_JOB
            elif not (self.rank < peer < self.world_size and 0 <= channel < self.n_channels):
                status = _BAD_IDENTITY
            elif (peer, channel) in self._partial:
                status = _DUPLICATE
            else:
                status = _ACCEPTED
                self._partial[(peer, channel)] = conn
        try:
            conn.send(memoryview(bytes([status])))
        except OSError:
            status = status or -1  # an accepted dialer that vanished mid-reply
        if status == _ACCEPTED:
            conn.set_timeout(self._op_timeout)
            return True

        with self._lock:
            if self._partial.get((peer, channel)) is conn:
                del self._partial[(peer, channel)]
        conn.close()
        reason = _REFUSALS.get(status, "it disconnected during the handshake")
        if status == _BAD_VERSION:
            reason += f" (protocol version {version}, expected {PROTOCOL_VERSION})"
        self._note(f"{where} claiming rank {peer} channel {channel}: {reason}")
        return False

    def _note(self, refusal: str) -> None:
        with self._lock:
            self._refused.append(refusal)

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
        try:
            hello = _HANDSHAKE.pack(
                _MAGIC, PROTOCOL_VERSION, self.rank, channel, self._fingerprint
            )
            conn.send(memoryview(hello))
            # Wait for the verdict. A slow acceptor may be busy dropping strays,
            # so allow whatever remains of the rendezvous budget.
            conn.set_timeout(max(deadline - time.monotonic(), _HANDSHAKE_TIMEOUT))
            status = bytearray(1)
            conn.recv_into(memoryview(status))
        except OSError as exc:
            conn.close()
            raise RendezvousError(
                f"rank {self.rank}: rank {peer} at {addr} did not complete the "
                f"handshake ({exc})"
            ) from None
        if status[0] != _ACCEPTED:
            conn.close()
            reason = _REFUSALS.get(status[0], f"unknown status {status[0]}")
            raise RendezvousError(
                f"rank {self.rank}: rank {peer} at {addr} refused the connection: {reason}"
            )
        conn.set_timeout(self._op_timeout)
        with self._lock:
            self._partial[(peer, channel)] = conn

    def set_op_timeout(self, op_timeout: float | None) -> None:
        for channels in self.conns.values():
            for conn in channels:
                conn.set_timeout(op_timeout)

    def close(self) -> None:
        self._stop.set()
        for channels in self.conns.values():
            for conn in channels:
                conn.close()
        self.conns.clear()
