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
- An optional **emulated link** (:class:`LinkModel`) rate-limits and delays
  every send. Loopback "networking" is a memory copy done by the same cores
  that do the reduction, which hides exactly the costs the algorithms exist to
  manage; a shaped link makes bytes cost time while leaving the CPU idle, the
  way a NIC does. See docs/emulated-link.md.
"""

from __future__ import annotations

import contextlib
import hashlib
import os
import socket
import struct
import threading
import time
from dataclasses import dataclass

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
# has not delivered all of it within this budget is not a peer, and must not
# hold up the others. Strays are admitted one at a time, so each one can still
# delay rendezvous by up to this much: the protection is against stray and
# stale connections, not a determined adversary, since rendezvous is not
# authenticated (nor are NCCL's or gloo's).
_HANDSHAKE_TIMEOUT = 5.0

# 2 MiB socket buffers: large enough that a ring step's chunk usually fits
# in flight without stalling the sender on loopback or a LAN.
_SOCK_BUF_BYTES = 1 << 21

DEFAULT_OP_TIMEOUT = 300.0

# How often the accepter checks whether the rendezvous has been abandoned.
_ACCEPT_POLL_SECONDS = 0.5


# ---- emulated link ----------------------------------------------------------

#: Largest piece a shaped send hands to the socket at once. Small enough that
#: concurrent senders interleave fairly, large enough that the per-chunk
#: bookkeeping is noise next to the time the chunk is charged for.
_SHAPE_CHUNK = 64 * 1024


@dataclass(frozen=True)
class LinkModel:
    """An emulated network link, applied to every send a rank makes.

    ``mbps`` caps the rank's traffic in megabits per second, in each
    direction, shared by every connection and channel it has: one full-duplex
    NIC per host. That sharing is the point. It is what makes the naive
    algorithm's bottleneck real, because rank 0 has to take in ``W-1`` tensors
    and push out ``W-1`` copies of the result through one link; a cap per
    connection would quietly hand it ``W-1`` links instead.

    ``flow_mbps`` additionally caps each connection on its own, which models a
    per-flow limit (a congestion window, a per-stream offload). Only under a
    per-flow cap is "do more channels help?" a fair question.

    ``latency_us`` is a fixed delay added once per send call: the per-message
    alpha of the alpha-beta model, not a propagation delay that pipelines.
    """

    mbps: float | None = None
    flow_mbps: float | None = None
    latency_us: float = 0.0

    def __post_init__(self) -> None:
        for name in ("mbps", "flow_mbps"):
            value = getattr(self, name)
            if value is not None and value <= 0:
                raise ValueError(f"LinkModel.{name} must be positive, got {value}")
        if self.latency_us < 0:
            raise ValueError(f"LinkModel.latency_us must be >= 0, got {self.latency_us}")

    @property
    def shaped(self) -> bool:
        return bool(self.mbps or self.flow_mbps or self.latency_us)

    @classmethod
    def from_env(cls) -> LinkModel:
        """Read ``MINI_NCCL_LINK_MBPS``, ``_FLOW_MBPS`` and ``_LATENCY_US``."""

        def number(name: str) -> float | None:
            raw = os.environ.get(name, "").strip()
            return float(raw) if raw else None

        return cls(
            mbps=number("MINI_NCCL_LINK_MBPS"),
            flow_mbps=number("MINI_NCCL_LINK_FLOW_MBPS"),
            latency_us=number("MINI_NCCL_LINK_LATENCY_US") or 0.0,
        )


# How far a later chunk of the same message may book back in time to absorb
# timer overshoot. Bounded so that a sender which stalled on a full socket
# buffer resumes at the link rate rather than in a burst.
_CATCH_UP_S = 0.002


class LinkClock:
    """When one direction of an emulated link is next free.

    Each chunk books ``nbytes / rate`` seconds, starting no earlier than the
    previous booking ends, and whoever booked it waits until it ends. Bookings
    never overlap, so the bytes through one clock cannot exceed ``rate`` over
    any interval. The guarantee is one-sided on purpose: a loaded machine can
    deliver less than the rate, never more.

    Three details carry the fidelity:

    - **No idle credit.** A message's first chunk books from "now" at the
      earliest, so an idle link does not bank time. (A token bucket with a burst
      allowance does bank it, and that erases the serialization delay of every
      message smaller than the burst, which is exactly the regime where tree
      beats ring.)
    - **No drift.** Later chunks of a message may book from where the previous
      one ended even if the sender woke late, so timer overshoot (coarse on
      Windows) is absorbed instead of accumulating, within ``_CATCH_UP_S``.
      Time is always measured, never inferred from the sleep that was asked for.
    - **No double charge.** The receive side books a chunk as having *finished*
      arriving when it arrived, so one flow already paced by its sender passes
      straight through, while several senders converging on one receiver
      (rank 0 in the naive algorithm) queue on its ingress.
    """

    def __init__(self, bytes_per_second: float) -> None:
        self.rate = float(bytes_per_second)
        self._free_at = 0.0
        self._lock = threading.Lock()

    def book(self, nbytes: int, earliest: float) -> float:
        """Book ``nbytes`` starting no earlier than ``earliest``; return its end."""
        # The lock covers arithmetic only; waiting happens outside it.
        with self._lock:
            start = max(self._free_at, earliest)
            self._free_at = start + nbytes / self.rate
            return self._free_at


def _wait_until(deadline: float) -> None:
    delay = deadline - time.perf_counter()
    if delay > 0:
        time.sleep(delay)


def _mbps_to_bytes(mbps: float) -> float:
    return mbps * 1e6 / 8


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

    def recv_by(self, view: memoryview, deadline: float) -> None:
        """Fill ``view`` before ``deadline`` (``time.monotonic()``), or raise.

        A socket timeout restarts on every read that returns data, so a peer
        trickling one byte at a time never trips it. Rendezvous reads the
        handshake through this instead, against one deadline for the whole
        message. Not used on the data path, which wants no per-read clock.
        """
        offset = 0
        while offset < len(view):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("deadline passed before the message completed")
            self._sock.settimeout(remaining)
            n = self._sock.recv_into(view[offset:], len(view) - offset)
            if n == 0:
                raise PeerClosedError("peer closed connection mid-message")
            offset += n

    def close(self) -> None:
        with contextlib.suppress(OSError):  # already closed by the peer
            self._sock.shutdown(socket.SHUT_RDWR)
        self._sock.close()


class ShapedConnection(Connection):
    """A connection whose traffic pays for an emulated link.

    Chosen once, when the connection is made, so an unshaped run executes
    exactly the plain :class:`Connection` code: the emulator costs nothing when
    it is off.
    """

    def __init__(
        self,
        sock: socket.socket,
        op_timeout: float,
        egress: tuple[LinkClock, ...],
        ingress: LinkClock | None,
        latency_s: float,
    ) -> None:
        super().__init__(sock, op_timeout)
        self._egress = egress
        self._ingress = ingress
        self._latency_s = latency_s

    def send(self, view: memoryview) -> None:
        if self._latency_s:
            time.sleep(self._latency_s)
        for offset in range(0, len(view), _SHAPE_CHUNK):
            chunk = view[offset : offset + _SHAPE_CHUNK]
            now = time.perf_counter()
            earliest = now if offset == 0 else now - _CATCH_UP_S
            if self._egress:
                _wait_until(max(clock.book(len(chunk), earliest) for clock in self._egress))
            self._sock.sendall(chunk)

    def recv_into(self, view: memoryview) -> None:
        offset = 0
        while offset < len(view):
            n = self._sock.recv_into(view[offset:], len(view) - offset)
            if n == 0:
                raise PeerClosedError("peer closed connection mid-message")
            if self._ingress is not None:
                arrived = time.perf_counter()
                _wait_until(self._ingress.book(n, arrived - n / self._ingress.rate))
            offset += n


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
        link: LinkModel | None = None,
    ) -> None:
        self.rank = rank
        self.world_size = world_size
        self.n_channels = n_channels
        self.link = link if link is not None else LinkModel()
        # One clock per direction for the whole rank: every connection and
        # channel this rank owns shares the same link, in and out.
        self._egress: LinkClock | None = None
        self._ingress: LinkClock | None = None
        if self.link.mbps:
            self._egress = LinkClock(_mbps_to_bytes(self.link.mbps))
            self._ingress = LinkClock(_mbps_to_bytes(self.link.mbps))
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
                if self._admit(self._connection(sock), origin):
                    accepted += 1
        finally:
            listener.close()

    def _admit(self, conn: Connection, origin: tuple) -> bool:
        """Check one incoming handshake, then file the connection or refuse it."""
        where = f"{origin[0]}:{origin[1]}"
        raw = bytearray(_HANDSHAKE.size)
        try:
            conn.recv_by(memoryview(raw), time.monotonic() + _HANDSHAKE_TIMEOUT)
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
        conn = self._connection(sock)
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

    def _connection(self, sock: socket.socket) -> Connection:
        if not self.link.shaped:
            return Connection(sock, self._op_timeout)
        egress: list[LinkClock] = []
        if self._egress is not None:
            egress.append(self._egress)
        if self.link.flow_mbps:
            egress.append(LinkClock(_mbps_to_bytes(self.link.flow_mbps)))
        return ShapedConnection(
            sock, self._op_timeout, tuple(egress), self._ingress, self.link.latency_us / 1e6
        )

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
