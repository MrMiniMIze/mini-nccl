"""Process group: point-to-point primitives the collectives are built from.

A process group owns one connection mesh, one send worker per channel, and a
flight recorder. Collectives must be issued in the same order on every rank
*per channel* (as with NCCL communicators), so a single thread drives each
channel.

Two primitives do the real work:

- ``send_recv`` pushes to one neighbor while the calling thread blocks on the
  receive from the other. That full-duplex step is what lets ring
  algorithms run at line rate instead of half of it.
- ``send_recv_sliced`` does the same, but hands the receive back in slices as
  they land, so the caller can reduce slice *i* while slice *i+1* is still
  in flight. That is the pipelining that keeps the CPU and the socket busy
  simultaneously.
"""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor

import torch

from .errors import CollectiveTimeoutError
from .recorder import Recorder, trace_enabled_by_env
from .transport import DEFAULT_OP_TIMEOUT, Mesh, byte_view

DEFAULT_BASE_PORT = 29500

# Two channels was the measured optimum on loopback TCP (1.45x at 16 MiB,
# while 4+ channels regressed on smaller payloads). A real NIC will usually
# want more; see benchmarks/bench_ablation.py.
DEFAULT_N_CHANNELS = 2


def default_addrs(world_size: int, base_port: int = DEFAULT_BASE_PORT) -> list[tuple[str, int]]:
    return [("127.0.0.1", base_port + r) for r in range(world_size)]


class ProcessGroup:
    def __init__(
        self,
        rank: int,
        world_size: int,
        addrs: list[tuple[str, int]] | None = None,
        base_port: int = DEFAULT_BASE_PORT,
        timeout: float = 60.0,
        n_channels: int = DEFAULT_N_CHANNELS,
        op_timeout: float = DEFAULT_OP_TIMEOUT,
        trace: bool | None = None,
    ) -> None:
        if not 0 <= rank < world_size:
            raise ValueError(f"rank {rank} out of range for world_size {world_size}")
        if n_channels < 1:
            raise ValueError("n_channels must be >= 1")
        self.rank = rank
        self.world_size = world_size
        self.n_channels = n_channels
        self.op_timeout: float | None = op_timeout
        self.recorder = Recorder(
            rank,
            world_size,
            enabled=trace_enabled_by_env() if trace is None else trace,
        )
        if addrs is None:
            addrs = default_addrs(world_size, base_port)
        if len(addrs) != world_size:
            raise ValueError("need one (host, port) address per rank")
        self._mesh = Mesh(rank, world_size, addrs, n_channels, timeout, op_timeout)
        self._send_pools = [
            ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"mn-send-r{rank}-c{c}")
            for c in range(n_channels)
        ]
        # Drives the per-channel halves of a single split collective.
        self._channel_pool = ThreadPoolExecutor(
            max_workers=n_channels, thread_name_prefix=f"mn-chan-r{rank}"
        )

    # ---- point to point --------------------------------------------------

    def send(self, tensor: torch.Tensor, dst: int, channel: int = 0) -> None:
        try:
            self._mesh.conns[dst][channel].send(byte_view(tensor))
        except TimeoutError as exc:
            raise self._timeout_error("send", dst, channel) from exc

    def recv(self, tensor: torch.Tensor, src: int, channel: int = 0) -> None:
        try:
            self._mesh.conns[src][channel].recv_into(byte_view(tensor))
        except TimeoutError as exc:
            raise self._timeout_error("recv", src, channel) from exc

    def send_recv(
        self,
        send_tensor: torch.Tensor,
        dst: int,
        recv_tensor: torch.Tensor,
        src: int,
        channel: int = 0,
    ) -> None:
        """Simultaneously send to ``dst`` and receive from ``src``.

        Sending on a worker thread while this thread receives avoids the
        deadlock where every rank in a ring blocks on ``send`` at once.
        (CPython releases the GIL inside socket syscalls, so the two
        directions genuinely overlap.)
        """
        fut = self._send_pools[channel].submit(self.send, send_tensor, dst, channel)
        try:
            self.recv(recv_tensor, src, channel)
        finally:
            fut.result()

    def send_recv_sliced(
        self,
        send_tensor: torch.Tensor,
        dst: int,
        recv_tensor: torch.Tensor,
        src: int,
        n_slices: int,
        on_slice: Callable[[int, int], None],
        channel: int = 0,
    ) -> None:
        """Full-duplex exchange that yields the receive in ``n_slices`` pieces.

        The whole payload goes out in one ``sendall`` (TCP is a stream, so
        the peer can start consuming immediately) while this thread receives
        slice by slice, calling ``on_slice(start, end)`` after each lands.
        Reducing inside that callback overlaps arithmetic with the transfer
        of the following slice.
        """
        flat = recv_tensor.view(-1)
        n = flat.numel()
        if n_slices <= 1 or n < n_slices:
            self.send_recv(send_tensor, dst, recv_tensor, src, channel)
            on_slice(0, n)
            return
        step = -(-n // n_slices)
        fut = self._send_pools[channel].submit(self.send, send_tensor, dst, channel)
        try:
            for start in range(0, n, step):
                end = min(start + step, n)
                self.recv(flat[start:end], src, channel)
                on_slice(start, end)
        finally:
            fut.result()

    def run_per_channel(self, fns: list[Callable[[], None]]) -> None:
        """Run one callable per channel concurrently and re-raise failures.

        ``fns[c]`` is expected to touch only channel ``c``, which is what
        keeps the per-channel ordering invariant intact.
        """
        if len(fns) == 1:
            fns[0]()
            return
        futures = [self._channel_pool.submit(fn) for fn in fns]
        errors = []
        for fut in futures:
            try:
                fut.result()
            except BaseException as exc:
                errors.append(exc)
        if errors:
            raise errors[0]

    # ---- diagnostics -----------------------------------------------------

    def _timeout_error(self, direction: str, peer: int, channel: int) -> CollectiveTimeoutError:
        return CollectiveTimeoutError(
            f"rank {self.rank}: {direction} on channel {channel} to/from rank {peer} "
            f"exceeded {self.op_timeout}s.\n"
            f"  local state: {self.recorder.context()}\n"
            f"  likely cause: rank {peer} is dead, stuck, or issuing a different "
            f"collective (ranks must call collectives in identical order)."
        )

    def set_op_timeout(self, op_timeout: float | None) -> None:
        self.op_timeout = op_timeout
        self._mesh.set_op_timeout(op_timeout)

    # ---- lifecycle -------------------------------------------------------

    def close(self) -> None:
        for pool in self._send_pools:
            pool.shutdown(wait=True)
        self._channel_pool.shutdown(wait=True)
        self._mesh.close()

    def __enter__(self) -> ProcessGroup:
        return self

    def __exit__(self, *exc) -> None:
        self.close()
