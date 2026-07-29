"""Process group: point-to-point primitives the collectives are built from.

A ``ProcessGroup`` is intentionally not thread-safe: collectives must be
issued in the same order on every rank (as with NCCL communicators), so
callers (including the DDP reducer) serialize onto a single thread.
The one internal thread is a send worker, which lets ``send_recv`` push to
one neighbor while the calling thread blocks on the receive from the other;
that full-duplex step is what makes ring algorithms run at line rate.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import torch

from .transport import Mesh, byte_view

DEFAULT_BASE_PORT = 29500


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
    ) -> None:
        if not 0 <= rank < world_size:
            raise ValueError(f"rank {rank} out of range for world_size {world_size}")
        self.rank = rank
        self.world_size = world_size
        if addrs is None:
            addrs = default_addrs(world_size, base_port)
        if len(addrs) != world_size:
            raise ValueError("need one (host, port) address per rank")
        self._mesh = Mesh(rank, world_size, addrs, timeout)
        self._send_pool = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix=f"mn-send-r{rank}"
        )

    def send(self, tensor: torch.Tensor, dst: int) -> None:
        self._mesh.conns[dst].send(byte_view(tensor))

    def recv(self, tensor: torch.Tensor, src: int) -> None:
        self._mesh.conns[src].recv_into(byte_view(tensor))

    def send_recv(
        self,
        send_tensor: torch.Tensor,
        dst: int,
        recv_tensor: torch.Tensor,
        src: int,
    ) -> None:
        """Simultaneously send to ``dst`` and receive from ``src``.

        Sending on a worker thread while this thread receives avoids the
        deadlock where every rank in a ring blocks on ``send`` at once.
        (CPython releases the GIL inside socket syscalls, so the two
        directions genuinely overlap.)
        """
        fut = self._send_pool.submit(self.send, send_tensor, dst)
        self.recv(recv_tensor, src)
        fut.result()

    def close(self) -> None:
        self._send_pool.shutdown(wait=True)
        self._mesh.close()

    def __enter__(self) -> "ProcessGroup":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
