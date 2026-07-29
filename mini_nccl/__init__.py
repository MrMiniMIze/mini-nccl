"""mini-nccl: collective communication from first principles.

A small, readable reimplementation of the algorithms inside libraries like
NCCL (ring, binomial-tree, and recursive halving-doubling all-reduce,
reduce-scatter, all-gather, broadcast, all-to-all) over plain TCP sockets,
plus a bucketed-overlap DDP wrapper built on nothing but these primitives.

Typical usage inside a worker process::

    import mini_nccl as mn

    pg = mn.init_process_group(rank, world_size)
    mn.all_reduce(tensor)                 # sum, algorithm chosen by size
    mn.broadcast(tensor, src=0)
    mn.destroy_process_group()

or spawn everything locally::

    mn.run(worker_fn, world_size=4)
"""

from __future__ import annotations

import os

import torch

from . import collectives as _c
from .collectives import ALGORITHMS
from .ddp import DistributedDataParallel
from .errors import (
    CollectiveTimeoutError,
    MiniNcclError,
    PeerClosedError,
    RendezvousError,
)
from .launcher import run
from .process_group import DEFAULT_N_CHANNELS, ProcessGroup
from .recorder import Recorder

__all__ = [
    "ALGORITHMS",
    "DEFAULT_N_CHANNELS",
    "CollectiveTimeoutError",
    "DistributedDataParallel",
    "MiniNcclError",
    "PeerClosedError",
    "ProcessGroup",
    "Recorder",
    "RendezvousError",
    "all_gather",
    "all_reduce",
    "all_to_all",
    "barrier",
    "broadcast",
    "destroy_process_group",
    "get_group",
    "init_process_group",
    "reduce_scatter",
    "run",
]

__version__ = "0.2.0"

_group: ProcessGroup | None = None


def _hosts_from_env() -> list[tuple[str, int]] | None:
    """Parse ``MINI_NCCL_HOSTS`` ("host:port,host:port,...") if present.

    This is the multi-node entry point: give every rank the same host list
    and the mesh forms across machines with no other changes.
    """
    raw = os.environ.get("MINI_NCCL_HOSTS")
    if not raw:
        return None
    addrs = []
    for entry in raw.split(","):
        entry = entry.strip()
        if not entry:
            continue
        host, _, port = entry.rpartition(":")
        if not host:
            raise ValueError(f"MINI_NCCL_HOSTS entry {entry!r} must be host:port")
        addrs.append((host, int(port)))
    return addrs


def init_process_group(
    rank: int | None = None,
    world_size: int | None = None,
    addrs: list[tuple[str, int]] | None = None,
    base_port: int | None = None,
    n_channels: int | None = None,
    **kwargs,
) -> ProcessGroup:
    """Create the default process group.

    ``rank`` / ``world_size`` fall back to the ``RANK`` / ``WORLD_SIZE``
    environment variables, ``base_port`` to ``MASTER_PORT``, ``n_channels``
    to ``MINI_NCCL_CHANNELS``, and ``addrs`` to ``MINI_NCCL_HOSTS``, so
    workers launched by an external runner need no arguments.
    """
    global _group
    if _group is not None:
        raise RuntimeError("default process group already initialized")
    if rank is None:
        rank = int(os.environ["RANK"])
    if world_size is None:
        world_size = int(os.environ["WORLD_SIZE"])
    if base_port is None:
        base_port = int(os.environ.get("MASTER_PORT", "29500"))
    if n_channels is None:
        n_channels = int(os.environ.get("MINI_NCCL_CHANNELS", DEFAULT_N_CHANNELS))
    if addrs is None:
        addrs = _hosts_from_env()
    _group = ProcessGroup(
        rank,
        world_size,
        addrs=addrs,
        base_port=base_port,
        n_channels=n_channels,
        **kwargs,
    )
    return _group


def get_group() -> ProcessGroup:
    if _group is None:
        raise RuntimeError("call init_process_group() first")
    return _group


def destroy_process_group() -> None:
    global _group
    if _group is not None:
        _group.close()
        _group = None


def all_reduce(tensor: torch.Tensor, op: str = "sum", algorithm: str = "auto") -> torch.Tensor:
    return _c.all_reduce(get_group(), tensor, op=op, algorithm=algorithm)


def broadcast(tensor: torch.Tensor, src: int = 0) -> torch.Tensor:
    return _c.broadcast(get_group(), tensor, src=src)


def all_gather(tensor: torch.Tensor) -> list[torch.Tensor]:
    return _c.all_gather(get_group(), tensor)


def reduce_scatter(tensor: torch.Tensor, op: str = "sum") -> torch.Tensor:
    return _c.reduce_scatter(get_group(), tensor, op=op)


def all_to_all(tensor: torch.Tensor) -> torch.Tensor:
    return _c.all_to_all(get_group(), tensor)


def barrier() -> None:
    _c.barrier(get_group())
