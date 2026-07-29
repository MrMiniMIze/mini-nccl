"""mini-nccl: collective communication from first principles.

A small, readable reimplementation of the algorithms inside libraries like
NCCL (ring and binomial-tree all-reduce, reduce-scatter, all-gather,
broadcast) over plain TCP sockets, plus a bucketed-overlap DDP wrapper
built on nothing but these primitives.

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
from .ddp import DistributedDataParallel
from .launcher import run
from .process_group import ProcessGroup

__all__ = [
    "ProcessGroup",
    "DistributedDataParallel",
    "init_process_group",
    "destroy_process_group",
    "get_group",
    "all_reduce",
    "broadcast",
    "all_gather",
    "reduce_scatter",
    "barrier",
    "run",
]

__version__ = "0.1.0"

_group: ProcessGroup | None = None


def init_process_group(
    rank: int | None = None,
    world_size: int | None = None,
    addrs: list[tuple[str, int]] | None = None,
    base_port: int | None = None,
) -> ProcessGroup:
    """Create the default process group.

    ``rank`` / ``world_size`` fall back to the ``RANK`` / ``WORLD_SIZE``
    environment variables (and ``base_port`` to ``MASTER_PORT``) so workers
    launched by an external runner need no arguments.
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
    _group = ProcessGroup(rank, world_size, addrs=addrs, base_port=base_port)
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


def barrier() -> None:
    _c.barrier(get_group())
