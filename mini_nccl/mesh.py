"""Sub-group communicators, and factoring ranks into a parallelism mesh.

Each parallelism strategy in this library works on its own. Real training
stacks compose them: tensor parallel inside a node where the links are fast,
data parallel across nodes where they are not, pipeline stages spanning both.
Doing that needs communicators over *subsets* of ranks, so a tensor-parallel
all-reduce touches only the ranks sharing that layer and a data-parallel
all-reduce touches only the corresponding ranks of each replica.

A :class:`SubGroup` is a **view**, not a new connection mesh. The parent group
already holds a socket to every peer, so a subgroup only needs to translate its
local rank numbering onto the parent's and reuse those sockets. Nothing is
reconnected, and no new threads are started.

That works because every collective in this library is written against a small
interface: ``rank``, ``world_size``, ``send``, ``recv``, ``send_recv``,
``send_recv_sliced``, ``run_per_channel``, ``recorder``, ``n_channels``. A
subgroup implements all of it with translation, so ring all-reduce, FSDP,
tensor parallel, and the pipeline schedules all run on a subgroup unchanged.

Ordering still matters. Collectives must be issued in the same order on every
rank *of a given group*, and two groups that share a socket must not interleave.
The mesh below sidesteps that entirely: the groups along different dimensions
are orthogonal partitions, so two ranks that talk along the tensor dimension
never talk along the data dimension, and no socket is shared between them.

Usage::

    mesh = ParallelMesh(pg, dp=2, tp=2)      # 4 ranks as a 2x2 grid
    tp_group = mesh.group("tp")              # this rank's tensor-parallel peers
    dp_group = mesh.group("dp")              # its data-parallel peers

    mlp = ParallelMLP(width, tp_group)        # layer split across tp
    model = DistributedDataParallel(model, dp_group)   # gradients reduced across dp

Ranks are laid out with the **last dimension fastest**, so with ``dp=2, tp=2``
the tensor-parallel groups are ``[0, 1]`` and ``[2, 3]``. Contiguous ranks land
in the same tensor-parallel group, which is what you want when neighbouring
ranks share the faster interconnect.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import torch

from .process_group import ProcessGroup


class SubGroup:
    """A communicator over a subset of a parent group's ranks."""

    def __init__(
        self,
        parent: ProcessGroup | SubGroup,
        global_ranks: Sequence[int],
        name: str = "",
    ) -> None:
        ranks = list(global_ranks)
        if len(set(ranks)) != len(ranks):
            raise ValueError(f"duplicate ranks in subgroup: {ranks}")
        if parent.rank not in ranks:
            raise ValueError(
                f"rank {parent.rank} is not a member of subgroup {ranks}; "
                f"build it only on member ranks"
            )
        self.parent = parent
        self.name = name
        self.global_ranks = ranks
        self.rank = ranks.index(parent.rank)
        self.world_size = len(ranks)
        self.n_channels = parent.n_channels
        self.recorder = parent.recorder
        self.op_timeout = parent.op_timeout

    def _global(self, local_rank: int) -> int:
        return self.global_ranks[local_rank]

    # ---- the interface every collective is written against ----------------

    def send(self, tensor: torch.Tensor, dst: int, channel: int = 0) -> None:
        self.parent.send(tensor, self._global(dst), channel)

    def recv(self, tensor: torch.Tensor, src: int, channel: int = 0) -> None:
        self.parent.recv(tensor, self._global(src), channel)

    def send_recv(
        self,
        send_tensor: torch.Tensor,
        dst: int,
        recv_tensor: torch.Tensor,
        src: int,
        channel: int = 0,
    ) -> None:
        self.parent.send_recv(
            send_tensor, self._global(dst), recv_tensor, self._global(src), channel
        )

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
        self.parent.send_recv_sliced(
            send_tensor,
            self._global(dst),
            recv_tensor,
            self._global(src),
            n_slices,
            on_slice,
            channel,
        )

    def run_per_channel(self, fns: list[Callable[[], None]]) -> None:
        self.parent.run_per_channel(fns)

    def subgroup(self, global_ranks: Sequence[int], name: str = "") -> SubGroup:
        """Nest another subgroup, still resolving against the same sockets."""
        return SubGroup(self, global_ranks, name)

    def __repr__(self) -> str:
        label = f" {self.name!r}" if self.name else ""
        return (
            f"<SubGroup{label} rank {self.rank}/{self.world_size} "
            f"global={self.global_ranks}>"
        )


class ParallelMesh:
    """Factor a process group's ranks into named parallelism dimensions.

    ``ParallelMesh(pg, dp=2, tp=2)`` treats 4 ranks as a 2x2 grid. The product
    of the dimensions must equal the world size. Dimensions are listed outermost
    first, and the last one varies fastest across consecutive ranks.
    """

    def __init__(self, pg: ProcessGroup, **dims: int) -> None:
        if not dims:
            raise ValueError("give at least one dimension, e.g. ParallelMesh(pg, tp=2)")
        for name, size in dims.items():
            if size < 1:
                raise ValueError(f"dimension {name}={size} must be at least 1")
        total = 1
        for size in dims.values():
            total *= size
        if total != pg.world_size:
            raise ValueError(
                f"mesh {dims} has {total} slots but the group has {pg.world_size} ranks"
            )

        self.pg = pg
        self.dims = dict(dims)
        self.names = list(dims)
        # Strides, last dimension fastest: with dp=2, tp=2 the tensor-parallel
        # groups are contiguous rank runs, which suits the faster interconnect.
        self.strides: dict[str, int] = {}
        stride = 1
        for name in reversed(self.names):
            self.strides[name] = stride
            stride *= self.dims[name]
        self.coords = {
            name: (pg.rank // self.strides[name]) % self.dims[name] for name in self.names
        }
        self._cache: dict[str, SubGroup] = {}

    def ranks_along(self, name: str) -> list[int]:
        """Global ranks that differ from this one only in dimension ``name``."""
        if name not in self.dims:
            raise KeyError(f"no dimension {name!r}; mesh has {self.names}")
        base = self.pg.rank - self.coords[name] * self.strides[name]
        return [base + i * self.strides[name] for i in range(self.dims[name])]

    def group(self, name: str) -> SubGroup | ProcessGroup:
        """This rank's communicator along dimension ``name``.

        A dimension of size 1 returns a single-rank subgroup, so callers do not
        need to special-case a degenerate axis: every collective is a no-op on
        a group of one.
        """
        if name not in self._cache:
            self._cache[name] = SubGroup(self.pg, self.ranks_along(name), name)
        return self._cache[name]

    def coordinate(self, name: str) -> int:
        return self.coords[name]

    def size(self, name: str) -> int:
        return self.dims[name]

    def describe(self) -> str:
        parts = [f"{n}={self.coords[n]}/{self.dims[n]}" for n in self.names]
        return f"rank {self.pg.rank}: " + " ".join(parts)

    def __repr__(self) -> str:
        return f"<ParallelMesh {self.dims} rank {self.pg.rank}>"
