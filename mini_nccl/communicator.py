"""The interface every collective is written against.

Nothing in this library's algorithms knows what a process group *is*. Ring
all-reduce, FSDP, tensor parallelism, and the pipeline schedules only ever
touch the members below. Stating that as a :class:`Protocol` turns an informal
claim into a checked one: :class:`~mini_nccl.process_group.ProcessGroup` and
:class:`~mini_nccl.mesh.SubGroup` are both required to satisfy it, so a
subgroup really is substitutable for the group it was carved out of, and a type
checker will say so if that ever stops being true.

That substitutability is the whole reason composing the parallelism strategies
needed no changes to any of them. Keeping the surface this small is what made
it possible, so adding a member here is a decision worth resisting: every
addition is a new thing a communicator has to provide.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol, runtime_checkable

import torch

from .recorder import Recorder


@runtime_checkable
class Communicator(Protocol):
    """What a collective needs from whatever it is communicating over."""

    #: This participant's index within the group, ``0 .. world_size-1``.
    rank: int
    #: Number of participants.
    world_size: int
    #: Independent connections available per peer.
    n_channels: int
    #: Where collectives log themselves for tracing and diagnosis.
    recorder: Recorder

    def send(self, tensor: torch.Tensor, dst: int, channel: int = 0) -> None: ...

    def recv(self, tensor: torch.Tensor, src: int, channel: int = 0) -> None: ...

    def send_recv(
        self,
        send_tensor: torch.Tensor,
        dst: int,
        recv_tensor: torch.Tensor,
        src: int,
        channel: int = 0,
    ) -> None:
        """Send and receive at once, so ring steps cannot deadlock."""
        ...

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
        """As ``send_recv``, but hand back the receive in pieces as they land."""
        ...

    def run_per_channel(self, fns: list[Callable[[], None]]) -> None:
        """Run one callable per channel concurrently, re-raising failures."""
        ...
