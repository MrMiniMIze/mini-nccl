"""Pipeline parallelism: the model split by depth across ranks.

Tensor parallelism splits a layer across ranks; pipeline parallelism splits the
*stack*. Rank ``s`` owns a contiguous slice of the layers, activations flow
forward ``0 -> W-1``, and gradients flow back ``W-1 -> 0``. Only two tensors
ever cross a rank boundary: one activation forward, one activation gradient
back. That makes it the cheapest form of model parallelism in bytes moved, and
the most awkward in scheduling, because a naive version leaves most ranks idle
most of the time.

**The bubble.** Run one batch straight through and rank 3 cannot start until
ranks 0-2 have finished, so utilization is `1/W`. The fix is to split the batch
into ``M`` microbatches and keep several in flight, which shrinks the idle
fraction to ``(W-1)/(M+W-1)``.

**GPipe** runs all ``M`` forwards, then all ``M`` backwards. Simple, but every
microbatch's activations must be kept alive until its backward, so stage
activation memory grows with ``M``.

**1F1B** (PipeDream-Flush, and what Megatron uses) reaches the same bubble with
far less memory. Each stage warms up with ``W-1-s`` forwards, then alternates
one forward and one backward, then drains. Stage ``s`` holds at most ``W-s``
microbatches instead of ``M``, so memory stops depending on how many
microbatches you chose. ``in_flight_peak`` reports the measured depth, and the
tests assert the bound.

**Deadlock.** A stage sending an activation forward while its neighbor sends a
gradient back is a real hazard once activations exceed the socket buffer: both
block, neither drains. The steady state therefore uses one *fused* exchange
(``send_recv``: send on a worker thread while receiving on this one), which is
the same fix Megatron applies with batched isend/irecv.

Usage::

    stage = nn.Sequential(*all_layers[my_slice])          # this rank's layers
    pp = PipelineParallel(stage, pg, activation_shape=(micro, seq, width),
                          loss_fn=F.mse_loss)
    opt = torch.optim.SGD(stage.parameters(), lr=0.1)

    opt.zero_grad()
    loss = pp.step(inputs, targets, n_microbatches=8)     # grads accumulated
    opt.step()

``inputs`` is read only on the first stage and ``targets`` only on the last, so
the other ranks may pass ``None``. The activation shape is fixed for the whole
run, which is the usual pipeline-parallel restriction: both sides of every
boundary need to size their buffers without an extra round trip.
"""

from __future__ import annotations

from collections import deque

import torch
from torch import nn

from .process_group import ProcessGroup

SCHEDULES = ("1f1b", "gpipe")


class PipelineParallel:
    def __init__(
        self,
        stage: nn.Module,
        pg: ProcessGroup,
        activation_shape: tuple[int, ...],
        loss_fn=None,
        schedule: str = "1f1b",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        if schedule not in SCHEDULES:
            raise ValueError(f"unknown schedule {schedule!r}; expected one of {SCHEDULES}")
        if loss_fn is None and pg.rank == pg.world_size - 1:
            raise ValueError("the last stage needs loss_fn to start the backward pass")
        self.stage = stage
        self.pg = pg
        self.schedule = schedule
        self.activation_shape = tuple(activation_shape)
        self.loss_fn = loss_fn
        self.dtype = dtype

        self.rank = pg.rank
        self.n_stages = pg.world_size
        self.is_first = self.rank == 0
        self.is_last = self.rank == self.n_stages - 1
        self.prev = self.rank - 1
        self.next = self.rank + 1

        # (input leaf, output) per microbatch awaiting its backward, in order.
        self._queue: deque[tuple[torch.Tensor | None, torch.Tensor]] = deque()
        self.in_flight_peak = 0

    # ---- boundary traffic -------------------------------------------------

    def _new_buffer(self) -> torch.Tensor:
        return torch.empty(self.activation_shape, dtype=self.dtype)

    def _recv_activation(self) -> torch.Tensor:
        buffer = self._new_buffer()
        self.pg.recv(buffer, self.prev)
        # A leaf that requires grad: after backward, its .grad is exactly what
        # the previous stage needs.
        return buffer.requires_grad_(True)

    def _send_activation(self, activation: torch.Tensor) -> None:
        self.pg.send(activation.detach().contiguous(), self.next)

    def _recv_gradient(self) -> torch.Tensor:
        buffer = self._new_buffer()
        self.pg.recv(buffer, self.next)
        return buffer

    def _send_gradient(self, grad: torch.Tensor) -> None:
        self.pg.send(grad.contiguous(), self.prev)

    def _send_activation_recv_gradient(self, activation: torch.Tensor) -> torch.Tensor:
        """Fused: push the activation to the next stage while pulling its gradient.

        Both halves talk to the same neighbor, so doing them concurrently is
        what keeps the steady state from deadlocking when an activation is
        larger than the socket buffer.
        """
        buffer = self._new_buffer()
        self.pg.send_recv(
            activation.detach().contiguous(), self.next, buffer, self.next
        )
        return buffer

    # ---- one microbatch ---------------------------------------------------

    def _forward(self, micro_input: torch.Tensor | None, micro_target, n_micro: int):
        """Run this stage's forward for one microbatch and queue it for backward."""
        if self.is_first:
            x = None
            activation_in = micro_input
        else:
            x = self._recv_activation()
            activation_in = x

        out = self.stage(activation_in)

        if self.is_last:
            # Scale so the microbatches sum to the full-batch mean gradient.
            out = self.loss_fn(out, micro_target) / n_micro
        self._queue.append((x, out))
        self.in_flight_peak = max(self.in_flight_peak, len(self._queue))
        return out

    def _backward(self, grad: torch.Tensor | None) -> None:
        """Run backward for the oldest queued microbatch.

        ``grad`` is the gradient of that microbatch's output, already received
        from the next stage. The last stage passes None and starts from its
        loss instead.

        The distinction that matters: the gradient arriving from the next stage
        belongs to the *oldest* microbatch in flight, not to whichever one this
        stage just pushed forward. The next stage is several microbatches behind
        on its own schedule, which is exactly why the queue is FIFO.
        """
        x, out = self._queue.popleft()
        if self.is_last:
            out.backward()
        else:
            out.backward(grad)
        if not self.is_first:
            self._send_gradient(x.grad)

    # ---- schedules --------------------------------------------------------

    def step(
        self,
        inputs: torch.Tensor | None = None,
        targets: torch.Tensor | None = None,
        n_microbatches: int = 1,
    ) -> float:
        """Run one optimizer step's worth of microbatches, accumulating grads.

        Returns the summed loss on the last stage and 0.0 elsewhere.
        """
        if n_microbatches < 1:
            raise ValueError("n_microbatches must be at least 1")
        if self.is_first and inputs is None:
            raise ValueError("the first stage needs inputs")
        if self.is_last and targets is None:
            raise ValueError("the last stage needs targets")

        # Checked explicitly: torch.chunk is happy to return uneven pieces, and
        # unequal microbatches would both break the fixed activation buffers and
        # silently misweight the averaged loss.
        for name, tensor in (("inputs", inputs), ("targets", targets)):
            if tensor is not None and tensor.shape[0] % n_microbatches:
                raise ValueError(
                    f"{name} batch of {tensor.shape[0]} is not divisible by "
                    f"n_microbatches={n_microbatches}"
                )

        micro_inputs = (
            list(torch.chunk(inputs, n_microbatches))
            if self.is_first
            else [None] * n_microbatches
        )
        micro_targets = (
            list(torch.chunk(targets, n_microbatches))
            if self.is_last
            else [None] * n_microbatches
        )

        self._queue.clear()
        self.in_flight_peak = 0
        losses: list[float] = []

        if self.n_stages == 1:
            for i in range(n_microbatches):
                out = self._forward(micro_inputs[i], micro_targets[i], n_microbatches)
                losses.append(float(out.detach()))
                self._backward(None)
            return sum(losses)

        if self.schedule == "gpipe":
            self._run_gpipe(micro_inputs, micro_targets, n_microbatches, losses)
        else:
            self._run_1f1b(micro_inputs, micro_targets, n_microbatches, losses)
        return sum(losses)

    def _record(self, out: torch.Tensor, losses: list[float]) -> None:
        if self.is_last:
            losses.append(float(out.detach()))

    def _run_gpipe(self, micro_inputs, micro_targets, n_micro, losses) -> None:
        """All forwards, then all backwards. Holds every microbatch at once.

        Peak activation memory therefore scales with ``n_micro``, which is the
        cost 1F1B exists to remove.
        """
        for i in range(n_micro):
            out = self._forward(micro_inputs[i], micro_targets[i], n_micro)
            self._record(out, losses)
            if not self.is_last:
                self._send_activation(out)
        for _ in range(n_micro):
            # Gradients arrive in the order the forwards were issued, matching
            # the FIFO queue.
            self._backward(None if self.is_last else self._recv_gradient())

    def _run_1f1b(self, micro_inputs, micro_targets, n_micro, losses) -> None:
        """Warm up, alternate one forward with one backward, then drain.

        Stage ``s`` warms up with ``W-1-s`` forwards, so the last stage starts
        its backwards immediately and the first stage carries the deepest queue.
        That gradient in the steady state is why the depth stops at ``W-s``.
        """
        warmup = min(self.n_stages - 1 - self.rank, n_micro)
        steady = n_micro - warmup
        index = 0

        for _ in range(warmup):
            out = self._forward(micro_inputs[index], micro_targets[index], n_micro)
            self._record(out, losses)
            if not self.is_last:
                self._send_activation(out)
            index += 1

        for _ in range(steady):
            out = self._forward(micro_inputs[index], micro_targets[index], n_micro)
            self._record(out, losses)
            index += 1
            if self.is_last:
                # Nothing to push forward, and the loss is already here.
                self._backward(None)
            else:
                # Push the microbatch just computed while pulling back the
                # gradient for the oldest one still queued. Fusing the two is
                # what prevents a deadlock against the neighbor's own send.
                self._backward(self._send_activation_recv_gradient(out))

        for _ in range(warmup):
            self._backward(None if self.is_last else self._recv_gradient())
