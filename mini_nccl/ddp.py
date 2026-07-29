"""Data-parallel training built on mini-nccl collectives.

This is the same architecture as ``torch.nn.parallel.DistributedDataParallel``,
reimplemented from scratch:

- **Gradient bucketing.** Parameters are grouped (in reverse registration
  order, which approximates the order gradients become ready during
  backward) into flat buckets of ~``bucket_cap_mb``. One all-reduce per
  bucket amortizes per-message latency across many small tensors.
- **Gradient views.** Each ``param.grad`` is a view into its bucket's flat
  buffer, so autograd accumulates directly into the communication buffer:
  no flatten/unflatten copies on the hot path.
- **Compute/communication overlap.** A post-accumulate-grad hook marks a
  bucket ready; a dedicated communication thread reduces buckets *while
  backward is still computing* earlier layers' gradients. Buckets are
  always reduced in fixed index order so every rank issues the identical
  collective sequence, the same invariant NCCL communicators require.

Usage::

    model = DistributedDataParallel(MyModel(), pg)
    for batch in loader:
        model.zero_grad()
        loss = criterion(model(batch))
        loss.backward()
        model.sync()          # wait for gradient reduction
        optimizer.step()

Gradient accumulation skips the reduction on all but the last microbatch::

    with model.no_sync():
        loss_a.backward()     # accumulates into bucket buffers, no traffic
    loss_b.backward()
    model.sync()              # one reduction covers both backwards

Not supported (kept out of scope deliberately): unused-parameter detection.
"""

from __future__ import annotations

import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager

import torch
from torch import nn

from . import collectives
from .process_group import ProcessGroup


def average_gradients(pg: ProcessGroup, params, algorithm: str = "ring") -> int:
    """Average the gradients of ``params`` across ``pg``. Returns bytes reduced.

    The explicit form of what DDP's reducer does implicitly, for the cases where
    hooking into backward is the wrong tool:

    - **Pipeline parallelism** runs backward once per microbatch, so a hook
      would reduce many times per step instead of once. The gradients want
      averaging after the whole schedule drains.
    - **Composed parallelism** needs gradients reduced along one mesh dimension
      only, over a subgroup rather than the whole world.

    One flat buffer, so it is one collective regardless of parameter count.
    """
    if pg.world_size == 1:
        return 0
    grads = [p.grad for p in params if p.grad is not None]
    if not grads:
        return 0
    flat = torch.cat([g.reshape(-1) for g in grads])
    collectives.all_reduce(pg, flat, algorithm=algorithm)
    flat.div_(pg.world_size)
    offset = 0
    for grad in grads:
        grad.copy_(flat[offset : offset + grad.numel()].view(grad.shape))
        offset += grad.numel()
    return flat.numel() * flat.element_size()


class _Bucket:
    def __init__(self, params: list[nn.Parameter], dtype: torch.dtype) -> None:
        self.params = params
        numels = [p.numel() for p in params]
        self.buffer = torch.zeros(sum(numels), dtype=dtype)
        self.views: list[torch.Tensor] = []
        offset = 0
        for p, n in zip(params, numels, strict=True):
            self.views.append(self.buffer[offset : offset + n].view(p.shape))
            offset += n
        self.ready_count = 0
        self.ready = threading.Event()
        self.lock = threading.Lock()

    def attach_grads(self) -> None:
        """Point every param.grad at its slice of the flat buffer."""
        for p, view in zip(self.params, self.views, strict=True):
            p.grad = view

    def mark_ready(self) -> None:
        with self.lock:
            self.ready_count += 1
            if self.ready_count == len(self.params):
                self.ready.set()

    def reset(self) -> None:
        with self.lock:
            self.ready_count = 0
            self.ready.clear()


class DistributedDataParallel(nn.Module):
    def __init__(
        self,
        module: nn.Module,
        pg: ProcessGroup,
        bucket_cap_mb: float = 1.0,
        overlap: bool = True,
        algorithm: str = "ring",
    ) -> None:
        super().__init__()
        self.module = module
        self.pg = pg
        self.algorithm = algorithm
        self._overlap = overlap and pg.world_size > 1
        self._reduce_future: Future | None = None
        self._buckets: list[_Bucket] = []
        self._accumulating = False
        #: ``perf_counter_ns`` when this iteration's reduction was dispatched,
        #: or None if it has not been. Observable so that "the reduction is
        #: dispatched during backward" can be checked rather than assumed;
        #: whether the comm thread then gets CPU time is the OS's business.
        self.reduce_submitted_ns: int | None = None

        # Every rank starts from rank 0's weights.
        if pg.world_size > 1:
            with torch.no_grad():
                for p in module.parameters():
                    collectives.broadcast(pg, p.data.contiguous(), src=0)
                for b in module.buffers():
                    collectives.broadcast(pg, b.data.contiguous(), src=0)

        if pg.world_size > 1:
            self._build_buckets(bucket_cap_mb)
            for bucket in self._buckets:
                bucket.attach_grads()
                for p in bucket.params:
                    p.register_post_accumulate_grad_hook(self._make_hook(bucket))
            self._comm = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mn-ddp")

    def _build_buckets(self, cap_mb: float) -> None:
        cap_bytes = int(cap_mb * 1024 * 1024)
        current: list[nn.Parameter] = []
        current_bytes = 0
        current_dtype: torch.dtype | None = None
        # Reverse order: the last layers' grads are produced first in backward,
        # so their bucket fills (and starts communicating) earliest.
        for p in reversed(list(self.module.parameters())):
            if not p.requires_grad:
                continue
            nbytes = p.numel() * p.element_size()
            if current and (current_bytes + nbytes > cap_bytes or p.dtype != current_dtype):
                self._buckets.append(_Bucket(current, current_dtype))
                current, current_bytes = [], 0
            current.append(p)
            current_bytes += nbytes
            current_dtype = p.dtype
        if current:
            self._buckets.append(_Bucket(current, current_dtype))

    def _make_hook(self, bucket: _Bucket):
        def hook(_param: nn.Parameter) -> None:
            if self._accumulating:
                # Inside no_sync(): autograd keeps adding into the bucket
                # buffers (grads are views), so there is nothing to do but
                # stay off the wire.
                return
            bucket.mark_ready()
            # The first ready mark of an iteration kicks off the reducer.
            if self._overlap and self._reduce_future is None:
                self.reduce_submitted_ns = time.perf_counter_ns()
                self._reduce_future = self._comm.submit(self._reduce_all)

        return hook

    def _reduce_all(self) -> None:
        for i, bucket in enumerate(self._buckets):
            bucket.ready.wait()
            ev = self.pg.recorder.start(
                f"ddp_bucket{i}",
                self.algorithm,
                channel=-1,
                nbytes=bucket.buffer.numel() * bucket.buffer.element_size(),
                params=len(bucket.params),
            )
            try:
                collectives.all_reduce(self.pg, bucket.buffer, algorithm=self.algorithm)
                bucket.buffer.div_(self.pg.world_size)
            finally:
                self.pg.recorder.finish(ev)

    @contextmanager
    def no_sync(self):
        """Accumulate gradients locally without any communication.

        Do not call ``zero_grad()`` inside the block: the point is for the
        bucket buffers to keep summing across microbatches.
        """
        previous = self._accumulating
        self._accumulating = True
        try:
            yield
        finally:
            self._accumulating = previous

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def sync(self) -> None:
        """Wait until all gradients are reduced. Call after ``backward()``."""
        if self.pg.world_size == 1:
            return
        if self._overlap:
            if self._reduce_future is not None:
                self._reduce_future.result()
                self._reduce_future = None
        else:
            self._reduce_all()
        for bucket in self._buckets:
            bucket.reset()
        self.reduce_submitted_ns = None

    def zero_grad(self, set_to_none: bool = False) -> None:
        """Zero gradients in place.

        ``set_to_none`` is ignored: grads must stay views into the bucket
        buffers, so they are zeroed rather than dropped.
        """
        if self.pg.world_size == 1:
            self.module.zero_grad(set_to_none=False)
            return
        for bucket in self._buckets:
            bucket.buffer.zero_()
            bucket.attach_grads()  # restore views in case something detached them
