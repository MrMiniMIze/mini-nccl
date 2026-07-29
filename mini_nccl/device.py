"""Collectives over tensors that live on an accelerator.

A socket can only send host memory, so a device tensor has to be staged
through the host on its way out and back:

    device buffer -> host buffer -> socket -> host buffer -> device buffer

Three things separate a fast implementation of that from a slow one.

**Pinned host memory.** A copy out of pageable host memory cannot be done by
DMA, so the driver stages it through an internal pinned buffer, costing an
extra copy and forcing the transfer to be synchronous. Allocating the staging
buffers with ``pin_memory=True`` lets the copy engine move bytes directly and
asynchronously.

**A separate copy stream.** Issued on their own stream, the device copies do
not serialize behind whatever the compute stream is doing, and the CPU can
keep working while they are in flight.

**Overlapping the copies with the network.** This is the part that matters
most and the reason this file exists. The payload is split into chunks, and
while chunk *k*'s host bytes are being written to the socket, chunk *k+1* is
already being copied off the device. The copy engine and the CPU doing the
socket write are different hardware, so the two genuinely proceed at once.

That last point is worth connecting to the CPU results elsewhere in this
repo: the same pipelining idea *lost* 5-40% on loopback TCP, because there
the "network" is a memory copy performed by the very cores that would do the
copying. Here the resources are actually distinct, which is the condition the
optimization always needed. ``benchmarks/bench_device.py`` measures whether
that holds.

Two entry points, so the two strategies can be compared rather than assumed:

- :func:`all_reduce_staged` moves the whole tensor to the host, runs any of
  the existing CPU collectives on it, and moves the result back. Simple, and
  it inherits channels, algorithm selection, and the narrow-wire option.
- :func:`all_reduce_pipelined` runs a ring directly on the device tensor with
  the chunked copy/network overlap described above, reducing on the device.

Both work on CPU tensors as well, where the staging copies are ordinary
memory copies and the events are no-ops. That is deliberate: it keeps the
chunking and double-buffering logic exercised by the test suite on machines
without a GPU.
"""

from __future__ import annotations

import torch

from . import collectives
from .process_group import ProcessGroup

DEFAULT_CHUNK_BYTES = 4 * 1024 * 1024


def is_accelerated(tensor: torch.Tensor) -> bool:
    return tensor.device.type == "cuda"


class _Slot:
    """One double-buffer slot: host staging space plus its copy events."""

    def __init__(self, numel: int, dtype: torch.dtype, pinned: bool) -> None:
        self.send = torch.empty(numel, dtype=dtype, pin_memory=pinned)
        self.recv = torch.empty(numel, dtype=dtype, pin_memory=pinned)
        # Events let the CPU wait for a specific copy instead of draining the
        # whole stream, which is what makes overlapping possible.
        self.out_done = torch.cuda.Event() if pinned else None
        self.in_done = torch.cuda.Event() if pinned else None
        self.in_flight = False


class Staging:
    """Host staging buffers and the copy stream for one collective.

    Reused across the steps of a collective so the pinned allocation (which is
    expensive, since it locks pages) happens once.
    """

    def __init__(
        self,
        numel: int,
        dtype: torch.dtype,
        device: torch.device,
        chunk_bytes: int = DEFAULT_CHUNK_BYTES,
        n_slots: int = 2,
    ) -> None:
        self.device = device
        self.accelerated = device.type == "cuda"
        itemsize = torch.empty(0, dtype=dtype).element_size()
        self.chunk_numel = max(1, min(numel, chunk_bytes // max(1, itemsize)))
        self.n_chunks = -(-numel // self.chunk_numel)
        self.stream = torch.cuda.Stream() if self.accelerated else None
        self.slots = [
            _Slot(self.chunk_numel, dtype, self.accelerated)
            for _ in range(min(n_slots, self.n_chunks))
        ]

    def ranges(self, numel: int) -> list[tuple[int, int]]:
        return [
            (start, min(start + self.chunk_numel, numel))
            for start in range(0, numel, self.chunk_numel)
        ]

    # ---- copies ----------------------------------------------------------

    def order_copies_after_compute(self) -> None:
        """Make the copy stream wait for work already queued on the compute stream.

        Required, and not obvious: a ring step's reduction runs on the compute
        stream, and the *next* step copies that same block off the device on the
        copy stream. Without this dependency the copy is free to read the block
        before the addition has landed, and the peer receives a partially
        reduced value. That failure is invisible on CPU, where the streams are
        no-ops, and it showed up on the first real GPU run.
        """
        if self.accelerated:
            self.stream.wait_stream(torch.cuda.current_stream())

    def start_copy_out(self, slot: _Slot, source: torch.Tensor) -> None:
        """Begin device -> host for ``source`` into ``slot.send``."""
        view = slot.send[: source.numel()]
        if not self.accelerated:
            view.copy_(source)
            return
        with torch.cuda.stream(self.stream):
            view.copy_(source, non_blocking=True)
        slot.out_done.record(self.stream)

    def wait_copy_out(self, slot: _Slot) -> None:
        """Block the CPU until this slot's outbound copy has landed."""
        if self.accelerated:
            slot.out_done.synchronize()

    def start_copy_in(self, slot: _Slot, destination: torch.Tensor) -> None:
        """Begin host -> device from ``slot.recv`` into ``destination``."""
        view = slot.recv[: destination.numel()]
        if not self.accelerated:
            destination.copy_(view)
            return
        with torch.cuda.stream(self.stream):
            destination.copy_(view, non_blocking=True)
        slot.in_done.record(self.stream)
        slot.in_flight = True

    def wait_copy_in(self, slot: _Slot) -> None:
        """Block the CPU until this slot's inbound copy is done reusing it."""
        if self.accelerated and slot.in_flight:
            slot.in_done.synchronize()
            slot.in_flight = False

    def order_compute_after_copies(self) -> None:
        """Make the compute stream wait for every outstanding inbound copy.

        Cheaper than a device synchronize: the CPU keeps running and only the
        device work queued afterwards is ordered behind the copies.
        """
        if not self.accelerated:
            return
        current = torch.cuda.current_stream()
        for slot in self.slots:
            if slot.in_flight:
                current.wait_event(slot.in_done)


def _staged_exchange(
    pg: ProcessGroup,
    send_device: torch.Tensor,
    recv_device: torch.Tensor,
    dst: int,
    src: int,
    staging: Staging,
) -> None:
    """Full-duplex exchange of device tensors, chunk-pipelined through the host.

    The loop keeps one chunk on the wire while the next is being copied off
    the device: chunk ``k+1``'s copy is issued *before* chunk ``k``'s socket
    exchange, so the copy engine and the network overlap.
    """
    ranges = staging.ranges(send_device.numel())
    slots = staging.slots

    # Everything this exchange reads off the device was produced by the
    # compute stream (the caller's data, or the previous step's reduction), so
    # the copy stream has to be ordered behind it exactly once, here.
    staging.order_copies_after_compute()

    def issue(index: int) -> None:
        lo, hi = ranges[index]
        slot = slots[index % len(slots)]
        staging.wait_copy_in(slot)  # do not clobber a slot still being read
        staging.start_copy_out(slot, send_device[lo:hi])

    issue(0)
    for k, (lo, hi) in enumerate(ranges):
        slot = slots[k % len(slots)]
        staging.wait_copy_out(slot)
        if k + 1 < len(ranges) and len(slots) > 1:
            issue(k + 1)  # overlaps with the exchange below
        width = hi - lo
        pg.send_recv(slot.send[:width], dst, slot.recv[:width], src)
        staging.start_copy_in(slot, recv_device[lo:hi])
        if k + 1 < len(ranges) and len(slots) == 1:
            issue(k + 1)
    staging.order_compute_after_copies()


def all_reduce_staged(
    pg: ProcessGroup,
    tensor: torch.Tensor,
    op: str = "sum",
    algorithm: str = "auto",
    wire_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Move the tensor to pinned host memory, reduce there, move it back.

    The straightforward approach, and the baseline the pipelined version has
    to beat. It reuses every CPU-side feature (channels, algorithm selection,
    narrow wire) at the cost of two full-size copies that overlap with
    nothing.
    """
    if not is_accelerated(tensor):
        return collectives.all_reduce(
            pg, tensor, op=op, algorithm=algorithm, wire_dtype=wire_dtype
        )
    flat = tensor.view(-1)
    host = torch.empty(flat.numel(), dtype=flat.dtype, pin_memory=True)
    host.copy_(flat, non_blocking=True)
    torch.cuda.current_stream().synchronize()
    collectives.all_reduce(pg, host, op=op, algorithm=algorithm, wire_dtype=wire_dtype)
    flat.copy_(host, non_blocking=True)
    torch.cuda.current_stream().synchronize()
    return tensor


def all_reduce_pipelined(
    pg: ProcessGroup,
    tensor: torch.Tensor,
    op: str = "sum",
    chunk_bytes: int = DEFAULT_CHUNK_BYTES,
) -> torch.Tensor:
    """Ring all-reduce on a device tensor, overlapping copies with the network.

    The arithmetic stays on the device; only the transfers are staged. Each of
    the ``2(W-1)`` ring steps moves one block through the host in chunks, and
    the reduction for a block runs on the device once its chunks have landed.
    """
    collectives._check_op(op)
    if not tensor.is_contiguous():
        # Checked before the flatten, since view() on a non-contiguous tensor
        # raises something far less informative.
        raise ValueError("all_reduce_pipelined requires a contiguous tensor")
    if pg.world_size == 1:
        return tensor

    W, r = pg.world_size, pg.rank
    flat = tensor.view(-1)

    chunk = -(-flat.numel() // W)
    padded = flat
    if chunk * W != flat.numel():
        padded = torch.zeros(chunk * W, dtype=flat.dtype, device=flat.device)
        padded[: flat.numel()] = flat
    blocks = padded.view(W, chunk)

    staging = Staging(chunk, flat.dtype, flat.device, chunk_bytes=chunk_bytes)
    scratch = torch.empty(chunk, dtype=flat.dtype, device=flat.device)
    reduce_op = collectives._OPS[op]
    right, left = (r + 1) % W, (r - 1) % W

    nbytes = flat.numel() * flat.element_size()
    ev = pg.recorder.start(
        "all_reduce_pipelined", "ring", channel=-1, nbytes=nbytes, op=op,
        device=str(flat.device), chunks=staging.n_chunks,
    )
    try:
        # Phase 1 (reduce-scatter): accumulate on the device as blocks land.
        for step in range(W - 1):
            send_idx = (r - step) % W
            recv_idx = (r - step - 1) % W
            _staged_exchange(pg, blocks[send_idx], scratch, right, left, staging)
            reduce_op(blocks[recv_idx], scratch)

        # Phase 2 (all-gather): circulate the finished blocks.
        for step in range(W - 1):
            send_idx = (r + 1 - step) % W
            recv_idx = (r - step) % W
            _staged_exchange(pg, blocks[send_idx], blocks[recv_idx], right, left, staging)

        if padded.data_ptr() != flat.data_ptr():
            flat.copy_(padded[: flat.numel()])
        if staging.accelerated:
            torch.cuda.current_stream().synchronize()
    finally:
        pg.recorder.finish(ev)
    return tensor


def device_report() -> dict[str, object]:
    """What the local machine can actually run, for benchmarks and tests."""
    report: dict[str, object] = {
        "torch": torch.__version__,
        "cuda_built": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        report["device_name"] = torch.cuda.get_device_name(0)
        report["device_count"] = torch.cuda.device_count()
        major, minor = torch.cuda.get_device_capability(0)
        report["compute_capability"] = f"{major}.{minor}"
        report["total_memory_gib"] = round(
            torch.cuda.get_device_properties(0).total_memory / 2**30, 2
        )
    return report
