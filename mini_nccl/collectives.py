"""Collective communication algorithms.

Four all-reduce strategies, mirroring the trade-offs inside NCCL:

- ``ring``: bandwidth-optimal. Reduce-scatter then all-gather around a
  ring; each rank sends ``2(W-1)/W * n`` bytes total regardless of world
  size, but latency grows linearly with ``W`` (2(W-1) serialized steps).
- ``tree``: latency-optimal. Binomial-tree reduce to rank 0 followed by
  a binomial-tree broadcast: ``2*ceil(log2 W)`` steps, but interior ranks
  forward the full payload, so it moves more bytes than ring on large
  tensors.
- ``halving``: recursive halving-doubling (Rabenseifner). Reduce-scatter by
  recursive halving, all-gather by recursive doubling: ring's per-rank byte
  count with tree's ``2*log2(W)`` step count. Requires a power-of-two world
  size; other world sizes fall back to ring.
- ``naive``: the parameter-server baseline; every rank sends to rank 0,
  which reduces and sends the result back. Rank 0 handles ``O(W * n)``
  bytes, so it is included mainly to show in benchmarks *why* the others
  exist.

``auto`` picks tree for small messages (latency-bound) and ring for large
ones (bandwidth-bound), like NCCL's algorithm selection.

Two optimizations apply to the bandwidth-bound path:

- **Channels.** A large ring all-reduce is split across several independent
  connections, each driven by its own thread, the same trick NCCL uses to
  fill a link with more than one flow.
- **Sliced reduction.** Inside a ring step the receive is consumed in
  slices, so each slice is reduced while the next one is still arriving.
"""

from __future__ import annotations

import os
from functools import partial

import torch

from .process_group import ProcessGroup

# Crossover point between latency-bound and bandwidth-bound messages.
# Measured on loopback TCP: at 1 MiB tree and ring tie at both world sizes
# tested, and ring pulls clearly ahead by 4 MiB. Re-derive it for your own
# fabric from benchmarks/bench_allreduce.py.
RING_THRESHOLD_BYTES = 1024 * 1024

# Splitting across channels only pays once each channel has a real payload.
# 8 MiB is where a second channel started winning on loopback TCP (below it,
# extra threads cost more than the parallel socket gains); re-derive it for
# your fabric with benchmarks/bench_ablation.py.
CHANNEL_MIN_BYTES = int(os.environ.get("MINI_NCCL_CHANNEL_MIN_BYTES", 8 * 1024 * 1024))

# Slice size for overlapping reduction with transfer inside a ring step.
# Off by default: measured on loopback TCP it *costs* 5-40% (see the ablation
# table in the README), because the reduction and the transfer contend for the
# same cores. It pays when reduction runs on hardware independent of the
# transport, which is the case NCCL is built for (GPU kernels + NIC DMA), so
# the mechanism stays available via MINI_NCCL_MAX_SLICES.
SLICE_TARGET_BYTES = 256 * 1024
MAX_SLICES = int(os.environ.get("MINI_NCCL_MAX_SLICES", "1"))

#: Selectable all-reduce algorithms, in the order benchmarks report them.
ALGORITHMS = ("ring", "tree", "halving", "naive")

_OPS = {
    "sum": lambda acc, other: acc.add_(other),
    "prod": lambda acc, other: acc.mul_(other),
    "max": lambda acc, other: torch.maximum(acc, other, out=acc),
    "min": lambda acc, other: torch.minimum(acc, other, out=acc),
}


def _flat(tensor: torch.Tensor) -> torch.Tensor:
    if not tensor.is_contiguous():
        raise ValueError("collectives require contiguous tensors")
    return tensor.view(-1)


def _check_op(op: str) -> None:
    if op not in _OPS:
        raise ValueError(f"unknown op {op!r}; expected one of {sorted(_OPS)}")


def _n_channels_for(pg: ProcessGroup, nbytes: int) -> int:
    """Channel count, derived only from sizes so every rank agrees."""
    return max(1, min(pg.n_channels, nbytes // CHANNEL_MIN_BYTES))


def _n_slices_for(nbytes: int) -> int:
    return max(1, min(MAX_SLICES, nbytes // SLICE_TARGET_BYTES))


def _split(flat: torch.Tensor, n_parts: int) -> list[torch.Tensor]:
    """Contiguous, deterministic split; remainder goes to the leading parts."""
    n = flat.numel()
    base, rem = divmod(n, n_parts)
    parts, offset = [], 0
    for i in range(n_parts):
        length = base + (1 if i < rem else 0)
        parts.append(flat[offset : offset + length])
        offset += length
    return parts


#: Dtypes too narrow to accumulate a sum of many terms without losing it.
LOW_PRECISION = (torch.bfloat16, torch.float16)


def all_reduce(
    pg: ProcessGroup,
    tensor: torch.Tensor,
    op: str = "sum",
    algorithm: str = "auto",
    wire_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """In-place all-reduce of ``tensor`` across all ranks.

    ``wire_dtype`` separates the precision carried between ranks from the
    precision the sum is accumulated in, the standard arrangement for
    large-model gradient reduction:

    - ``float32`` tensor, ``wire_dtype=torch.bfloat16``: halves the bytes on
      the wire, with every addition still performed in float32.
    - ``bfloat16`` tensor, ``wire_dtype=torch.bfloat16``: same bytes, but the
      running sum is promoted to float32 explicitly rather than relying on
      PyTorch's kernels to widen internally.

    A measured caveat, because it is the opposite of what the name suggests:
    widening the accumulator does **not** measurably improve a bfloat16
    reduction here (see tests/test_low_precision.py). The error is dominated
    by rounding the partial sum back onto the narrow wire at every hop, so it
    scales with hop count, not accumulator width. That makes the *algorithm*
    the real lever on low-precision accuracy: tree crosses O(log W) hops and
    holds its error roughly constant in W, while ring crosses O(W) and grows
    linearly. Low precision therefore inverts the usual ranking, since ring is
    the bandwidth-optimal choice but the least accurate one.

    Supported for the ``ring`` and ``tree`` algorithms.
    """
    _check_op(op)
    if pg.world_size == 1:
        return tensor
    nbytes = tensor.numel() * tensor.element_size()
    if algorithm == "auto":
        algorithm = "tree" if nbytes <= RING_THRESHOLD_BYTES else "ring"
    if algorithm == "halving" and pg.world_size & (pg.world_size - 1):
        # Non-power-of-two would need the extra-ranks fold-in step; ring is
        # the honest fallback rather than a silently different algorithm.
        algorithm = "ring"

    if wire_dtype is not None:
        if algorithm not in ("ring", "tree"):
            raise ValueError(f"wire_dtype is implemented for ring and tree, not {algorithm!r}")
        if wire_dtype.itemsize > tensor.element_size():
            raise ValueError(
                f"wire_dtype {wire_dtype} is wider than the tensor's {tensor.dtype}"
            )
        nbytes = tensor.numel() * wire_dtype.itemsize

    ev = pg.recorder.start(
        "all_reduce",
        algorithm,
        channel=-1,
        nbytes=nbytes,
        op=op,
        wire=str(wire_dtype) if wire_dtype else "native",
    )
    try:
        # A narrow tensor gets a float32 working copy, so the accumulator is
        # wide even though the input and output are not.
        working = tensor
        if wire_dtype is not None and tensor.dtype in LOW_PRECISION:
            working = _flat(tensor).float()

        if algorithm == "ring":
            n_channels = _n_channels_for(pg, nbytes)
            parts = _split(_flat(working), n_channels)
            pg.run_per_channel(
                [
                    partial(_ring_on_channel, pg, part, op, c, wire_dtype)
                    for c, part in enumerate(parts)
                ]
            )
        elif algorithm == "tree":
            _binomial_reduce(pg, working, op, root=0, wire_dtype=wire_dtype)
            _binomial_broadcast(pg, working, root=0, wire_dtype=wire_dtype)
        elif algorithm == "halving":
            _halving_doubling(pg, working, op)
        elif algorithm == "naive":
            _naive_all_reduce(pg, working, op)
        else:
            raise ValueError(f"unknown algorithm {algorithm!r}")

        if working is not tensor:
            _flat(tensor).copy_(working)
    finally:
        pg.recorder.finish(ev)
    return tensor


def _ring_on_channel(
    pg: ProcessGroup,
    segment: torch.Tensor,
    op: str,
    channel: int,
    wire_dtype: torch.dtype | None = None,
) -> None:
    """Full ring all-reduce of one segment, entirely on one channel.

    With ``wire_dtype`` set, every hop casts down to send and back up to
    accumulate, so the blocks stay in the segment's (wider) dtype throughout.
    Each hop re-rounds the partial sum, which is the price of a narrow wire;
    what it buys is that the arithmetic itself never happens in the narrow
    type.
    """
    W, r = pg.world_size, pg.rank
    flat = segment.view(-1)
    nbytes = flat.numel() * flat.element_size()
    ev = pg.recorder.start("ring_segment", "ring", channel=channel, nbytes=nbytes)
    try:
        chunk = -(-flat.numel() // W)  # ceil division

        padded = flat
        if chunk * W != flat.numel():
            padded = torch.zeros(chunk * W, dtype=flat.dtype)
            padded[: flat.numel()] = flat
        blocks = padded.view(W, chunk)

        tmp = torch.empty(chunk, dtype=flat.dtype)
        right, left = (r + 1) % W, (r - 1) % W
        reduce_op = _OPS[op]
        n_slices = _n_slices_for(chunk * flat.element_size())
        send_wire = recv_wire = None
        if wire_dtype is not None:
            send_wire = torch.empty(chunk, dtype=wire_dtype)
            recv_wire = torch.empty(chunk, dtype=wire_dtype)

        # Phase 1 (reduce-scatter): after W-1 steps, this rank holds the
        # fully reduced block (r + 1) % W. Each step forwards the block we
        # just accumulated while receiving the next partial from the left,
        # reducing each slice as it lands.
        for step in range(W - 1):
            send_idx = (r - step) % W
            recv_idx = (r - step - 1) % W
            target = blocks[recv_idx]
            if wire_dtype is None:
                pg.send_recv_sliced(
                    blocks[send_idx],
                    right,
                    tmp,
                    left,
                    n_slices,
                    lambda start, end, t=target: reduce_op(t[start:end], tmp[start:end]),
                    channel,
                )
            else:
                send_wire.copy_(blocks[send_idx])
                pg.send_recv(send_wire, right, recv_wire, left, channel)
                if op == "sum":
                    # Type promotion widens each element inside the add, so
                    # this stays one pass over the data instead of a separate
                    # widening copy followed by an add.
                    target.add_(recv_wire)
                else:
                    tmp.copy_(recv_wire)
                    reduce_op(target, tmp)

        # Phase 2 (all-gather): circulate the reduced blocks around the ring.
        for step in range(W - 1):
            send_idx = (r + 1 - step) % W
            recv_idx = (r - step) % W
            if wire_dtype is None:
                pg.send_recv(blocks[send_idx], right, blocks[recv_idx], left, channel)
            else:
                send_wire.copy_(blocks[send_idx])
                pg.send_recv(send_wire, right, recv_wire, left, channel)
                blocks[recv_idx].copy_(recv_wire)

        if padded.data_ptr() != flat.data_ptr():
            flat.copy_(padded[: flat.numel()])
    finally:
        pg.recorder.finish(ev)


def _halving_doubling(pg: ProcessGroup, tensor: torch.Tensor, op: str) -> None:
    """Recursive halving reduce-scatter, then recursive doubling all-gather.

    Rank ``r`` ends the halving phase owning segment ``r`` of the reduced
    result (the bit pattern of the rank *is* the segment index, because the
    splits walk from the most significant bit down), so the doubling phase
    simply reverses the exchange order.
    """
    W, r = pg.world_size, pg.rank
    flat = _flat(tensor)
    chunk = -(-flat.numel() // W)
    padded = flat
    if chunk * W != flat.numel():
        padded = torch.zeros(chunk * W, dtype=flat.dtype)
        padded[: flat.numel()] = flat

    reduce_op = _OPS[op]
    tmp = torch.empty(padded.numel() // 2, dtype=flat.dtype)
    lo, hi = 0, padded.numel()

    mask = W >> 1
    while mask:
        partner = r ^ mask
        mid = (lo + hi) // 2
        if r & mask:
            send_view, keep = padded[lo:mid], padded[mid:hi]
        else:
            send_view, keep = padded[mid:hi], padded[lo:mid]
        recv_view = tmp[: keep.numel()]
        pg.send_recv_sliced(
            send_view,
            partner,
            recv_view,
            partner,
            _n_slices_for(keep.numel() * flat.element_size()),
            lambda start, end, k=keep, rv=recv_view: reduce_op(k[start:end], rv[start:end]),
        )
        lo, hi = (mid, hi) if r & mask else (lo, mid)
        mask >>= 1

    mask = 1
    while mask < W:
        partner = r ^ mask
        length = hi - lo
        if r & mask:
            pg.send_recv(padded[lo:hi], partner, padded[lo - length : lo], partner)
            lo -= length
        else:
            pg.send_recv(padded[lo:hi], partner, padded[hi : hi + length], partner)
            hi += length
        mask <<= 1

    if padded.data_ptr() != flat.data_ptr():
        flat.copy_(padded[: flat.numel()])


def _binomial_reduce(
    pg: ProcessGroup,
    tensor: torch.Tensor,
    op: str,
    root: int,
    wire_dtype: torch.dtype | None = None,
) -> None:
    """Reduce onto ``root`` along a binomial tree (works for any world size).

    This is where a wide accumulator pays most: an interior node adds up to
    ``log2(W)`` incoming buffers into one running total, and the root's total
    covers every rank. Accumulating that in the wire's narrow type would round
    after every addition.
    """
    W, r = pg.world_size, pg.rank
    vr = (r - root) % W  # virtual rank: the tree is always rooted at vr 0
    flat = _flat(tensor)
    tmp = torch.empty_like(flat)
    reduce_op = _OPS[op]
    wire = torch.empty(flat.numel(), dtype=wire_dtype) if wire_dtype else None

    mask = 1
    while mask < W:
        if vr & mask:
            dst = ((vr & ~mask) + root) % W
            if wire is None:
                pg.send(tensor, dst)
            else:
                wire.copy_(flat)
                pg.send(wire, dst)
            return  # non-roots send exactly once, then are done
        src = vr | mask
        if src < W:
            if wire is None:
                pg.recv(tmp, (src + root) % W)
                reduce_op(flat, tmp)
            elif op == "sum":
                pg.recv(wire, (src + root) % W)
                flat.add_(wire)  # promotion widens inside the add
            else:
                pg.recv(wire, (src + root) % W)
                tmp.copy_(wire)
                reduce_op(flat, tmp)
        mask <<= 1


def _binomial_broadcast(
    pg: ProcessGroup,
    tensor: torch.Tensor,
    root: int,
    wire_dtype: torch.dtype | None = None,
) -> None:
    W, r = pg.world_size, pg.rank
    if W == 1:
        return
    flat = _flat(tensor)
    wire = torch.empty(flat.numel(), dtype=wire_dtype) if wire_dtype else None
    vr = (r - root) % W
    if vr != 0:
        lowbit = vr & -vr
        parent = ((vr - lowbit) + root) % W
        if wire is None:
            pg.recv(tensor, parent)
        else:
            pg.recv(wire, parent)
            flat.copy_(wire)
        mask = lowbit >> 1
    else:
        mask = 1 << ((W - 1).bit_length() - 1)
    if wire is not None:
        wire.copy_(flat)
    while mask:
        if vr + mask < W:
            dst = ((vr + mask) + root) % W
            pg.send(tensor if wire is None else wire, dst)
        mask >>= 1


def _naive_all_reduce(pg: ProcessGroup, tensor: torch.Tensor, op: str) -> None:
    W, r = pg.world_size, pg.rank
    flat = _flat(tensor)
    reduce_op = _OPS[op]
    if r == 0:
        tmp = torch.empty_like(flat)
        for src in range(1, W):
            pg.recv(tmp, src)
            reduce_op(flat, tmp)
        for dst in range(1, W):
            pg.send(tensor, dst)
    else:
        pg.send(tensor, 0)
        pg.recv(tensor, 0)


def broadcast(pg: ProcessGroup, tensor: torch.Tensor, src: int = 0) -> torch.Tensor:
    """Broadcast ``tensor`` from rank ``src`` to all ranks, in place."""
    if pg.world_size == 1:
        return tensor
    nbytes = tensor.numel() * tensor.element_size()
    ev = pg.recorder.start("broadcast", "tree", channel=-1, nbytes=nbytes, root=src)
    try:
        _binomial_broadcast(pg, tensor, root=src)
    finally:
        pg.recorder.finish(ev)
    return tensor


def all_gather(pg: ProcessGroup, tensor: torch.Tensor) -> list[torch.Tensor]:
    """Gather equal-shaped tensors from every rank, via a ring.

    Returns a list of ``world_size`` tensors, indexed by rank.
    """
    W, r = pg.world_size, pg.rank
    src = tensor.contiguous()
    flat = src.view(-1)
    out = torch.empty((W, flat.numel()), dtype=flat.dtype)
    out[r] = flat
    if W > 1:
        ev = pg.recorder.start(
            "all_gather", "ring", channel=-1, nbytes=flat.numel() * flat.element_size()
        )
        try:
            right, left = (r + 1) % W, (r - 1) % W
            for step in range(W - 1):
                send_idx = (r - step) % W
                recv_idx = (r - step - 1) % W
                pg.send_recv(out[send_idx], right, out[recv_idx], left)
        finally:
            pg.recorder.finish(ev)
    return [out[i].view(tensor.shape).clone() for i in range(W)]


def reduce_scatter(pg: ProcessGroup, tensor: torch.Tensor, op: str = "sum") -> torch.Tensor:
    """Reduce across ranks and return this rank's 1/W-th of the result.

    ``tensor.numel()`` must be divisible by ``world_size`` (NCCL contract).
    The input is left unmodified.
    """
    _check_op(op)
    W, r = pg.world_size, pg.rank
    flat = _flat(tensor)
    if flat.numel() % W != 0:
        raise ValueError(f"numel {flat.numel()} not divisible by world_size {W}")
    if W == 1:
        return flat.clone()
    chunk = flat.numel() // W
    blocks = flat.clone().view(W, chunk)
    tmp = torch.empty(chunk, dtype=flat.dtype)
    right, left = (r + 1) % W, (r - 1) % W
    reduce_op = _OPS[op]
    ev = pg.recorder.start(
        "reduce_scatter", "ring", channel=-1, nbytes=flat.numel() * flat.element_size(), op=op
    )
    try:
        # Same schedule as the ring reduce-scatter phase, shifted by one so
        # the fully reduced block each rank ends up holding is its own index.
        for step in range(W - 1):
            send_idx = (r - step - 1) % W
            recv_idx = (r - step - 2) % W
            pg.send_recv(blocks[send_idx], right, tmp, left)
            reduce_op(blocks[recv_idx], tmp)
    finally:
        pg.recorder.finish(ev)
    return blocks[r].clone()


def all_to_all(pg: ProcessGroup, tensor: torch.Tensor) -> torch.Tensor:
    """Transpose data across ranks: rank ``i`` receives chunk ``i`` from everyone.

    ``tensor`` is treated as ``world_size`` equal chunks; chunk ``j`` is sent
    to rank ``j``, and the returned tensor's chunk ``i`` is what rank ``i``
    sent here. This is the exchange pattern behind expert (MoE) parallelism.
    """
    W, r = pg.world_size, pg.rank
    flat = _flat(tensor)
    if flat.numel() % W != 0:
        raise ValueError(f"numel {flat.numel()} not divisible by world_size {W}")
    chunk = flat.numel() // W
    out = torch.empty_like(flat)
    out[r * chunk : (r + 1) * chunk] = flat[r * chunk : (r + 1) * chunk]
    if W > 1:
        ev = pg.recorder.start(
            "all_to_all", "pairwise", channel=-1, nbytes=flat.numel() * flat.element_size()
        )
        try:
            # Distance-ordered pairwise exchange: at step d every rank talks
            # to exactly one sender and one receiver, so no rank is a hotspot.
            for d in range(1, W):
                dst, src = (r + d) % W, (r - d) % W
                pg.send_recv(
                    flat[dst * chunk : (dst + 1) * chunk],
                    dst,
                    out[src * chunk : (src + 1) * chunk],
                    src,
                )
        finally:
            pg.recorder.finish(ev)
    return out.view(tensor.shape)


def barrier(pg: ProcessGroup) -> None:
    """Block until every rank has entered the barrier."""
    if pg.world_size == 1:
        return
    ev = pg.recorder.start("barrier", "tree", channel=-1, nbytes=0)
    try:
        tensor = torch.zeros(1)
        _binomial_reduce(pg, tensor, "sum", root=0)
        _binomial_broadcast(pg, tensor, root=0)
    finally:
        pg.recorder.finish(ev)
