"""Collective communication algorithms.

Three all-reduce strategies, mirroring the trade-offs inside NCCL:

- ``ring``   — bandwidth-optimal. Reduce-scatter then all-gather around a
  ring; each rank sends ``2(W-1)/W * n`` bytes total regardless of world
  size, but latency grows linearly with ``W`` (2(W-1) serialized steps).
- ``tree``   — latency-optimal. Binomial-tree reduce to rank 0 followed by
  a binomial-tree broadcast: ``2*ceil(log2 W)`` steps, but interior ranks
  forward the full payload, so it moves more bytes than ring on large
  tensors.
- ``naive``  — the parameter-server baseline: every rank sends to rank 0,
  which reduces and sends the result back. Rank 0 handles ``O(W * n)``
  bytes; included so benchmarks show *why* the other two exist.

``auto`` picks tree for small messages (latency-bound) and ring for large
ones (bandwidth-bound), like NCCL's algorithm selection.
"""

from __future__ import annotations

import torch

from .process_group import ProcessGroup

# Crossover point between latency-bound and bandwidth-bound messages.
# Tune this from benchmarks/bench_allreduce.py results on your own fabric —
# on loopback TCP it sits around a few hundred KiB.
RING_THRESHOLD_BYTES = 256 * 1024

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


def all_reduce(
    pg: ProcessGroup,
    tensor: torch.Tensor,
    op: str = "sum",
    algorithm: str = "auto",
) -> torch.Tensor:
    """In-place all-reduce of ``tensor`` across all ranks."""
    if op not in _OPS:
        raise ValueError(f"unknown op {op!r}; expected one of {sorted(_OPS)}")
    if pg.world_size == 1:
        return tensor
    if algorithm == "auto":
        nbytes = tensor.numel() * tensor.element_size()
        algorithm = "tree" if nbytes <= RING_THRESHOLD_BYTES else "ring"
    if algorithm == "ring":
        _ring_all_reduce(pg, tensor, op)
    elif algorithm == "tree":
        _tree_all_reduce(pg, tensor, op)
    elif algorithm == "naive":
        _naive_all_reduce(pg, tensor, op)
    else:
        raise ValueError(f"unknown algorithm {algorithm!r}")
    return tensor


def _ring_all_reduce(pg: ProcessGroup, tensor: torch.Tensor, op: str) -> None:
    W, r = pg.world_size, pg.rank
    flat = _flat(tensor)
    chunk = -(-flat.numel() // W)  # ceil division

    padded = flat
    if chunk * W != flat.numel():
        padded = torch.zeros(chunk * W, dtype=flat.dtype)
        padded[: flat.numel()] = flat
    blocks = padded.view(W, chunk)

    tmp = torch.empty(chunk, dtype=flat.dtype)
    right, left = (r + 1) % W, (r - 1) % W
    reduce_op = _OPS[op]

    # Phase 1 — reduce-scatter: after W-1 steps, this rank holds the fully
    # reduced block (r + 1) % W. Each step forwards the block we just
    # accumulated while receiving the next partial from the left.
    for step in range(W - 1):
        send_idx = (r - step) % W
        recv_idx = (r - step - 1) % W
        pg.send_recv(blocks[send_idx], right, tmp, left)
        reduce_op(blocks[recv_idx], tmp)

    # Phase 2 — all-gather: circulate the reduced blocks around the ring.
    for step in range(W - 1):
        send_idx = (r + 1 - step) % W
        recv_idx = (r - step) % W
        pg.send_recv(blocks[send_idx], right, blocks[recv_idx], left)

    if padded.data_ptr() != flat.data_ptr():
        flat.copy_(padded[: flat.numel()])


def _tree_all_reduce(pg: ProcessGroup, tensor: torch.Tensor, op: str) -> None:
    _binomial_reduce(pg, tensor, op, root=0)
    _binomial_broadcast(pg, tensor, root=0)


def _binomial_reduce(pg: ProcessGroup, tensor: torch.Tensor, op: str, root: int) -> None:
    """Reduce onto ``root`` along a binomial tree (works for any world size)."""
    W, r = pg.world_size, pg.rank
    vr = (r - root) % W  # virtual rank: the tree is always rooted at vr 0
    flat = _flat(tensor)
    tmp = torch.empty_like(flat)
    reduce_op = _OPS[op]

    mask = 1
    while mask < W:
        if vr & mask:
            dst = ((vr & ~mask) + root) % W
            pg.send(tensor, dst)
            return  # non-roots send exactly once, then are done
        src = vr | mask
        if src < W:
            pg.recv(tmp, (src + root) % W)
            reduce_op(flat, tmp)
        mask <<= 1


def _binomial_broadcast(pg: ProcessGroup, tensor: torch.Tensor, root: int) -> None:
    W, r = pg.world_size, pg.rank
    if W == 1:
        return
    vr = (r - root) % W
    if vr != 0:
        lowbit = vr & -vr
        pg.recv(tensor, ((vr - lowbit) + root) % W)
        mask = lowbit >> 1
    else:
        mask = 1 << ((W - 1).bit_length() - 1)
    while mask:
        if vr + mask < W:
            pg.send(tensor, ((vr + mask) + root) % W)
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
    if pg.world_size > 1:
        _binomial_broadcast(pg, tensor, root=src)
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
        right, left = (r + 1) % W, (r - 1) % W
        for step in range(W - 1):
            send_idx = (r - step) % W
            recv_idx = (r - step - 1) % W
            pg.send_recv(out[send_idx], right, out[recv_idx], left)
    return [out[i].view(tensor.shape).clone() for i in range(W)]


def reduce_scatter(pg: ProcessGroup, tensor: torch.Tensor, op: str = "sum") -> torch.Tensor:
    """Reduce across ranks and return this rank's 1/W-th of the result.

    ``tensor.numel()`` must be divisible by ``world_size`` (NCCL contract).
    The input is left unmodified.
    """
    if op not in _OPS:
        raise ValueError(f"unknown op {op!r}; expected one of {sorted(_OPS)}")
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
    # Same schedule as the ring reduce-scatter phase, shifted by one so the
    # fully reduced block each rank ends up holding is its own index.
    for step in range(W - 1):
        send_idx = (r - step - 1) % W
        recv_idx = (r - step - 2) % W
        pg.send_recv(blocks[send_idx], right, tmp, left)
        reduce_op(blocks[recv_idx], tmp)
    return blocks[r].clone()


def barrier(pg: ProcessGroup) -> None:
    """Block until every rank has entered the barrier."""
    all_reduce(pg, torch.zeros(1), algorithm="tree")
