"""Tensor parallelism: one layer's weights split across ranks.

Data parallelism replicates the model and splits the batch. Tensor
parallelism splits the *layer*, so a matrix multiply too large for one device
runs as several smaller ones. Megatron's insight is that a pair of linear
layers can be split so the whole pair needs exactly one collective in
forward and one in backward.

Take an MLP ``y = B · f(A · x)``:

- **Column parallel** on ``A``: rank ``r`` holds a slice of ``A``'s *output*
  rows, so it computes a slice of ``f(A · x)`` from the full ``x``. No
  communication in forward. Each rank's ``dL/dx`` is only its own partial
  contribution, so backward must **all-reduce** the input gradient.
- **Row parallel** on ``B``: rank ``r`` holds a slice of ``B``'s *input*
  columns, matching the slice of activations it already has, so it computes a
  partial sum of the output. Forward must **all-reduce** those partials.
  Backward needs no communication, because each rank's activation gradient
  slice is exactly the one it owns.

Chained, the sharded activation never has to be gathered: column parallel
hands its slice straight to row parallel. The two collectives are the
identity-forward/all-reduce-backward pair and its mirror, which Megatron
calls ``f`` and ``g`` and which are the only two autograd functions needed.

Usage::

    class ParallelMLP(nn.Module):
        def __init__(self, width, pg):
            super().__init__()
            self.up = ColumnParallelLinear(width, 4 * width, pg)
            self.down = RowParallelLinear(4 * width, width, pg)

        def forward(self, x):
            return self.down(torch.nn.functional.gelu(self.up(x)))

Every rank must call these layers in the same order, as always. Each rank
holds ``1/W`` of the layer's weights, and gradients need no averaging: they
are already the gradient of the global loss with respect to the slice this
rank owns.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from . import collectives
from .communicator import Communicator


class _IdentityToAllReduce(torch.autograd.Function):
    """Megatron's ``f``: identity forward, all-reduce backward.

    Placed at a column-parallel layer's input. Each rank computes a partial
    ``dL/dx`` from its own weight slice; the true gradient is their sum.
    """

    @staticmethod
    def forward(ctx, pg: Communicator, x: torch.Tensor):
        ctx.pg = pg
        return x

    @staticmethod
    def backward(ctx, grad: torch.Tensor):
        pg: Communicator = ctx.pg
        if pg.world_size > 1:
            grad = grad.contiguous()
            collectives.all_reduce(pg, grad)
        return None, grad


class _AllReduceToIdentity(torch.autograd.Function):
    """Megatron's ``g``: all-reduce forward, identity backward.

    Placed at a row-parallel layer's output, where each rank holds a partial
    sum. Backward needs nothing: each rank's incoming gradient is already the
    gradient of its own partial contribution.
    """

    @staticmethod
    def forward(ctx, pg: Communicator, x: torch.Tensor):
        if pg.world_size > 1:
            x = x.contiguous()
            collectives.all_reduce(pg, x)
        return x

    @staticmethod
    def backward(ctx, grad: torch.Tensor):
        return None, grad


class _GatherFromRanks(torch.autograd.Function):
    """Concatenate feature shards in forward, slice the gradient in backward."""

    @staticmethod
    def forward(ctx, pg: Communicator, x: torch.Tensor):
        ctx.pg = pg
        ctx.width = x.shape[-1]
        if pg.world_size == 1:
            return x
        return torch.cat(collectives.all_gather(pg, x.contiguous()), dim=-1)

    @staticmethod
    def backward(ctx, grad: torch.Tensor):
        pg: Communicator = ctx.pg
        if pg.world_size == 1:
            return None, grad
        start = pg.rank * ctx.width
        return None, grad[..., start : start + ctx.width].contiguous()


def _shard_size(total: int, world_size: int, rank: int) -> tuple[int, int]:
    """Even split, with the remainder spread over the leading ranks."""
    base, extra = divmod(total, world_size)
    size = base + (1 if rank < extra else 0)
    start = rank * base + min(rank, extra)
    return start, size


class ColumnParallelLinear(nn.Module):
    """``y = x A^T + b`` with ``A`` split along its output features.

    Returns this rank's slice of the output unless ``gather_output=True``.

    ``shard_rows`` overrides the default contiguous split with an explicit
    list of output rows. A fused QKV projection needs this: its output is
    ``[all q | all k | all v]``, so a contiguous split would hand rank 0 every
    q head plus part of k, which is not a valid attention shard. The right
    slice is *this rank's heads* of each of q, k, and v.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        pg: Communicator,
        bias: bool = True,
        gather_output: bool = False,
        shard_rows: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.pg = pg
        self.in_features = in_features
        self.out_features = out_features
        self.gather_output = gather_output
        if shard_rows is None:
            self.shard_start, self.shard_size = _shard_size(
                out_features, pg.world_size, pg.rank
            )
            self.shard_rows = None
        else:
            self.register_buffer("shard_rows", shard_rows, persistent=False)
            self.shard_start, self.shard_size = -1, int(shard_rows.numel())
        self.weight = nn.Parameter(torch.empty(self.shard_size, in_features))
        self.bias = nn.Parameter(torch.zeros(self.shard_size)) if bias else None
        nn.init.normal_(self.weight, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = _IdentityToAllReduce.apply(self.pg, x)
        y = F.linear(x, self.weight, self.bias)
        if self.gather_output:
            if self.shard_rows is not None:
                raise RuntimeError("gather_output assumes a contiguous output shard")
            y = _GatherFromRanks.apply(self.pg, y)
        return y

    @torch.no_grad()
    def load_full_weight(self, weight: torch.Tensor, bias: torch.Tensor | None = None) -> None:
        """Take this rank's slice out of an unsharded weight (for tests/loading)."""
        if self.shard_rows is not None:
            self.weight.copy_(weight.index_select(0, self.shard_rows))
            if self.bias is not None and bias is not None:
                self.bias.copy_(bias.index_select(0, self.shard_rows))
            return
        stop = self.shard_start + self.shard_size
        self.weight.copy_(weight[self.shard_start : stop])
        if self.bias is not None and bias is not None:
            self.bias.copy_(bias[self.shard_start : stop])


class RowParallelLinear(nn.Module):
    """``y = x A^T + b`` with ``A`` split along its input features.

    Expects ``x`` already sharded along features (the output of a
    column-parallel layer) unless ``input_is_sharded=False``, in which case
    this rank's slice is taken locally.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        pg: Communicator,
        bias: bool = True,
        input_is_sharded: bool = True,
    ) -> None:
        super().__init__()
        self.pg = pg
        self.in_features = in_features
        self.out_features = out_features
        self.input_is_sharded = input_is_sharded
        self.shard_start, self.shard_size = _shard_size(in_features, pg.world_size, pg.rank)
        self.weight = nn.Parameter(torch.empty(out_features, self.shard_size))
        # The bias is added once, after the partial sums are reduced, so it
        # lives unsharded on every rank.
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        nn.init.normal_(self.weight, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.input_is_sharded:
            stop = self.shard_start + self.shard_size
            x = x[..., self.shard_start : stop]
        partial = F.linear(x, self.weight, None)
        y = _AllReduceToIdentity.apply(self.pg, partial)
        if self.bias is not None:
            y = y + self.bias
        return y

    @torch.no_grad()
    def load_full_weight(self, weight: torch.Tensor, bias: torch.Tensor | None = None) -> None:
        stop = self.shard_start + self.shard_size
        self.weight.copy_(weight[:, self.shard_start : stop])
        if self.bias is not None and bias is not None:
            self.bias.copy_(bias)


class ParallelMLP(nn.Module):
    """The transformer MLP, tensor-parallel: one all-reduce each way.

    ``up`` is column parallel and ``down`` is row parallel, so the 4x-wide
    hidden activation is never gathered. It stays sharded from the moment it
    is produced to the moment it is consumed.
    """

    def __init__(self, width: int, pg: Communicator, expansion: int = 4) -> None:
        super().__init__()
        self.up = ColumnParallelLinear(width, expansion * width, pg)
        self.down = RowParallelLinear(expansion * width, width, pg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.gelu(self.up(x)))


def _qkv_shard_rows(width: int, n_head: int, head_dim: int, pg: Communicator) -> torch.Tensor:
    """Rows of a fused QKV weight belonging to this rank.

    The fused weight is laid out ``[q(width) | k(width) | v(width)]`` and each
    third is ``n_head`` blocks of ``head_dim``. This rank owns the same head
    range in all three, so the rows come out as
    ``q[heads] ++ k[heads] ++ v[heads]``, which is why the forward can simply
    split its local output into three equal parts.
    """
    heads_per_rank = n_head // pg.world_size
    first = pg.rank * heads_per_rank
    rows: list[int] = []
    for third in range(3):
        for head in range(first, first + heads_per_rank):
            base = third * width + head * head_dim
            rows.extend(range(base, base + head_dim))
    return torch.tensor(rows, dtype=torch.long)


class ParallelSelfAttention(nn.Module):
    """Causal self-attention split across ranks by attention head.

    Heads are independent, which is what makes attention such a natural fit:
    the QKV projection is column parallel (each rank gets whole heads) and the
    output projection is row parallel, so a rank computes its own heads end to
    end and the single all-reduce at the output combines them.
    """

    def __init__(self, width: int, n_head: int, pg: Communicator) -> None:
        super().__init__()
        if n_head % pg.world_size:
            raise ValueError(f"n_head={n_head} must be divisible by world_size={pg.world_size}")
        if width % n_head:
            raise ValueError(f"width={width} must be divisible by n_head={n_head}")
        self.pg = pg
        self.n_head = n_head
        self.local_heads = n_head // pg.world_size
        self.head_dim = width // n_head
        self.qkv = ColumnParallelLinear(
            width,
            3 * width,
            pg,
            shard_rows=_qkv_shard_rows(width, n_head, width // n_head, pg),
        )
        self.proj = RowParallelLinear(width, width, pg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        local_width = self.local_heads * self.head_dim
        qkv = self.qkv(x)
        # The column-parallel shard interleaves q, k, v thirds of *this rank's*
        # slice, so split by three before reshaping into heads.
        q, k, v = qkv.split(local_width, dim=-1)
        q, k, v = (
            t.view(B, T, self.local_heads, self.head_dim).transpose(1, 2) for t in (q, k, v)
        )
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).reshape(B, T, local_width)
        return self.proj(y)
