"""Tensor parallel correctness against an unsharded reference.

The bar: a layer split across W ranks must produce the same output *and the
same gradients* as the equivalent single-process layer holding the full
weights. Gradients matter as much as outputs here, because the whole design
rests on two autograd functions (identity-forward/all-reduce-backward and its
mirror) being placed correctly. Get one backwards and forward still looks
perfect while training silently diverges.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from mini_nccl import collectives
from mini_nccl.launcher import run
from mini_nccl.tensor_parallel import (
    ColumnParallelLinear,
    ParallelMLP,
    ParallelSelfAttention,
    RowParallelLinear,
    _qkv_shard_rows,
)

WIDTH = 16
BATCH = 4


def _reference_mlp(width: int, expansion: int = 4) -> nn.Module:
    torch.manual_seed(3)
    return nn.Sequential(
        nn.Linear(width, expansion * width),
        nn.GELU(),
        nn.Linear(expansion * width, width),
    )


def _mlp_worker(pg) -> None:
    reference = _reference_mlp(WIDTH)
    up_ref, down_ref = reference[0], reference[2]

    parallel = ParallelMLP(WIDTH, pg)
    parallel.up.load_full_weight(up_ref.weight.detach(), up_ref.bias.detach())
    parallel.down.load_full_weight(down_ref.weight.detach(), down_ref.bias.detach())

    # Each rank holds 1/W of both weight matrices.
    if pg.world_size > 1:
        assert parallel.up.weight.numel() < up_ref.weight.numel()
        assert parallel.down.weight.numel() < down_ref.weight.numel()

    gen = torch.Generator().manual_seed(5)
    x = torch.randn(BATCH, WIDTH, generator=gen)
    grad_out = torch.randn(BATCH, WIDTH, generator=gen)

    x_ref = x.clone().requires_grad_(True)
    y_ref = reference(x_ref)
    y_ref.backward(grad_out)

    x_par = x.clone().requires_grad_(True)
    y_par = parallel(x_par)
    y_par.backward(grad_out.clone())

    torch.testing.assert_close(y_par, y_ref, rtol=1e-5, atol=1e-6)
    # The input gradient is the one that needs the backward all-reduce.
    torch.testing.assert_close(x_par.grad, x_ref.grad, rtol=1e-5, atol=1e-6)

    # Each rank's weight gradient must equal the matching slice of the
    # reference gradient, with no averaging: it is already the global
    # gradient with respect to the slice this rank owns.
    up_start = parallel.up.shard_start
    up_stop = up_start + parallel.up.shard_size
    torch.testing.assert_close(
        parallel.up.weight.grad, up_ref.weight.grad[up_start:up_stop], rtol=1e-5, atol=1e-6
    )
    down_start = parallel.down.shard_start
    down_stop = down_start + parallel.down.shard_size
    torch.testing.assert_close(
        parallel.down.weight.grad,
        down_ref.weight.grad[:, down_start:down_stop],
        rtol=1e-5,
        atol=1e-6,
    )
    # Row-parallel bias is replicated, so its gradient is the full one.
    torch.testing.assert_close(
        parallel.down.bias.grad, down_ref.bias.grad, rtol=1e-5, atol=1e-6
    )


def test_parallel_mlp_matches_reference_world2() -> None:
    run(_mlp_worker, 2)


def test_parallel_mlp_matches_reference_world4() -> None:
    run(_mlp_worker, 4)


class _RefAttention(nn.Module):
    """Single-process causal attention, matching ParallelSelfAttention."""

    def __init__(self, width: int, n_head: int) -> None:
        super().__init__()
        torch.manual_seed(7)
        self.n_head = n_head
        self.head_dim = width // n_head
        self.qkv = nn.Linear(width, 3 * width)
        self.proj = nn.Linear(width, width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=-1)
        q, k, v = (
            t.view(B, T, self.n_head, self.head_dim).transpose(1, 2) for t in (q, k, v)
        )
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.proj(y.transpose(1, 2).reshape(B, T, C))


def _attention_worker(pg) -> None:
    n_head, seq = 4, 6
    reference = _RefAttention(WIDTH, n_head)
    parallel = ParallelSelfAttention(WIDTH, n_head, pg)
    parallel.qkv.load_full_weight(reference.qkv.weight.detach(), reference.qkv.bias.detach())
    parallel.proj.load_full_weight(reference.proj.weight.detach(), reference.proj.bias.detach())

    assert parallel.local_heads == n_head // pg.world_size

    gen = torch.Generator().manual_seed(9)
    x = torch.randn(2, seq, WIDTH, generator=gen)
    grad_out = torch.randn(2, seq, WIDTH, generator=gen)

    x_ref = x.clone().requires_grad_(True)
    reference(x_ref).backward(grad_out)

    x_par = x.clone().requires_grad_(True)
    y_par = parallel(x_par)
    y_par.backward(grad_out.clone())

    torch.testing.assert_close(x_par.grad, x_ref.grad, rtol=1e-4, atol=1e-6)
    # The fused QKV gradient must line up with the interleaved head rows.
    rows = _qkv_shard_rows(WIDTH, n_head, WIDTH // n_head, pg)
    torch.testing.assert_close(
        parallel.qkv.weight.grad,
        reference.qkv.weight.grad.index_select(0, rows),
        rtol=1e-4,
        atol=1e-6,
    )


def test_parallel_attention_matches_reference() -> None:
    run(_attention_worker, 2)
    run(_attention_worker, 4)


def _gather_output_worker(pg) -> None:
    """gather_output=True must reproduce the full unsharded output."""
    # Seed before constructing: every rank must build the *same* reference,
    # since torch's default init differs per process and each rank slices
    # this weight to load its own shard.
    torch.manual_seed(21)
    reference = nn.Linear(WIDTH, 3 * WIDTH)
    torch.nn.init.normal_(reference.weight, std=0.05)
    layer = ColumnParallelLinear(WIDTH, 3 * WIDTH, pg, gather_output=True)
    layer.load_full_weight(reference.weight.detach(), reference.bias.detach())

    gen = torch.Generator().manual_seed(13)
    x = torch.randn(BATCH, WIDTH, generator=gen)
    grad_out = torch.randn(BATCH, 3 * WIDTH, generator=gen)

    x_ref = x.clone().requires_grad_(True)
    reference(x_ref).backward(grad_out)
    x_par = x.clone().requires_grad_(True)
    layer(x_par).backward(grad_out.clone())

    torch.testing.assert_close(x_par.grad, x_ref.grad, rtol=1e-5, atol=1e-6)


def test_column_parallel_gather_output() -> None:
    run(_gather_output_worker, 4)


def _replica_drift_worker(pg, steps: int = 8) -> None:
    """Replicated layers beside tensor-parallel ones must not drift apart.

    A tensor-parallel model has no data-parallel gradient sync, so replicated
    layers stay in step only because every rank receives an identical
    activation gradient from the backward all-reduce. That holds bitwise:
    ring and tree all-reduce each reduce a given element on one rank and copy
    the result, rather than each rank summing in its own order. If that were
    not true, replicas would diverge slowly and silently, which is exactly
    the class of bug worth a regression test.
    """
    torch.manual_seed(31)
    norm = nn.LayerNorm(WIDTH)  # replicated
    mlp = ParallelMLP(WIDTH, pg)  # sharded
    opt = torch.optim.SGD(list(norm.parameters()) + list(mlp.parameters()), lr=0.1)

    gen = torch.Generator().manual_seed(37)
    for _ in range(steps):
        x = torch.randn(BATCH, WIDTH, generator=gen)
        opt.zero_grad()
        mlp(norm(x)).square().mean().backward()
        opt.step()

    # Exact equality, not approximate: any drift at all is a bug.
    for name, param in norm.named_parameters():
        gathered = collectives.all_gather(pg, param.detach())
        for rank, other in enumerate(gathered):
            assert torch.equal(other, gathered[0]), (
                f"replicated {name} diverged on rank {rank}"
            )


def test_replicated_layers_stay_bitwise_identical() -> None:
    run(_replica_drift_worker, 4)


def _unsharded_input_worker(pg) -> None:
    """RowParallelLinear can take a full input and slice it locally."""
    torch.manual_seed(23)
    reference = nn.Linear(4 * WIDTH, WIDTH)
    layer = RowParallelLinear(4 * WIDTH, WIDTH, pg, input_is_sharded=False)
    layer.load_full_weight(reference.weight.detach(), reference.bias.detach())

    gen = torch.Generator().manual_seed(17)
    x = torch.randn(BATCH, 4 * WIDTH, generator=gen)
    x_ref = x.clone().requires_grad_(True)
    x_par = x.clone().requires_grad_(True)
    torch.testing.assert_close(layer(x_par), reference(x_ref), rtol=1e-5, atol=1e-6)


def test_row_parallel_unsharded_input() -> None:
    run(_unsharded_input_worker, 2)
