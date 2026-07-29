"""Sub-groups, and composing two parallelism strategies at once.

The claim being tested is that the collectives were written against a small
enough interface that a rank-translating view is a drop-in substitute for a
process group. If that holds, ring all-reduce and DDP work on a subgroup with
no changes, and 2D parallelism is just two subgroups used for different things.

The parity test is the real one: a model whose layers are split across the
tensor dimension and whose gradients are averaged across the data dimension
must produce the same gradients as one process training the whole model on the
whole batch.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from mini_nccl import collectives as c
from mini_nccl.ddp import DistributedDataParallel
from mini_nccl.launcher import run
from mini_nccl.mesh import ParallelMesh, SubGroup
from mini_nccl.tensor_parallel import ParallelMLP

WIDTH = 12


# ---- mesh arithmetic, no communication ----------------------------------


def _layout_worker(pg) -> dict:
    mesh = ParallelMesh(pg, dp=2, tp=2)
    return {
        "rank": pg.rank,
        "dp": mesh.coordinate("dp"),
        "tp": mesh.coordinate("tp"),
        "tp_ranks": mesh.ranks_along("tp"),
        "dp_ranks": mesh.ranks_along("dp"),
    }


def test_mesh_layout_puts_neighbours_in_the_same_tensor_group() -> None:
    reports = {r["rank"]: r for r in run(_layout_worker, 4)}

    # Last dimension fastest: tensor groups are contiguous rank runs.
    assert reports[0]["tp_ranks"] == [0, 1]
    assert reports[1]["tp_ranks"] == [0, 1]
    assert reports[2]["tp_ranks"] == [2, 3]
    # Data-parallel groups stride across the tensor groups.
    assert reports[0]["dp_ranks"] == [0, 2]
    assert reports[1]["dp_ranks"] == [1, 3]
    # Coordinates decompose the rank.
    for rank, report in reports.items():
        assert report["dp"] * 2 + report["tp"] == rank

    # The two partitions are orthogonal, which is why they never share a socket:
    # any two ranks share at most one dimension's group.
    for rank, report in reports.items():
        overlap = set(report["tp_ranks"]) & set(report["dp_ranks"])
        assert overlap == {rank}, (rank, overlap)


def _mesh_errors_worker(pg) -> None:
    with pytest.raises(ValueError, match="slots but the group has"):
        ParallelMesh(pg, dp=3, tp=3)
    with pytest.raises(ValueError, match="at least one dimension"):
        ParallelMesh(pg)
    mesh = ParallelMesh(pg, dp=2, tp=2)
    with pytest.raises(KeyError, match="no dimension"):
        mesh.ranks_along("pp")
    with pytest.raises(ValueError, match="not a member"):
        SubGroup(pg, [r for r in range(4) if r != pg.rank])


def test_mesh_errors_are_clear() -> None:
    run(_mesh_errors_worker, 4)


# ---- collectives on a subgroup ------------------------------------------


def _subgroup_collective_worker(pg) -> None:
    """A ring all-reduce over half the ranks must only see that half."""
    mesh = ParallelMesh(pg, dp=2, tp=2)
    tp = mesh.group("tp")
    dp = mesh.group("dp")
    assert tp.world_size == 2 and dp.world_size == 2

    # Sum over the tensor group only: members contribute their global rank.
    x = torch.full((1024,), float(pg.rank))
    c.all_reduce(tp, x, algorithm="ring")
    expected_tp = float(sum(mesh.ranks_along("tp")))
    torch.testing.assert_close(x, torch.full((1024,), expected_tp))

    # And again over the data group, which is a different pair of ranks.
    y = torch.full((1024,), float(pg.rank))
    c.all_reduce(dp, y, algorithm="tree")
    expected_dp = float(sum(mesh.ranks_along("dp")))
    torch.testing.assert_close(y, torch.full((1024,), expected_dp))

    # Non-all-reduce collectives translate ranks too.
    gathered = c.all_gather(tp, torch.tensor([float(pg.rank)]))
    assert [g.item() for g in gathered] == [float(r) for r in mesh.ranks_along("tp")]
    c.broadcast(tp, x, src=0)
    torch.testing.assert_close(x, torch.full((1024,), expected_tp))
    c.barrier(dp)


def test_collectives_run_unchanged_on_a_subgroup() -> None:
    run(_subgroup_collective_worker, 4)


def _degenerate_worker(pg) -> None:
    """A dimension of size 1 must behave like a no-op group, not a special case."""
    mesh = ParallelMesh(pg, dp=4, tp=1)
    tp = mesh.group("tp")
    assert tp.world_size == 1
    x = torch.full((8,), float(pg.rank))
    c.all_reduce(tp, x)  # group of one: unchanged
    torch.testing.assert_close(x, torch.full((8,), float(pg.rank)))
    c.all_reduce(mesh.group("dp"), x)
    torch.testing.assert_close(x, torch.full((8,), float(sum(range(4)))))


def test_single_rank_dimension_is_a_noop() -> None:
    run(_degenerate_worker, 4)


# ---- 2D: tensor parallel inside data parallel ---------------------------


class Model(nn.Module):
    """An embedding, a tensor-parallel MLP, and a head."""

    def __init__(self, tp_group) -> None:
        super().__init__()
        torch.manual_seed(8)
        self.embed = nn.Linear(WIDTH, WIDTH)
        self.mlp = ParallelMLP(WIDTH, tp_group, expansion=2)
        self.head = nn.Linear(WIDTH, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.mlp(torch.tanh(self.embed(x))))


class ReferenceModel(nn.Module):
    """The same computation, unsharded, in one process."""

    def __init__(self) -> None:
        super().__init__()
        torch.manual_seed(8)
        self.embed = nn.Linear(WIDTH, WIDTH)
        self.up = nn.Linear(WIDTH, 2 * WIDTH)
        self.down = nn.Linear(2 * WIDTH, WIDTH)
        self.head = nn.Linear(WIDTH, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.tanh(self.embed(x))
        h = self.down(F.gelu(self.up(h)))
        return self.head(h)


def _two_dimensional_worker(pg, steps: int = 4) -> None:
    mesh = ParallelMesh(pg, dp=2, tp=2)
    tp, dp = mesh.group("tp"), mesh.group("dp")

    reference = ReferenceModel()
    model = Model(tp)
    # Copy every weight across explicitly rather than relying on a shared seed:
    # the two models draw from the RNG in a different order, so equal seeds do
    # not imply equal initial weights.
    with torch.no_grad():
        for name in ("embed", "head"):
            getattr(model, name).weight.copy_(getattr(reference, name).weight)
            getattr(model, name).bias.copy_(getattr(reference, name).bias)
    model.mlp.up.load_full_weight(reference.up.weight.detach(), reference.up.bias.detach())
    model.mlp.down.load_full_weight(
        reference.down.weight.detach(), reference.down.bias.detach()
    )

    # Gradients are averaged across the data dimension only: each tensor-parallel
    # rank owns a different shard, so there is nothing to reduce along tp.
    ddp = DistributedDataParallel(model, dp)
    opt = torch.optim.SGD(model.parameters(), lr=0.05)
    ref_opt = torch.optim.SGD(reference.parameters(), lr=0.05)

    per_replica = 5
    gen = torch.Generator().manual_seed(29)
    for _ in range(steps):
        x = torch.randn(mesh.size("dp") * per_replica, WIDTH, generator=gen)
        y = torch.randn(mesh.size("dp") * per_replica, 1, generator=gen)
        # Replicas split the batch; ranks within a tensor group share a shard.
        start = mesh.coordinate("dp") * per_replica
        shard = slice(start, start + per_replica)

        ddp.zero_grad()
        F.mse_loss(ddp(x[shard]), y[shard]).backward()
        ddp.sync()
        opt.step()

        ref_opt.zero_grad()
        F.mse_loss(reference(x), y).backward()
        ref_opt.step()

    # Replicated layers must match the reference outright.
    for name in ("embed", "head"):
        local = getattr(model, name)
        ref = getattr(reference, name)
        torch.testing.assert_close(local.weight, ref.weight, rtol=1e-4, atol=1e-5)
        torch.testing.assert_close(local.bias, ref.bias, rtol=1e-4, atol=1e-5)

    # Sharded layers must match their slice of it.
    up_start = model.mlp.up.shard_start
    up_stop = up_start + model.mlp.up.shard_size
    torch.testing.assert_close(
        model.mlp.up.weight, reference.up.weight[up_start:up_stop], rtol=1e-4, atol=1e-5
    )
    down_start = model.mlp.down.shard_start
    down_stop = down_start + model.mlp.down.shard_size
    torch.testing.assert_close(
        model.mlp.down.weight,
        reference.down.weight[:, down_start:down_stop],
        rtol=1e-4,
        atol=1e-5,
    )


def test_tensor_parallel_inside_data_parallel_matches_single_process() -> None:
    run(_two_dimensional_worker, 4)
