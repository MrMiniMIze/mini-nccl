"""All three dimensions at once: data x pipeline x tensor.

Eight ranks as a 2x2x2 mesh. Each rank holds one stage of the model, one shard
of that stage's layers, and belongs to one of two replicas. This is the shape
real large-model training uses, and it is the strongest statement the project
can make: gradients from a model split three ways must equal the gradients of
the whole model trained in one process on the whole batch.

Two things are worth stating about how the dimensions interact, because both
are easy to get wrong and neither shows up as a crash:

- **Gradients are averaged along the data dimension only.** Tensor-parallel
  ranks own different slices, so there is nothing to average along that axis,
  and pipeline stages own different layers entirely.
- **The reduction happens after the pipeline drains, not in a backward hook.**
  Pipeline parallelism runs backward once per microbatch, so DDP's hook would
  fire ``M`` times per step and reduce partial gradients. ``average_gradients``
  is the explicit form for exactly this case.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from mini_nccl.ddp import average_gradients
from mini_nccl.launcher import run
from mini_nccl.mesh import ParallelMesh
from mini_nccl.pipeline import PipelineParallel
from mini_nccl.tensor_parallel import ColumnParallelLinear, RowParallelLinear

WIDTH = 8
EXPANSION = 2
PER_REPLICA = 4
MICROBATCHES = 2


class ReferenceBlock(nn.Module):
    """One block, unsharded: the thing tensor parallelism will split."""

    def __init__(self) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(WIDTH)
        self.up = nn.Linear(WIDTH, EXPANSION * WIDTH)
        self.down = nn.Linear(EXPANSION * WIDTH, WIDTH)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.down(F.gelu(self.up(self.norm(x))))


class ReferenceModel(nn.Module):
    """embed -> block0 -> block1 -> head, all in one process."""

    def __init__(self) -> None:
        super().__init__()
        torch.manual_seed(12)
        self.embed = nn.Linear(WIDTH, WIDTH)
        self.block0 = ReferenceBlock()
        self.block1 = ReferenceBlock()
        self.head = nn.Linear(WIDTH, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.block1(self.block0(torch.tanh(self.embed(x)))))


class ShardedBlock(nn.Module):
    """The same block with its MLP split across the tensor dimension."""

    def __init__(self, tp_group) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(WIDTH)
        self.up = ColumnParallelLinear(WIDTH, EXPANSION * WIDTH, tp_group)
        self.down = RowParallelLinear(EXPANSION * WIDTH, WIDTH, tp_group)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.down(F.gelu(self.up(self.norm(x))))


class FirstStage(nn.Module):
    def __init__(self, tp_group) -> None:
        super().__init__()
        self.embed = nn.Linear(WIDTH, WIDTH)
        self.block = ShardedBlock(tp_group)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(torch.tanh(self.embed(x)))


class LastStage(nn.Module):
    def __init__(self, tp_group) -> None:
        super().__init__()
        self.block = ShardedBlock(tp_group)
        self.head = nn.Linear(WIDTH, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.block(x))


def _load_from_reference(stage: nn.Module, reference: ReferenceModel, pp_rank: int) -> None:
    """Copy this rank's stage and tensor shard out of the reference model."""
    ref_block = reference.block0 if pp_rank == 0 else reference.block1
    with torch.no_grad():
        stage.block.norm.weight.copy_(ref_block.norm.weight)
        stage.block.norm.bias.copy_(ref_block.norm.bias)
        stage.block.up.load_full_weight(
            ref_block.up.weight.detach(), ref_block.up.bias.detach()
        )
        stage.block.down.load_full_weight(
            ref_block.down.weight.detach(), ref_block.down.bias.detach()
        )
        if pp_rank == 0:
            stage.embed.weight.copy_(reference.embed.weight)
            stage.embed.bias.copy_(reference.embed.bias)
        else:
            stage.head.weight.copy_(reference.head.weight)
            stage.head.bias.copy_(reference.head.bias)


def _three_dimensional_worker(pg) -> dict:
    mesh = ParallelMesh(pg, dp=2, pp=2, tp=2)
    tp, pp, dp = mesh.group("tp"), mesh.group("pp"), mesh.group("dp")
    pp_rank = mesh.coordinate("pp")

    reference = ReferenceModel()
    stage = FirstStage(tp) if pp_rank == 0 else LastStage(tp)
    _load_from_reference(stage, reference, pp_rank)

    pipeline = PipelineParallel(
        stage,
        pg=pp,
        activation_shape=(PER_REPLICA // MICROBATCHES, WIDTH),
        loss_fn=F.mse_loss,
        schedule="1f1b",
    )

    # The global batch is split across replicas; ranks sharing a replica see
    # the same data regardless of their stage or tensor shard.
    gen = torch.Generator().manual_seed(31)
    global_batch = mesh.size("dp") * PER_REPLICA
    x = torch.randn(global_batch, WIDTH, generator=gen)
    y = torch.randn(global_batch, 1, generator=gen)
    start = mesh.coordinate("dp") * PER_REPLICA
    shard = slice(start, start + PER_REPLICA)

    for p in stage.parameters():
        p.grad = None
    loss = pipeline.step(
        inputs=x[shard] if pp_rank == 0 else None,
        targets=y[shard] if pp_rank == mesh.size("pp") - 1 else None,
        n_microbatches=MICROBATCHES,
    )
    # Averaged along the data dimension only, and only once the pipeline has
    # drained, since backward ran once per microbatch.
    reduced_bytes = average_gradients(dp, list(stage.parameters()))

    # Reference: the whole model, the whole batch, one process.
    reference.zero_grad()
    reference_loss = F.mse_loss(reference(x), y)
    reference_loss.backward()

    ref_block = reference.block0 if pp_rank == 0 else reference.block1
    up_start = stage.block.up.shard_start
    up_stop = up_start + stage.block.up.shard_size
    down_start = stage.block.down.shard_start
    down_stop = down_start + stage.block.down.shard_size

    checks = {
        "norm.weight": (stage.block.norm.weight.grad, ref_block.norm.weight.grad),
        "up.weight": (stage.block.up.weight.grad, ref_block.up.weight.grad[up_start:up_stop]),
        "down.weight": (
            stage.block.down.weight.grad,
            ref_block.down.weight.grad[:, down_start:down_stop],
        ),
        # Row-parallel bias is replicated, so its gradient is the whole thing.
        "down.bias": (stage.block.down.bias.grad, ref_block.down.bias.grad),
    }
    if pp_rank == 0:
        checks["embed.weight"] = (stage.embed.weight.grad, reference.embed.weight.grad)
    else:
        checks["head.weight"] = (stage.head.weight.grad, reference.head.weight.grad)

    for name, (got, expected) in checks.items():
        assert got is not None, f"rank {pg.rank}: {name} got no gradient"
        torch.testing.assert_close(
            got, expected, rtol=1e-4, atol=1e-6,
            msg=lambda m, name=name: f"rank {pg.rank} {name}: {m}",
        )

    return {
        "rank": pg.rank,
        "dp": mesh.coordinate("dp"),
        "pp": pp_rank,
        "tp": mesh.coordinate("tp"),
        "loss": loss,
        "reference_loss": float(reference_loss.detach()),
        "reduced_bytes": reduced_bytes,
        "local_params": sum(p.numel() for p in stage.parameters()),
        "reference_params": sum(p.numel() for p in reference.parameters()),
    }


def test_three_dimensional_matches_single_process() -> None:
    reports = {r["rank"]: r for r in run(_three_dimensional_worker, 8, timeout=300.0)}
    assert len(reports) == 8

    # The mesh should decompose every rank consistently.
    for rank, report in reports.items():
        assert report["dp"] * 4 + report["pp"] * 2 + report["tp"] == rank, report

    # Each rank holds a fraction of the model: one stage of two, and within it
    # the MLP split two ways.
    for report in reports.values():
        assert report["local_params"] < report["reference_params"] / 2, report

    # Only the last stage produces a loss, and both replicas' last stages should
    # land near the reference (they each saw half the batch, so not identical).
    last_stage = [r for r in reports.values() if r["pp"] == 1]
    assert len(last_stage) == 4
    for report in last_stage:
        assert report["loss"] > 0, report

    # Ranks sharing a replica and stage differ only in tensor shard, so their
    # losses must agree exactly: tensor parallelism is mathematically replicated.
    for dp_index in (0, 1):
        pair = [r["loss"] for r in reports.values() if r["dp"] == dp_index and r["pp"] == 1]
        assert pair[0] == pair[1], (dp_index, pair)

    # Something was actually reduced along the data dimension.
    assert all(r["reduced_bytes"] > 0 for r in reports.values())


def _degenerate_3d_worker(pg) -> None:
    """A 3D mesh with two dimensions of size 1 must behave like plain DP."""
    mesh = ParallelMesh(pg, dp=4, pp=1, tp=1)
    assert mesh.group("pp").world_size == 1
    assert mesh.group("tp").world_size == 1
    model = nn.Linear(WIDTH, 1)
    with torch.no_grad():
        model.weight.fill_(float(pg.rank))
        model.bias.zero_()
    model.weight.grad = torch.full_like(model.weight, float(pg.rank))
    average_gradients(mesh.group("dp"), list(model.parameters()))
    expected = sum(range(4)) / 4
    torch.testing.assert_close(model.weight.grad, torch.full_like(model.weight, expected))


def test_degenerate_dimensions_reduce_to_data_parallel() -> None:
    run(_degenerate_3d_worker, 4)
