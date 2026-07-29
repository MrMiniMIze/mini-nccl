"""FSDP correctness: sharding must not change what the model learns.

The parity bar is the same one the DDP tests use. Training with parameters
split across W ranks, gathered only transiently, must produce the same
weights as ordinary single-process training on the full batch. On top of
that, the sharding has to actually save memory (otherwise it is elaborate
bookkeeping for nothing) and the full state must be reassemblable.
"""

from __future__ import annotations

import copy

import torch
import torch.nn.functional as F
from torch import nn

from mini_nccl.fsdp import FullyShardedDataParallel
from mini_nccl.launcher import run


class Block(nn.Module):
    """Stands in for a transformer block: one tensor in, one tensor out."""

    def __init__(self, width: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(width, 2 * width)
        self.fc2 = nn.Linear(2 * width, width)
        self.norm = nn.LayerNorm(width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.fc2(torch.tanh(self.fc1(self.norm(x))))


class Model(nn.Module):
    def __init__(self, width: int = 12, depth: int = 3) -> None:
        super().__init__()
        self.embed = nn.Linear(8, width)
        self.blocks = nn.ModuleList(Block(width) for _ in range(depth))
        self.head = nn.Linear(width, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.embed(x)
        for block in self.blocks:
            h = block(h)
        return self.head(h)


def _make_model() -> Model:
    torch.manual_seed(0)
    return Model()


def _parity_worker(pg, steps: int = 5) -> dict:
    W, per_rank = pg.world_size, 6
    model = _make_model()
    reference = copy.deepcopy(model)

    fsdp = FullyShardedDataParallel(model, pg, unit_cls=Block)
    assert len(fsdp.units) == 3, len(fsdp.units)
    # Every block parameter must now be sharded, not replicated.
    assert all(u.shard.numel() < u.padded_numel or W == 1 for u in fsdp.units)

    opt = torch.optim.SGD(fsdp.parameters(), lr=0.05, momentum=0.9)
    ref_opt = torch.optim.SGD(reference.parameters(), lr=0.05, momentum=0.9)

    gen = torch.Generator().manual_seed(11)
    for _ in range(steps):
        x = torch.randn(W * per_rank, 8, generator=gen)
        y = torch.randn(W * per_rank, 1, generator=gen)
        shard = slice(pg.rank * per_rank, (pg.rank + 1) * per_rank)

        fsdp.zero_grad()
        F.mse_loss(fsdp(x[shard]), y[shard]).backward()
        fsdp.sync()
        opt.step()

        ref_opt.zero_grad()
        F.mse_loss(reference(x), y).backward()
        ref_opt.step()

    # Compare the reassembled model against the reference.
    state = fsdp.full_state_dict()
    for name, ref_param in reference.named_parameters():
        torch.testing.assert_close(
            state[name], ref_param.detach(), rtol=1e-4, atol=1e-5,
            msg=lambda m, name=name: f"{name}: {m}",
        )
    return fsdp.memory_report()


def test_fsdp_matches_single_process() -> None:
    reports = run(_parity_worker, 2)
    assert reports[0] == reports[1]


def test_fsdp_matches_single_process_world4() -> None:
    run(_parity_worker, 4)


def _memory_worker(pg) -> dict:
    model = Model(width=64, depth=4)
    fsdp = FullyShardedDataParallel(model, pg, unit_cls=Block)
    x = torch.randn(4, 8)
    fsdp(x).sum().backward()
    fsdp.sync()
    return fsdp.memory_report()


def test_fsdp_reduces_parameter_memory() -> None:
    W = 4
    report = run(_memory_worker, W)[0]

    # Each rank keeps 1/W of the sharded parameters.
    expected_resident = report["sharded_params_full"] // W
    assert abs(report["sharded_params_resident"] - expected_resident) <= 4 * W

    # Only one unit is gathered at a time, so the transient cost is one
    # unit's worth rather than the whole model's.
    per_unit = report["sharded_params_full"] // 4
    assert report["peak_transient"] <= per_unit * 1.1, report

    # The point of all of it: peak parameter memory beats DDP's.
    assert report["fsdp_peak"] < report["ddp_equivalent"], report


def _optimizer_state_worker(pg) -> tuple[int, int]:
    """Optimizer state should scale with the shard, not the full model."""
    model = Model(width=64, depth=4)
    fsdp = FullyShardedDataParallel(model, pg, unit_cls=Block)
    opt = torch.optim.Adam(fsdp.parameters(), lr=1e-3)
    fsdp.zero_grad()
    fsdp(torch.randn(4, 8)).sum().backward()
    fsdp.sync()
    opt.step()

    state_numel = sum(
        v.numel()
        for group in opt.state.values()
        for v in group.values()
        if isinstance(v, torch.Tensor)
    )
    full_numel = sum(p.numel() for p in model.parameters() if p.numel() > 0)
    return state_numel, full_numel


def test_optimizer_state_is_sharded() -> None:
    results = run(_optimizer_state_worker, 4)
    state_numel, _ = results[0]
    # Adam keeps two moments per element, so state should be about
    # 2 * (sharded/W + replicated), well under 2 * full model.
    single = run(_optimizer_state_worker, 1)[0][0]
    assert state_numel < single * 0.6, (state_numel, single)
