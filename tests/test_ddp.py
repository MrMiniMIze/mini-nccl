"""DDP correctness: distributed training must match single-process training.

The gold-standard test for a data-parallel implementation: train the same
model (a) on one process with the full batch and (b) on W processes each
taking 1/W of the batch with gradient averaging. After several optimizer
steps the parameters must agree to floating-point tolerance.
"""

from __future__ import annotations

import copy

import torch
import torch.nn.functional as F
from torch import nn

from mini_nccl.ddp import DistributedDataParallel
from mini_nccl.launcher import run


def _make_model() -> nn.Module:
    torch.manual_seed(0)
    return nn.Sequential(
        nn.Linear(16, 64),
        nn.Tanh(),
        nn.Linear(64, 32),
        nn.Tanh(),
        nn.Linear(32, 1),
    )


def _train_parity(pg, overlap: bool, bucket_cap_mb: float, steps: int = 6) -> None:
    W = pg.world_size
    per_rank_batch = 8

    model = _make_model()
    reference = copy.deepcopy(model)
    ddp = DistributedDataParallel(
        model, pg, bucket_cap_mb=bucket_cap_mb, overlap=overlap
    )
    if bucket_cap_mb < 0.01:
        assert len(ddp._buckets) > 1, "tiny cap should force multiple buckets"

    opt = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    ref_opt = torch.optim.SGD(reference.parameters(), lr=0.1, momentum=0.9)

    data_gen = torch.Generator().manual_seed(42)
    for _ in range(steps):
        x = torch.randn(W * per_rank_batch, 16, generator=data_gen)
        y = torch.randn(W * per_rank_batch, 1, generator=data_gen)
        shard = slice(pg.rank * per_rank_batch, (pg.rank + 1) * per_rank_batch)

        ddp.zero_grad()
        F.mse_loss(ddp(x[shard]), y[shard]).backward()
        ddp.sync()
        opt.step()

        # Reference: identical model trained on the full batch, one process.
        ref_opt.zero_grad()
        F.mse_loss(reference(x), y).backward()
        ref_opt.step()

    for p, ref_p in zip(model.parameters(), reference.parameters()):
        torch.testing.assert_close(p, ref_p, rtol=1e-4, atol=1e-5)


def _parity_battery_worker(pg) -> None:
    # overlap on (the real configuration), overlap off (the debugging one),
    # and a 1 KiB bucket cap that slices the model into many buckets.
    _train_parity(pg, overlap=True, bucket_cap_mb=1.0)
    _train_parity(pg, overlap=False, bucket_cap_mb=1.0)
    _train_parity(pg, overlap=True, bucket_cap_mb=0.001)


def test_ddp_matches_single_process() -> None:
    run(_parity_battery_worker, 2)


def test_ddp_matches_single_process_world3() -> None:
    run(_parity_battery_worker, 3)


def _broadcast_init_worker(pg) -> None:
    # Ranks start with *different* weights; DDP must sync them to rank 0's.
    torch.manual_seed(100 + pg.rank)
    model = nn.Linear(8, 8)
    DistributedDataParallel(model, pg)
    torch.manual_seed(100)  # rank 0's seed
    expected = nn.Linear(8, 8)
    torch.testing.assert_close(model.weight, expected.weight)
    torch.testing.assert_close(model.bias, expected.bias)


def test_ddp_broadcasts_initial_weights() -> None:
    run(_broadcast_init_worker, 2)
