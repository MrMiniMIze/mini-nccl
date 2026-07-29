"""Narrow wire, wide accumulator, and where the error actually comes from.

Sending gradients as bfloat16 halves the bytes on the wire. The interesting
question is what it costs in accuracy, and the answer is not the one the
"accumulate in float32" framing suggests.

These tests pin down the measured behavior:

1. A float32 tensor with a bfloat16 wire still produces the right answer, at
   half the bytes.
2. Widening the accumulator changes nothing detectable, because PyTorch's CPU
   bfloat16 kernels already compute in float32 and round on store. The
   explicit float32 accumulator is insurance against that not being true (on
   another backend, or another dtype), not a fix for a live bug.
3. The error is set by **how many hops the partial sum crosses**, since each
   hop rounds it back to the wire dtype. So tree, at O(log W) hops, keeps its
   error nearly flat in world size, while ring, at O(W) hops, grows linearly.
   Low precision inverts the usual ranking: ring moves the fewest bytes and is
   the least accurate.

That third point is the one worth a regression test, since it is the property
an implementation could silently lose.
"""

from __future__ import annotations

import pytest
import torch

from mini_nccl import collectives as c
from mini_nccl.launcher import run

BIG, SMALL = 1.0, 0.004


def _error_worker(pg) -> dict:
    """Rank 0 contributes 1.0; everyone else contributes 0.004.

    bfloat16 has 8 mantissa bits, so near 1.0 its spacing is about 0.0039:
    each small contribution sits right at the rounding threshold, which makes
    accumulated rounding easy to see.
    """
    W = pg.world_size
    exact = BIG + SMALL * (W - 1)
    value = BIG if pg.rank == 0 else SMALL
    out = {"exact": exact, "world": W}

    for algorithm in ("ring", "tree"):
        narrow = torch.full((64,), value, dtype=torch.bfloat16)
        c.all_reduce(pg, narrow, algorithm=algorithm)
        out[f"{algorithm}_narrow"] = narrow[0].item()

        wide = torch.full((64,), value, dtype=torch.bfloat16)
        c.all_reduce(pg, wide, algorithm=algorithm, wire_dtype=torch.bfloat16)
        out[f"{algorithm}_wide_accum"] = wide[0].item()

        full = torch.full((64,), value, dtype=torch.float32)
        c.all_reduce(pg, full, algorithm=algorithm)
        out[f"{algorithm}_fp32"] = full[0].item()
    return out


def _errors(report: dict) -> dict[str, float]:
    exact = report["exact"]
    return {
        key: abs(value - exact)
        for key, value in report.items()
        if key not in ("exact", "world")
    }


def test_tree_is_far_more_accurate_than_ring_in_low_precision() -> None:
    small = run(_error_worker, 4)
    large = run(_error_worker, 8)

    for reports in (small, large):
        # Whatever the precision, every rank must land on the same value.
        for other in reports[1:]:
            assert other == reports[0], (reports[0], other)

    err4, err8 = _errors(small[0]), _errors(large[0])

    # float32 end to end is the reference and stays exact.
    assert err4["ring_fp32"] < 1e-6 and err4["tree_fp32"] < 1e-6
    assert err8["ring_fp32"] < 1e-6 and err8["tree_fp32"] < 1e-6

    # Ring's error grows with world size: more hops, more roundings.
    assert err8["ring_narrow"] > 1.5 * err4["ring_narrow"], (err4, err8)

    # Tree's does not, because its hop count grows logarithmically.
    assert err8["tree_narrow"] < 1.5 * err4["tree_narrow"], (err4, err8)

    # And so tree is dramatically more accurate at the larger world size.
    assert err8["tree_narrow"] < err8["ring_narrow"] / 4, (err4, err8)


def test_wide_accumulator_is_not_the_lever() -> None:
    """Documents the negative result rather than pretending it is a win."""
    report = run(_error_worker, 8)[0]
    errors = _errors(report)
    for algorithm in ("ring", "tree"):
        # Never worse than accumulating in the narrow type...
        assert errors[f"{algorithm}_wide_accum"] <= errors[f"{algorithm}_narrow"] + 1e-9
        # ...and on this backend, not measurably better either, because the
        # narrow kernels already widen internally. If this ever starts
        # failing, PyTorch changed and the docstring needs revisiting.
        assert errors[f"{algorithm}_wide_accum"] == pytest.approx(
            errors[f"{algorithm}_narrow"], abs=1e-9
        )


def _exact_payload_worker(pg) -> None:
    """A float32 tensor whose sum survives a bfloat16 round trip exactly."""
    W = pg.world_size
    for algorithm in ("ring", "tree"):
        for numel in (1, 7, 1024, 100_003):
            # Powers of two are representable in bfloat16, so the only thing
            # under test is that the narrow wire moves the right bits.
            x = torch.full((numel,), 0.5, dtype=torch.float32)
            c.all_reduce(pg, x, algorithm=algorithm, wire_dtype=torch.bfloat16)
            torch.testing.assert_close(x, torch.full((numel,), 0.5 * W))


def test_float32_tensor_over_a_narrow_wire() -> None:
    run(_exact_payload_worker, 4)


def _mixed_payload_worker(pg) -> None:
    """Random float32 payload: right answer to within bfloat16's resolution."""
    W = pg.world_size
    for algorithm in ("ring", "tree"):
        x = torch.full((4096,), float(pg.rank + 1))
        c.all_reduce(pg, x, algorithm=algorithm, wire_dtype=torch.bfloat16)
        expected = torch.full((4096,), float(W * (W + 1) // 2))
        torch.testing.assert_close(x, expected, rtol=0.01, atol=0.01)


def test_narrow_wire_stays_within_bfloat16_resolution() -> None:
    run(_mixed_payload_worker, 4)


def _rejects_worker(pg) -> None:
    with pytest.raises(ValueError, match="ring and tree"):
        c.all_reduce(pg, torch.zeros(16), algorithm="naive", wire_dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="wider than"):
        c.all_reduce(pg, torch.zeros(16, dtype=torch.bfloat16), wire_dtype=torch.float32)


def test_unsupported_combinations_are_rejected() -> None:
    run(_rejects_worker, 2)
