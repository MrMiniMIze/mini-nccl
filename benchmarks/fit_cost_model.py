"""Fit an alpha-beta cost model to the measured all-reduce times.

    python benchmarks/fit_cost_model.py

The classic model for a collective is

    t(n) = alpha * steps + beta * bytes_on_critical_path(n)

where ``alpha`` is per-message latency and ``beta`` is per-byte cost
(1/beta is the achieved bandwidth). Each algorithm has a known step count
and byte count, so fitting the measured times recovers alpha and beta from
data instead of assuming them, and the fitted model predicts where the
algorithms cross over. Comparing that prediction with the measured crossover
is a check on whether the model (and therefore the mental picture behind it)
is right.

Reads benchmarks/results/allreduce.csv, prints a table, and writes
docs/img/cost_model.png.
"""

from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

CSV_PATH = Path(__file__).parent / "results" / "allreduce.csv"
OUT_PATH = Path(__file__).parent.parent / "docs" / "img" / "cost_model.png"

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
COLORS = {
    "ring": "#2a78d6",
    "tree": "#eb6834",
    "halving": "#1baf7a",
    "naive": "#eda100",
}
MARKERS = {"ring": "o", "tree": "s", "halving": "^", "naive": "D"}


def steps_and_byte_factor(algorithm: str, world: int) -> tuple[int, float]:
    """Step count, and bytes-on-the-critical-path per byte of payload.

    - ring: 2(W-1) steps, each rank moves 2(W-1)/W of the payload
    - tree: 2*ceil(log2 W) steps; the root forwards the whole payload each time
    - halving: 2*log2(W) steps with ring's byte count (falls back to ring
      when W is not a power of two)
    - naive: rank 0 serially receives from and sends to W-1 peers
    """
    log2w = math.ceil(math.log2(world))
    if algorithm == "ring":
        return 2 * (world - 1), 2 * (world - 1) / world
    if algorithm == "tree":
        return 2 * log2w, 2 * log2w
    if algorithm == "halving":
        if world & (world - 1):
            return 2 * (world - 1), 2 * (world - 1) / world
        return 2 * log2w, 2 * (world - 1) / world
    if algorithm == "naive":
        return 2 * (world - 1), 2 * (world - 1)
    raise KeyError(algorithm)


def load() -> dict[int, dict[str, list[tuple[int, float]]]]:
    data: dict[int, dict[str, list[tuple[int, float]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    with CSV_PATH.open() as f:
        for row in csv.DictReader(f):
            if row["algorithm"] not in COLORS:
                continue  # gloo has no step count we can claim to know
            data[int(row["world"])][row["algorithm"]].append(
                (int(row["bytes"]), float(row["seconds"]))
            )
    return data


def fit(points: list[tuple[int, float]], steps: int, byte_factor: float) -> dict:
    """Fit t = alpha*steps + beta*byte_factor*n, minimizing *relative* error.

    Message sizes span 4 KiB to 64 MiB, so an absolute-error fit would be
    decided entirely by the largest points and the latency term would be
    fitted to noise. Dividing each equation by its own measured time makes
    every size count equally, which is what we want when one parameter
    (alpha) only shows up at the small end and the other (beta) only at the
    large end. Both parameters are physically non-negative, so a negative
    solution is refitted with that term pinned at zero.
    """
    n = np.array([p[0] for p in points], dtype=float)
    t = np.array([p[1] for p in points], dtype=float)
    payload_bytes = byte_factor * n
    design = np.stack([np.full_like(t, float(steps)) / t, payload_bytes / t], axis=1)
    target = np.ones_like(t)

    alpha, beta = np.linalg.lstsq(design, target, rcond=None)[0]
    if alpha < 0 or beta < 0:
        keep = 1 if alpha < 0 else 0
        column = design[:, keep]
        solved = float(column @ target / (column @ column))
        alpha, beta = (0.0, solved) if alpha < 0 else (solved, 0.0)

    predicted = alpha * steps + beta * payload_bytes
    relative = np.abs(predicted - t) / t
    return {
        "alpha_s": float(alpha),
        "beta_s_per_byte": float(beta),
        "steps": steps,
        "byte_factor": byte_factor,
        "mean_rel_err": float(relative.mean()),
        "max_rel_err": float(relative.max()),
    }


def model_seconds(model: dict, nbytes: np.ndarray | float):
    return model["alpha_s"] * model["steps"] + model["beta_s_per_byte"] * (
        model["byte_factor"] * nbytes
    )


def crossover_bytes(a: dict, b: dict) -> float | None:
    """Payload size where model ``a`` and model ``b`` cost the same."""
    latency_gap = a["alpha_s"] * a["steps"] - b["alpha_s"] * b["steps"]
    slope_gap = b["beta_s_per_byte"] * b["byte_factor"] - a["beta_s_per_byte"] * a[
        "byte_factor"
    ]
    if abs(slope_gap) < 1e-24:
        return None
    n = latency_gap / slope_gap
    return n if n > 0 else None


def measured_crossover(
    fast_small: list[tuple[int, float]], fast_large: list[tuple[int, float]]
) -> int | None:
    """Smallest measured size where ``fast_large`` first beats ``fast_small``."""
    small = dict(fast_small)
    for nbytes, seconds in sorted(fast_large):
        if nbytes in small and seconds < small[nbytes]:
            return nbytes
    return None


def human(nbytes: float) -> str:
    for unit, scale in (("MiB", 2**20), ("KiB", 2**10)):
        if nbytes >= scale:
            return f"{nbytes / scale:.1f} {unit}"
    return f"{nbytes:.0f} B"


def main() -> None:
    data = load()
    fits: dict[int, dict[str, dict]] = {}

    for world in sorted(data):
        fits[world] = {}
        print(f"\nworld_size = {world}")
        print("| algorithm | steps | payload moved | alpha (per step) | beta (per byte) | "
              "implied bandwidth | mean err |")
        print("|---|---|---|---|---|---|---|")
        for algorithm in ("ring", "tree", "halving", "naive"):
            if algorithm not in data[world]:
                continue
            steps, byte_factor = steps_and_byte_factor(algorithm, world)
            model = fit(data[world][algorithm], steps, byte_factor)
            fits[world][algorithm] = model
            bandwidth = 1 / model["beta_s_per_byte"] / 1e9 if model["beta_s_per_byte"] else 0
            print(
                f"| {algorithm} | {steps} | {byte_factor:.2f}n | "
                f"{model['alpha_s'] * 1e6:.0f} us | "
                f"{model['beta_s_per_byte'] * 1e9:.2f} ns | {bandwidth:.2f} GB/s | "
                f"{model['mean_rel_err'] * 100:.1f}% |"
            )

        if "tree" in fits[world] and "ring" in fits[world]:
            predicted = crossover_bytes(fits[world]["tree"], fits[world]["ring"])
            observed = measured_crossover(data[world]["tree"], data[world]["ring"])
            print(
                f"\ntree/ring crossover: model says {human(predicted) if predicted else 'n/a'}"
                f", measurement says {human(observed) if observed else 'n/a'}"
            )

    plot(data, fits)


def plot(data, fits) -> None:
    worlds = sorted(data)
    fig, axes = plt.subplots(
        1, len(worlds), figsize=(5.6 * len(worlds), 4.3), facecolor=SURFACE, sharey=True
    )
    axes = np.atleast_1d(axes)
    for ax, world in zip(axes, worlds, strict=True):
        for algorithm in ("ring", "tree", "halving", "naive"):
            if algorithm not in data[world]:
                continue
            points = sorted(data[world][algorithm])
            xs = np.array([p[0] for p in points], dtype=float)
            ys = np.array([p[1] * 1e3 for p in points])
            model = fits[world][algorithm]
            ax.plot(
                xs, ys, linestyle="none", marker=MARKERS[algorithm],
                color=COLORS[algorithm], markersize=6,
                markeredgecolor=SURFACE, markeredgewidth=0.8,
                label=f"{algorithm} (measured)",
            )
            fine = np.geomspace(xs.min(), xs.max(), 100)
            ax.plot(
                fine, model_seconds(model, fine) * 1e3,
                color=COLORS[algorithm], linewidth=1.6, alpha=0.75,
            )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_facecolor(SURFACE)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(AXIS)
            ax.spines[side].set_linewidth(0.8)
        ax.grid(True, axis="y", color=GRID, linewidth=0.8)
        ax.tick_params(colors=INK_MUTED, labelsize=9)
        ax.set_xlabel("message size (bytes)", color=INK_MUTED)
        ax.set_title(f"{world} ranks", color=INK, fontsize=11)
        if ax is axes[0]:
            ax.set_ylabel("time per all-reduce (ms)", color=INK_MUTED)
    axes[-1].legend(frameon=False, fontsize=9, labelcolor=INK, loc="upper left")
    fig.suptitle(
        "alpha-beta model (lines) vs measurement (points)", color=INK, fontsize=12
    )
    fig.tight_layout()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=160, facecolor=SURFACE)
    print(f"\nwrote {OUT_PATH}")


if __name__ == "__main__":
    main()
