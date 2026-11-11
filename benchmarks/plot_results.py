"""Render benchmark charts from the CSVs in benchmarks/results/.

    python benchmarks/plot_results.py

Reads allreduce.csv (bench_allreduce.py) and, if present, emulated_link.csv
(bench_emulated_link.py). Writes PNGs to docs/img/.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, NullLocator

CSV_PATH = Path(__file__).parent / "results" / "allreduce.csv"
OUT_DIR = Path(__file__).parent.parent / "docs" / "img"

# Fixed series order and colors; identity never depends on plot order.
SERIES = {
    "ring": {"color": "#2a78d6", "marker": "o", "label": "mini-nccl ring"},
    "tree": {"color": "#eb6834", "marker": "s", "label": "mini-nccl tree"},
    "halving": {"color": "#1baf7a", "marker": "^", "label": "mini-nccl halving-doubling"},
    "naive": {"color": "#eda100", "marker": "D", "label": "mini-nccl naive"},
    "gloo": {"color": "#e87ba4", "marker": "v", "label": "torch.distributed gloo"},
}

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"


def _load() -> dict[int, dict[str, list[tuple[int, float]]]]:
    data: dict[int, dict[str, list[tuple[int, float]]]] = defaultdict(lambda: defaultdict(list))
    with CSV_PATH.open() as f:
        for row in csv.DictReader(f):
            data[int(row["world"])][row["algorithm"]].append(
                (int(row["bytes"]), float(row["seconds"]))
            )
    return data


def _style_axes(ax) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
        ax.spines[side].set_linewidth(0.8)
    ax.grid(True, axis="y", color=GRID, linewidth=0.8)
    ax.tick_params(colors=INK_MUTED, labelsize=9)
    ax.xaxis.label.set_color(INK_MUTED)
    ax.yaxis.label.set_color(INK_MUTED)
    ax.title.set_color(INK)


_SIZE_TICKS = [4096, 65536, 1048576, 16777216, 67108864]
_SIZE_LABELS = ["4 KiB", "64 KiB", "1 MiB", "16 MiB", "64 MiB"]


def _size_axis(ax) -> None:
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(FixedLocator(_SIZE_TICKS))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xticklabels(_SIZE_LABELS)
    ax.set_xlabel("message size")


def plot_busbw(data) -> None:
    worlds = sorted(data)
    fig, axes = plt.subplots(
        1, len(worlds), figsize=(5.4 * len(worlds), 4.2), facecolor=SURFACE, sharey=True
    )
    for ax, world in zip(axes, worlds, strict=True):
        for name, spec in SERIES.items():
            if name not in data[world]:
                continue
            points = sorted(data[world][name])
            xs = [b for b, _ in points]
            ys = [b * 2 * (world - 1) / world / s / 1e9 for b, s in points]
            ax.plot(
                xs, ys,
                color=spec["color"], marker=spec["marker"], label=spec["label"],
                linewidth=2, markersize=6, markeredgecolor=SURFACE, markeredgewidth=0.8,
            )
        _style_axes(ax)
        _size_axis(ax)
        ax.set_title(f"{world} ranks", fontsize=11)
        if ax is axes[0]:
            ax.set_ylabel("bus bandwidth (GB/s)")
    axes[-1].legend(frameon=False, fontsize=9, labelcolor=INK, loc="upper left")
    fig.suptitle(
        "all-reduce bus bandwidth, loopback TCP (higher is better)",
        color=INK, fontsize=12,
    )
    fig.tight_layout()
    out = OUT_DIR / "allreduce_busbw.png"
    fig.savefig(out, dpi=160, facecolor=SURFACE)
    print(f"wrote {out}")


def plot_latency(data) -> None:
    world = max(data)
    fig, ax = plt.subplots(figsize=(7.2, 4.4), facecolor=SURFACE)
    for name, spec in SERIES.items():
        if name not in data[world]:
            continue
        points = sorted(data[world][name])
        xs = [b for b, _ in points]
        ys = [s * 1e3 for _, s in points]
        ax.plot(
            xs, ys,
            color=spec["color"], marker=spec["marker"], label=spec["label"],
            linewidth=2, markersize=6, markeredgecolor=SURFACE, markeredgewidth=0.8,
        )
    _style_axes(ax)
    _size_axis(ax)
    ax.set_yscale("log")
    ax.set_ylabel("time per all-reduce (ms)")
    ax.set_title(
        f"all-reduce latency, {world} ranks: tree wins small, ring wins large",
        fontsize=11,
    )
    ax.legend(frameon=False, fontsize=9, labelcolor=INK, loc="upper left")
    fig.tight_layout()
    out = OUT_DIR / "allreduce_latency.png"
    fig.savefig(out, dpi=160, facecolor=SURFACE)
    print(f"wrote {out}")


LINK_CSV = Path(__file__).parent / "results" / "emulated_link.csv"

# One color per link, fixed, so a link keeps its color across both panels.
LINKS = {
    "100 Mb/s": {"color": "#eb6834", "marker": "s"},
    "1000 Mb/s": {"color": "#2a78d6", "marker": "o"},
    "loopback": {"color": INK_MUTED, "marker": "D"},
}

# The pre-registered predictions, copied from docs/emulated-link.md (committed
# before any shaped run). Each: label, predicted ratio (low, high), and how to
# compute the measured ratio as (experiment, link, world, numerator, denominator).
PREDICTIONS = [
    ("naive / ring, 2 ranks, 1 Gb/s", (1.97, 1.97), ("naive", "1000 Mb/s", 2, "naive", "ring")),
    ("naive / ring, 4 ranks, 1 Gb/s", (3.88, 3.88), ("naive", "1000 Mb/s", 4, "naive", "ring")),
    ("naive / ring, 8 ranks, 1 Gb/s", (7.72, 7.72), ("naive", "1000 Mb/s", 8, "naive", "ring")),
    ("naive / ring, 4 ranks, 100 Mb/s", (3.99, 3.99), ("naive", "100 Mb/s", 4, "naive", "ring")),
    ("bf16 wire, 1 Gb/s", (1.91, 1.91), ("bf16", "1000 Mb/s", 4, "ring", "ring+bf16")),
    ("bf16 wire, 100 Mb/s", (1.99, 1.99), ("bf16", "100 Mb/s", 4, "ring", "ring+bf16")),
    ("4 channels, per-rank cap", (1.00, 1.00), ("channels", "1000 Mb/s per-rank", 4, "1ch", "4ch")),
    ("4 channels, per-flow cap", (3.50, 3.94), ("channels", "1000 Mb/s per-flow", 4, "1ch", "4ch")),
    ("16 slices, 1 Gb/s", (1.00, 1.00), ("slices", "1000 Mb/s", 4, "slices=1", "slices=16")),
    ("16 slices, 100 Mb/s", (1.00, 1.00), ("slices", "100 Mb/s", 4, "slices=1", "slices=16")),
]
TOLERANCE = 0.15  # the pass band fixed in docs/emulated-link.md

# DDP overlap was predicted as a gain g, judged at g ± 15%. Plotted on the same
# axis as the ratios above, as the step-time ratio 1 / (1 - g), so its band is
# converted rather than widened to ±15% of the ratio (which would pass misses).
DDP_GAINS = [("100 Mb/s", 0.040), ("1000 Mb/s", 0.22), ("4700 Mb/s", 0.367)]
for _link, _gain in DDP_GAINS:
    PREDICTIONS.append((
        f"DDP overlap, {_link.replace('1000 Mb/s', '1 Gb/s').replace('4700 Mb/s', '4.7 Gb/s')}",
        (1 / (1 - _gain), 1 / (1 - _gain)),
        ("ddp", _link, 4, "no-overlap:step", "overlap:step"),
    ))


def _band(label: str, lo: float, hi: float) -> tuple[float, float]:
    if label.startswith("DDP"):
        gain = 1 - 1 / lo
        return 1 / (1 - gain * (1 - TOLERANCE)), 1 / (1 - gain * (1 + TOLERANCE))
    if lo == hi == 1.0:
        return 0.95, 1.05  # "no change" predictions use a fixed 0.95x to 1.05x
    return lo * (1 - TOLERANCE), hi * (1 + TOLERANCE)


def _load_link() -> dict[tuple[str, str, int, str, int], float]:
    with LINK_CSV.open() as f:
        return {
            (r["experiment"], r["link"], int(r["world"]), r["config"], int(r["bytes"])):
            float(r["seconds"])
            for r in csv.DictReader(f)
        }


def _measured(link_data, spec) -> float | None:
    experiment, link, world, num, den = spec
    times = {
        config: s for (e, lk, w, config, _), s in link_data.items()
        if (e, lk, w) == (experiment, link, world)
    }
    if num not in times or den not in times:
        return None
    return times[num] / times[den]


def plot_emulated_link(link_data) -> None:
    fig, (left, right) = plt.subplots(
        1, 2, figsize=(12.4, 5.6), facecolor=SURFACE, gridspec_kw={"width_ratios": [1, 1.15]}
    )

    # Left: tree time over ring time by size. Above 1.0, ring is faster.
    for link, spec in LINKS.items():
        sizes = sorted({b for (e, lk, _, _, b) in link_data if e == "crossover" and lk == link})
        points = []
        for size in sizes:
            ring = link_data.get(("crossover", link, 4, "ring", size))
            tree = link_data.get(("crossover", link, 4, "tree", size))
            if ring and tree:
                points.append((size, tree / ring))
        if not points:
            continue
        left.plot(
            [b for b, _ in points], [r for _, r in points],
            color=spec["color"], marker=spec["marker"],
            label=link.replace("1000 Mb/s", "1 Gb/s"),
            linewidth=2, markersize=6, markeredgecolor=SURFACE, markeredgewidth=0.8,
        )
    left.axhline(1.0, color=AXIS, linewidth=1, linestyle="--")
    _style_axes(left)
    left.set_xscale("log")
    ticks = [1024, 4096, 16384, 65536, 262144, 1048576]
    left.xaxis.set_major_locator(FixedLocator(ticks))
    left.xaxis.set_minor_locator(NullLocator())
    left.set_xticklabels(["1 KiB", "4 KiB", "16 KiB", "64 KiB", "256 KiB", "1 MiB"])
    left.set_xlabel("message size")
    left.set_ylabel("tree time / ring time")
    left.set_title("4 ranks: above the dashed line, ring is faster", fontsize=11)
    left.legend(frameon=False, fontsize=9, labelcolor=INK, loc="upper left")

    # Right: every pre-registered ratio, its pass band, and what was measured.
    rows = [(label, band, _measured(link_data, spec)) for label, band, spec in PREDICTIONS]
    rows = [r for r in rows if r[2] is not None]
    for y, (label, (lo, hi), measured) in enumerate(rows):
        band_lo, band_hi = _band(label, lo, hi)
        right.plot(
            [band_lo, band_hi], [y, y],
            color=GRID, linewidth=7, solid_capstyle="butt", zorder=1,
        )
        right.plot(
            [lo, hi] if hi > lo else [lo], [y, y] if hi > lo else [y],
            color=INK_MUTED, marker="|", markersize=12, linewidth=2, zorder=2,
        )
        held = band_lo <= measured <= band_hi
        right.plot([measured], [y], color="#2a78d6" if held else "#eb6834", marker="o",
                   markersize=7, zorder=3, markeredgecolor=SURFACE, markeredgewidth=0.8)
    _style_axes(right)
    right.grid(False)
    right.grid(True, axis="x", color=GRID, linewidth=0.8)
    right.set_xscale("log")
    right.xaxis.set_major_locator(FixedLocator([1, 2, 4, 8]))
    right.xaxis.set_minor_locator(NullLocator())
    right.set_xticklabels(["1x", "2x", "4x", "8x"])
    right.set_yticks(range(len(rows)))
    right.set_yticklabels([label for label, _, _ in rows], fontsize=9, color=INK)
    right.invert_yaxis()
    right.set_xlabel("tick: predicted · band: passes · dot: measured (orange: a miss)")
    right.set_title("predicted before measuring, then measured", fontsize=11)

    fig.suptitle("an emulated link: 16 MiB all-reduce, or one DDP training step", color=INK, fontsize=12)
    fig.tight_layout()
    out = OUT_DIR / "emulated_link.png"
    fig.savefig(out, dpi=160, facecolor=SURFACE)
    print(f"wrote {out}")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    data = _load()
    plot_busbw(data)
    plot_latency(data)
    if LINK_CSV.exists():
        plot_emulated_link(_load_link())


if __name__ == "__main__":
    main()
