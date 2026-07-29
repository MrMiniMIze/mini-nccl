"""Render benchmark charts from the CSV produced by bench_allreduce.py.

    python benchmarks/plot_results.py

Writes PNGs to docs/img/.
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


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    data = _load()
    plot_busbw(data)
    plot_latency(data)


if __name__ == "__main__":
    main()
