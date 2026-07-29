"""Read flight recorder dumps and say which rank is the problem.

    python -m mini_nccl.diagnose <trace_dir>
    python -m mini_nccl.diagnose <trace_dir> --trace merged.json

Given a directory of ``rank*.json`` dumps (see ``run(..., trace_dir=...)``),
this reports each rank's progress and flags the two failure modes that make
distributed jobs hang:

- **A lagging rank.** Every rank must issue the same collectives in the same
  order. If rank 3 has completed 47 collectives while everyone else has
  completed 48, rank 3 is the one to look at, and the others are blocked on
  it rather than broken themselves.
- **An unfinished collective.** Any event still open when the dump was taken
  is a collective that never returned, printed with what it was waiting for.

``--trace`` merges every rank's events into one Trace Event Format file for
Perfetto or chrome://tracing, with one process per rank and one track per
channel.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

CHANNEL_LABEL = {-1: "collective order"}


def load(directory: Path) -> list[dict]:
    states = []
    for path in sorted(directory.glob("rank*.json")):
        states.append(json.loads(path.read_text(encoding="utf-8")))
    if not states:
        raise SystemExit(f"no rank*.json dumps found in {directory}")
    return sorted(states, key=lambda s: s["rank"])


def _order_streams(states: list[dict]) -> dict[int, dict[int, dict]]:
    """Per-rank map of sequence number to top-level collective.

    Keyed by sequence number rather than position, because the event log is
    a ring buffer: two ranks may have dropped different amounts, and only the
    sequence number identifies the same slot in the collective stream on
    every rank.
    """
    return {
        s["rank"]: {ev["seq"]: ev for ev in s["events"] if ev["channel"] == -1}
        for s in states
    }


def _find_divergence(streams: dict[int, dict[int, dict]]) -> int:
    """Report the first collective where ranks disagree. Returns findings count."""
    if len(streams) < 2:
        return 0

    for seq in sorted(set().union(*(set(d) for d in streams.values()))):
        present = {rank: d[seq] for rank, d in streams.items() if seq in d}
        if len(present) < 2:
            continue  # only one rank retained this slot; nothing to compare
        signatures = {r: (ev["op"], ev["nbytes"]) for r, ev in present.items()}
        if len(set(signatures.values())) <= 1:
            continue
        print(f"\nDESYNC at collective #{seq}: ranks issued *different* collectives.")
        for rank in sorted(signatures):
            op, nbytes = signatures[rank]
            print(f"    rank {rank}: {op} ({nbytes} bytes)")
        print(
            "  Ranks must issue identical collective sequences. Look at what the "
            "odd rank out did differently just before this point (a conditional "
            "branch, an early return, an unequal batch count)."
        )
        return 1

    # No mismatch in the overlap, so look for ranks that simply stopped early.
    reached = {rank: (max(d) if d else -1) for rank, d in streams.items()}
    if len(set(reached.values())) > 1:
        furthest = max(reached.values())
        behind = sorted(r for r, n in reached.items() if n < furthest)
        print(
            f"\nDESYNC: ranks {behind} stopped early, reaching collective "
            f"#{[reached[r] for r in behind]} while others reached #{furthest}.\n"
            f"  The lagging ranks are the cause; the rest are blocked waiting on them."
        )
        return 1
    return 0


def _straggler_report(
    states: list[dict], factor: float = 1.5, min_samples: int = 3
) -> None:
    """Name the rank the others are waiting for.

    A rank that is slow rather than wrong produces no desync and no unfinished
    collective: the job completes, at the pace of its worst member. It is
    visible here instead, through a signal that is worth stating carefully
    because the obvious version of it is backwards.

    The straggler does **not** spend longer inside its collectives. It spends
    *less*. Arriving last means it finds its peers already waiting, so it
    returns almost immediately, while every rank that arrived on time sits
    blocked until it shows up. So the rank to look at is the one whose
    collectives are consistently the *fastest*.

    Using durations rather than entry timestamps also keeps this usable across
    machines, where wall clocks need not agree closely enough to rank arrival
    order. Reported as information: a straggler is a performance problem, so it
    does not change the exit code.

    Operations called fewer than ``min_samples`` times per rank are skipped. One
    observation cannot establish that a rank is *consistently* slow, and letting
    a single barrier weigh as much as a hundred all-reduces is enough to hide a
    real straggler.
    """
    if len(states) < 3:
        return  # with two ranks there is no majority to compare against

    per_rank_op: dict[int, dict[str, list[float]]] = {}
    for state in states:
        by_op: dict[str, list[float]] = {}
        for ev in state["events"]:
            if ev["dur_us"] is not None and ev["channel"] == -1:
                by_op.setdefault(ev["op"], []).append(ev["dur_us"])
        per_rank_op[state["rank"]] = by_op

    ops = sorted({op for by_op in per_rank_op.values() for op in by_op})
    fast_counts: dict[int, int] = dict.fromkeys(per_rank_op, 0)
    comparable = 0
    for op in ops:
        medians = {
            rank: statistics.median(by_op[op])
            for rank, by_op in per_rank_op.items()
            if len(by_op.get(op, ())) >= min_samples
        }
        if len(medians) < 3:
            continue
        comparable += 1
        across = statistics.median(medians.values())
        if across <= 0:
            continue
        for rank, value in medians.items():
            if value * factor < across:
                fast_counts[rank] += 1

    if not comparable:
        return
    for rank, count in sorted(fast_counts.items()):
        if count > comparable / 2:
            print(
                f"\nSTRAGGLER: rank {rank} spent less than 1/{factor:g} of the "
                f"median time inside {count} of {comparable} collectives.\n"
                f"  That is the signature of the rank everyone else waits for: it "
                f"arrives last, finds its peers already blocked, and returns at "
                f"once. Look for a slower host, an unbalanced shard, or contention "
                f"on that rank rather than at the ranks reporting long waits."
            )


def report(states: list[dict]) -> int:
    """Print a diagnosis. Returns a shell exit code (0 clean, 1 suspicious)."""
    world = states[0]["world_size"]
    print(f"ranks reporting: {len(states)}/{world}")
    findings = 0
    missing = sorted(set(range(world)) - {s["rank"] for s in states})
    if missing:
        print(f"  MISSING DUMPS from ranks {missing} (crashed before writing?)")
        findings += 1

    print("\nper-rank progress (collectives issued, by channel):")
    for s in states:
        counts = {
            CHANNEL_LABEL.get(int(c), f"ch{c}"): n for c, n in sorted(s["next_seq"].items())
        }
        note = ""
        if s.get("dropped"):
            note = f"  [ring buffer dropped {s['dropped']} older events]"
        print(f"  rank {s['rank']}: {counts}{note}")

    streams = _order_streams(states)
    findings += _find_divergence(streams)
    _straggler_report(states)

    for s in states:
        for ev in s["pending"]:
            print(
                f"\nUNFINISHED on rank {s['rank']}: {ev['op']}[{ev['algorithm']}] "
                f"seq={ev['seq']} channel={ev['channel']} bytes={ev['nbytes']} "
                f"never completed."
            )
            findings += 1

    if not findings:
        print("\nno desync or unfinished collectives found; all ranks agree.")
    return 1 if findings else 0


def merge_trace(states: list[dict], out_path: Path) -> None:
    """Combine per-rank events into one Trace Event Format file."""
    zero = min(s["wall_start"] for s in states)
    events: list[dict] = []
    for s in states:
        rank = s["rank"]
        events.append(
            {"name": "process_name", "ph": "M", "pid": rank, "args": {"name": f"rank {rank}"}}
        )
        seen = set()
        for ev in s["events"]:
            channel = ev["channel"]
            if channel not in seen:
                seen.add(channel)
                events.append(
                    {
                        "name": "thread_name",
                        "ph": "M",
                        "pid": rank,
                        "tid": channel,
                        "args": {
                            "name": CHANNEL_LABEL.get(channel, f"channel {channel}"),
                        },
                    }
                )
            events.append(
                {
                    "name": f"{ev['op']}:{ev['algorithm']}" if ev["algorithm"] else ev["op"],
                    "cat": "collective",
                    "ph": "X",
                    "ts": (ev["wall_start"] - zero) * 1e6,
                    "dur": ev["dur_us"] if ev["dur_us"] is not None else 0.0,
                    "pid": rank,
                    "tid": channel,
                    "args": {
                        "seq": ev["seq"],
                        "bytes": ev["nbytes"],
                        "completed": ev["dur_us"] is not None,
                        **ev.get("detail", {}),
                    },
                }
            )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(events), encoding="utf-8")
    print(f"\nwrote merged trace {out_path} ({len(events)} records)")
    print("open it in https://ui.perfetto.dev or chrome://tracing")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("directory", type=Path, help="directory of rank*.json dumps")
    ap.add_argument("--trace", type=Path, help="also write a merged Chrome trace here")
    args = ap.parse_args()

    states = load(args.directory)
    code = report(states)
    if args.trace:
        merge_trace(states, args.trace)
    raise SystemExit(code)


if __name__ == "__main__":
    main()
