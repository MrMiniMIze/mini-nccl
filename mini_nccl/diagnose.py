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
from pathlib import Path

CHANNEL_LABEL = {-1: "collective order"}


def load(directory: Path) -> list[dict]:
    states = []
    for path in sorted(directory.glob("rank*.json")):
        states.append(json.loads(path.read_text(encoding="utf-8")))
    if not states:
        raise SystemExit(f"no rank*.json dumps found in {directory}")
    return sorted(states, key=lambda s: s["rank"])


def _order_streams(states: list[dict]) -> dict[int, list[dict]]:
    """Per-rank list of top-level collectives, in issue order."""
    streams = {}
    for s in states:
        events = [ev for ev in s["events"] if ev["channel"] == -1]
        streams[s["rank"]] = sorted(events, key=lambda ev: ev["seq"])
    return streams


def _find_divergence(streams: dict[int, list[dict]]) -> int:
    """Report the first collective where ranks disagree. Returns findings count."""
    if len(streams) < 2:
        return 0
    longest = max(len(s) for s in streams.values())
    for i in range(longest):
        signatures = {}
        for rank, stream in streams.items():
            signatures[rank] = (
                (stream[i]["op"], stream[i]["nbytes"]) if i < len(stream) else None
            )
        distinct = {sig for sig in signatures.values() if sig is not None}
        if len(distinct) <= 1 and None not in signatures.values():
            continue
        if len(distinct) <= 1 and None in signatures.values():
            missing = sorted(r for r, sig in signatures.items() if sig is None)
            print(
                f"\nDESYNC at collective #{i}: ranks {missing} never issued it, "
                f"while the others did.\n"
                f"  Ranks must issue identical collective sequences. The ranks "
                f"listed above stopped early; the rest are blocked waiting on them."
            )
            return 1
        print(f"\nDESYNC at collective #{i}: ranks issued *different* collectives.")
        for rank in sorted(signatures):
            sig = signatures[rank]
            what = f"{sig[0]} ({sig[1]} bytes)" if sig else "nothing (stopped earlier)"
            print(f"    rank {rank}: {what}")
        print(
            "  Ranks must issue identical collective sequences. Look at what the "
            "odd rank out did differently just before this point (a conditional "
            "branch, an early return, an unequal batch count)."
        )
        return 1
    return 0


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
        print(f"  rank {s['rank']}: {counts}")

    streams = _order_streams(states)
    findings += _find_divergence(streams)

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
