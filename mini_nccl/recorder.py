"""Flight recorder for collective operations.

Every collective is stamped with a per-channel sequence number and timed.
Two things fall out of that:

- **Traces.** ``dump_chrome_trace`` writes Trace Event Format JSON, so a run
  can be opened in Perfetto or ``chrome://tracing`` and you can *see* ring
  channels running in parallel and DDP buckets reducing while backward is
  still computing.
- **Desync diagnosis.** Ranks must issue collectives in identical order. If
  one skips or reorders a collective, the others block on a peer that is
  waiting for something else. When that times out, the recorder answers the
  only question worth asking: which rank was on which sequence number, and
  what was it still waiting for.

Recording is off unless asked for, either with ``ProcessGroup(trace=True)``
or by setting ``MINI_NCCL_TRACE=1``. When enabled, the per-collective cost
is one timestamp pair and a list append.
"""

from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Event:
    op: str
    algorithm: str
    channel: int
    seq: int
    nbytes: int
    start_ns: int
    end_ns: int | None = None
    detail: dict = field(default_factory=dict)

    @property
    def duration_us(self) -> float:
        end = self.end_ns if self.end_ns is not None else time.perf_counter_ns()
        return (end - self.start_ns) / 1e3

    def describe(self) -> str:
        state = "completed" if self.end_ns is not None else "PENDING"
        return (
            f"{self.op}[{self.algorithm}] seq={self.seq} channel={self.channel} "
            f"bytes={self.nbytes} {state} after {self.duration_us / 1e3:.1f}ms"
        )


def trace_enabled_by_env() -> bool:
    return os.environ.get("MINI_NCCL_TRACE", "").lower() in {"1", "true", "yes", "on"}


class Recorder:
    """Thread-safe event log for one rank.

    Sequence numbers are per channel, because that is the ordering ranks
    must agree on: two collectives on different channels are genuinely
    concurrent, two on the same channel are not.
    """

    def __init__(self, rank: int, world_size: int, enabled: bool = False) -> None:
        self.rank = rank
        self.world_size = world_size
        self.enabled = enabled
        self._events: list[Event] = []
        self._seq: dict[int, int] = {}
        self._lock = threading.Lock()
        self._t0_ns = time.perf_counter_ns()
        self._t0_wall = time.time()

    def next_seq(self, channel: int) -> int:
        with self._lock:
            seq = self._seq.get(channel, 0)
            self._seq[channel] = seq + 1
            return seq

    def start(
        self,
        name: str,
        algorithm: str = "",
        channel: int = 0,
        nbytes: int = 0,
        **detail,
    ) -> Event | None:
        """Open an event. Returns None when recording is disabled.

        Extra keyword arguments are stored as trace annotations, so ``name``
        rather than ``op`` is the parameter here: ``op="sum"`` is a legitimate
        annotation a caller will want to pass.
        """
        if not self.enabled:
            return None
        ev = Event(
            op=name,
            algorithm=algorithm,
            channel=channel,
            seq=self.next_seq(channel),
            nbytes=nbytes,
            start_ns=time.perf_counter_ns(),
            detail=detail,
        )
        with self._lock:
            self._events.append(ev)
        return ev

    def finish(self, ev: Event | None) -> None:
        if ev is None:
            return
        ev.end_ns = time.perf_counter_ns()

    def pending(self) -> list[Event]:
        with self._lock:
            return [ev for ev in self._events if ev.end_ns is None]

    def last_completed(self) -> Event | None:
        with self._lock:
            done = [ev for ev in self._events if ev.end_ns is not None]
        return done[-1] if done else None

    def context(self) -> str:
        """One-line summary of local state, for error messages."""
        if not self.enabled:
            return "flight recorder disabled (set MINI_NCCL_TRACE=1 for details)"
        pending = self.pending()
        last = self.last_completed()
        parts = [f"rank {self.rank}"]
        parts.append(
            "in flight: " + "; ".join(ev.describe() for ev in pending)
            if pending
            else "in flight: nothing"
        )
        parts.append(f"last completed: {last.describe()}" if last else "last completed: none")
        with self._lock:
            next_seqs = dict(sorted(self._seq.items()))
        parts.append(f"next seq per channel: {next_seqs}")
        return " | ".join(parts)

    # ---- output ----------------------------------------------------------

    def state(self) -> dict:
        """Machine-readable snapshot, one file per rank (see diagnose.py)."""
        return {
            "rank": self.rank,
            "world_size": self.world_size,
            "wall_start": self._t0_wall,
            "next_seq": self._seq,
            "pending": [self._event_dict(ev) for ev in self.pending()],
            "events": [self._event_dict(ev) for ev in self._events],
        }

    def _event_dict(self, ev: Event) -> dict:
        return {
            "op": ev.op,
            "algorithm": ev.algorithm,
            "channel": ev.channel,
            "seq": ev.seq,
            "nbytes": ev.nbytes,
            "start_us": (ev.start_ns - self._t0_ns) / 1e3,
            "dur_us": ev.duration_us if ev.end_ns is not None else None,
            "wall_start": self._t0_wall + (ev.start_ns - self._t0_ns) / 1e9,
            "detail": ev.detail,
        }

    def dump(self, directory: str | Path) -> Path:
        """Write this rank's state to ``<directory>/rank<N>.json``."""
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        path = d / f"rank{self.rank}.json"
        path.write_text(json.dumps(self.state(), indent=1), encoding="utf-8")
        return path

    def chrome_trace_events(self, wall_zero: float | None = None) -> list[dict]:
        """Trace Event Format records: one process per rank, one thread per channel."""
        base = wall_zero if wall_zero is not None else self._t0_wall
        out: list[dict] = [
            {
                "name": "process_name",
                "ph": "M",
                "pid": self.rank,
                "args": {"name": f"rank {self.rank}"},
            }
        ]
        seen_channels = set()
        for ev in self._events:
            if ev.channel not in seen_channels:
                seen_channels.add(ev.channel)
                out.append(
                    {
                        "name": "thread_name",
                        "ph": "M",
                        "pid": self.rank,
                        "tid": ev.channel,
                        "args": {
                            "name": "collective" if ev.channel < 0 else f"channel {ev.channel}"
                        },
                    }
                )
            start_wall = self._t0_wall + (ev.start_ns - self._t0_ns) / 1e9
            out.append(
                {
                    "name": f"{ev.op}:{ev.algorithm}" if ev.algorithm else ev.op,
                    "cat": "collective",
                    "ph": "X",
                    "ts": (start_wall - base) * 1e6,
                    "dur": ev.duration_us,
                    "pid": self.rank,
                    "tid": ev.channel,
                    "args": {"seq": ev.seq, "bytes": ev.nbytes, **ev.detail},
                }
            )
        return out

    def dump_chrome_trace(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.chrome_trace_events()), encoding="utf-8")
        return p
