"""Check that the test suite would actually notice if the library broke.

    python scripts/mutation_check.py            # all mutations
    python scripts/mutation_check.py --list     # just show them
    python scripts/mutation_check.py -k ring    # only matching ones

A passing test suite proves the code works on the cases it happens to cover. It
says nothing about whether the tests would *fail* if the code were wrong, and
those are different properties. A test that cannot fail is worse than no test,
because it reports confidence it has not earned.

So this deliberately breaks the library, one edit at a time, and asserts the
suite catches each break. Every mutation below is a mistake a person could
plausibly make: an off-by-one in a ring index, a forgotten division by the world
size, popping a queue from the wrong end, dropping a cross-stream dependency.
Two of them (``fsdp-rng`` and ``device-stream-order``) reproduce bugs this
project actually shipped and later fixed, so they also serve as a check that
those regression tests keep their value.

A mutation that survives is a gap in the tests, not a curiosity: it means that
line of reasoning is unprotected. The report lists survivors separately for that
reason.

Files are restored in a ``finally`` block, and the script verifies afterwards
that the working tree is byte-identical to how it started.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


@dataclass
class Mutation:
    name: str
    file: str
    old: str
    new: str
    breaks: str
    tests: list[str] = field(default_factory=list)
    #: Hardware the detecting test needs. A mutation whose test skips for want
    #: of a device has not been shown to survive, so it must not be reported as
    #: a gap: on a GPU-less runner the device tests skip and would otherwise
    #: make a perfectly good regression test look useless.
    requires: str | None = None
    #: True for a deliberate no-op. A checker that never reports a survivor
    #: cannot distinguish a strong suite from a broken checker, so one mutation
    #: is expected to survive as a self-test.
    expect_survival: bool = False


MUTATIONS: list[Mutation] = [
    Mutation(
        name="ring-send-index",
        file="mini_nccl/collectives.py",
        old="            send_idx = (r - step) % W\n            recv_idx = (r - step - 1) % W",
        new="            send_idx = (r - step + 1) % W\n            recv_idx = (r - step - 1) % W",
        breaks="off-by-one in the ring reduce-scatter schedule, so each rank "
        "forwards the wrong block",
        tests=["tests/test_collectives.py::test_collective_battery"],
    ),
    Mutation(
        name="ring-skip-allgather",
        file="mini_nccl/collectives.py",
        old="        # Phase 2 (all-gather): circulate the reduced blocks around the ring.\n"
        "        for step in range(W - 1):",
        new="        # Phase 2 (all-gather): circulate the reduced blocks around the ring.\n"
        "        for step in range(max(0, W - 2)):",
        breaks="one too few all-gather steps, so the last block never finishes "
        "circulating",
        tests=["tests/test_collectives.py::test_collective_battery"],
    ),
    Mutation(
        name="halving-swap-halves",
        file="mini_nccl/collectives.py",
        old="        if r & mask:\n            send_view, keep = padded[lo:mid], padded[mid:hi]",
        new="        if r & mask:\n            send_view, keep = padded[mid:hi], padded[lo:mid]",
        breaks="halving-doubling keeps the half it should have sent, so ranks "
        "reduce the wrong segments",
        tests=["tests/test_collectives.py::test_collective_battery"],
    ),
    Mutation(
        name="ddp-no-average",
        file="mini_nccl/ddp.py",
        old="                bucket.buffer.div_(self.pg.world_size)",
        new="                pass  # mutation: gradients summed, never averaged",
        breaks="DDP sums gradients instead of averaging them, so the effective "
        "learning rate scales with world size",
        tests=["tests/test_ddp.py::test_ddp_matches_single_process"],
    ),
    Mutation(
        name="ddp-no-overlap-dispatch",
        file="mini_nccl/ddp.py",
        old="            if self._overlap and self._reduce_future is None:",
        new="            if False and self._overlap and self._reduce_future is None:",
        breaks="the reducer is never dispatched during backward, so overlap "
        "silently stops happening",
        tests=["tests/test_overlap.py"],
    ),
    Mutation(
        name="fsdp-rng",
        file="mini_nccl/fsdp.py",
        old="                torch.set_rng_state(ctx.cpu_rng_state)",
        new="                pass  # mutation: recompute draws fresh randomness",
        breaks="FSDP's backward recompute draws a different dropout mask than "
        "its forward (a bug this project shipped)",
        tests=["tests/test_fsdp.py::test_recompute_replays_randomness"],
    ),
    Mutation(
        name="fsdp-no-grad-scale",
        file="mini_nccl/fsdp.py",
        old="        return shard_grad.div_(self.pg.world_size)",
        new="        return shard_grad",
        breaks="FSDP reduce-scatters gradients without averaging them",
        tests=["tests/test_fsdp.py::test_fsdp_matches_single_process"],
    ),
    Mutation(
        name="tp-no-backward-allreduce",
        file="mini_nccl/tensor_parallel.py",
        old="        pg: Communicator = ctx.pg\n        if pg.world_size > 1:\n"
        "            grad = grad.contiguous()\n            collectives.all_reduce(pg, grad)\n"
        "        return None, grad",
        new="        pg: Communicator = ctx.pg\n        return None, grad",
        breaks="column-parallel layers skip the backward all-reduce, so each "
        "rank keeps only its partial input gradient",
        tests=["tests/test_tensor_parallel.py::test_parallel_mlp_matches_reference_world2"],
    ),
    Mutation(
        name="pipeline-lifo-queue",
        file="mini_nccl/pipeline.py",
        old="        x, out = self._queue.popleft()",
        new="        x, out = self._queue.pop()",
        breaks="the pipeline matches each gradient to the newest microbatch "
        "instead of the oldest",
        tests=["tests/test_pipeline.py::test_1f1b_matches_single_process_four_stages"],
    ),
    Mutation(
        name="pipeline-warmup",
        file="mini_nccl/pipeline.py",
        old="        warmup = min(self.n_stages - 1 - self.rank, n_micro)",
        new="        warmup = min(self.n_stages - 1 - self.rank, n_micro) + 0 * self.rank",
        breaks="a no-op edit to the warmup formula (control: this one is "
        "expected to survive)",
        tests=["tests/test_pipeline.py::test_1f1b_bounds_in_flight_microbatches"],
        expect_survival=True,
    ),
    Mutation(
        name="mesh-transposed-strides",
        file="mini_nccl/mesh.py",
        old="        for name in reversed(self.names):",
        new="        for name in self.names:",
        breaks="mesh dimensions are laid out in the wrong order, so tensor "
        "groups are strided instead of contiguous",
        tests=["tests/test_mesh.py::test_mesh_layout_puts_neighbours_in_the_same_tensor_group"],
    ),
    Mutation(
        name="subgroup-no-translation",
        file="mini_nccl/mesh.py",
        old="    def _global(self, local_rank: int) -> int:\n        return self.global_ranks[local_rank]",
        new="    def _global(self, local_rank: int) -> int:\n        return local_rank",
        breaks="a subgroup talks to global ranks instead of its own members",
        tests=["tests/test_mesh.py::test_collectives_run_unchanged_on_a_subgroup"],
    ),
    Mutation(
        name="recorder-frozen-seq",
        file="mini_nccl/recorder.py",
        old="            seq = self._seq.get(channel, 0)\n            self._seq[channel] = seq + 1\n            return seq",
        new="            self._seq.setdefault(channel, 0)\n            return 0",
        breaks="every collective records sequence number 0, so desync detection "
        "can no longer line ranks up",
        tests=["tests/test_faults.py::test_desync_times_out_with_diagnosis"],
    ),
    Mutation(
        name="device-stream-order",
        file="mini_nccl/device.py",
        old="        if self.accelerated:\n            assert self.stream is not None\n"
        "            self.stream.wait_stream(torch.cuda.current_stream())",
        new="        if False:\n            assert self.stream is not None\n"
        "            self.stream.wait_stream(torch.cuda.current_stream())",
        breaks="the copy stream stops waiting for the reduction (a bug this "
        "project shipped; only detectable with a GPU)",
        tests=["tests/test_device.py"],
        requires="cuda",
    ),
    Mutation(
        name="transport-partial-recv",
        file="mini_nccl/transport.py",
        old="        while remaining:",
        new="        if remaining:",
        breaks="a receive stops looping, so any message larger than one TCP "
        "segment is silently truncated",
        tests=["tests/test_collectives.py::test_collective_battery"],
    ),
    Mutation(
        name="transport-handshake-identity",
        file="mini_nccl/transport.py",
        old="        conn.send(memoryview(_HANDSHAKE.pack(self.rank, channel)))",
        new="        conn.send(memoryview(_HANDSHAKE.pack(0, channel)))",
        breaks="every dialing rank claims to be rank 0, so connections are "
        "filed against the wrong peers",
        tests=["tests/test_collectives.py::test_collective_battery"],
    ),
    # The diagnostic tooling needs its own mutations: it is the part of the
    # project whose whole job is to be right when everything else is wrong, and
    # a broken diagnosis is worse than none because it points somewhere false.
    Mutation(
        name="diagnose-blind-to-divergence",
        file="mini_nccl/diagnose.py",
        old="        if len(set(signatures.values())) <= 1:\n            continue",
        new="        if True:\n            continue",
        breaks="the desync analysis never reports a divergence, so a hung job "
        "gets a clean bill of health",
        tests=["tests/test_faults.py::test_desync_times_out_with_diagnosis"],
    ),
    Mutation(
        name="diagnose-straggler-inverted",
        file="mini_nccl/diagnose.py",
        old="            if value * factor < across:",
        new="            if value > factor * across:",
        breaks="straggler detection blames the ranks that were waiting instead "
        "of the one they waited for (a mistake I made writing it)",
        tests=["tests/test_overlap.py::test_diagnose_names_a_straggler"],
    ),
    Mutation(
        name="launcher-swallow-errors",
        file="mini_nccl/launcher.py",
        old="    if errors:\n        detail =",
        new="    if False:\n        detail =",
        breaks="worker failures are silently discarded, so every test would "
        "pass no matter what the ranks did",
        tests=["tests/test_faults.py::test_dead_rank_fails_fast"],
    ),
]


class Patch:
    """Applies one edit and puts the file back byte for byte.

    Deliberately byte-oriented. An earlier version read and wrote text, which
    on Windows rewrote every mutated file with CRLF line endings and left the
    tree dirty afterwards. Worse, its own "restored" check compared
    newline-normalized text, so it could not see the one thing it had changed.
    A tool that edits a repository has to be exact about it, and has to verify
    the claim it makes.
    """

    def __init__(self, mutation: Mutation) -> None:
        self.path = ROOT / mutation.file
        self.mutation = mutation
        self.original = self.path.read_bytes()
        text = self.original.decode("utf-8")
        # Match the file's own line-ending convention rather than imposing
        # one, so a CRLF checkout is edited and restored exactly as found.
        lf, crlf = chr(10), chr(13) + chr(10)
        uses_crlf = crlf in text
        self.text = text
        self.old = mutation.old.replace(lf, crlf) if uses_crlf else mutation.old
        self.new = mutation.new.replace(lf, crlf) if uses_crlf else mutation.new

    def __enter__(self) -> bool:
        if self.old not in self.text:
            return False
        mutated = self.text.replace(self.old, self.new, 1)
        self.path.write_bytes(mutated.encode("utf-8"))
        return True

    def __exit__(self, *exc) -> None:
        self.path.write_bytes(self.original)


def run_tests(targets: list[str], timeout: float) -> tuple[bool, str]:
    """True if the tests passed (meaning the mutation went unnoticed)."""
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", *targets, "-x", "-q", "--no-header"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    tail = [line for line in proc.stdout.splitlines() if line.strip()][-1:]
    return proc.returncode == 0, tail[0] if tail else "(no output)"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-k", dest="filter", help="only mutations whose name contains this")
    ap.add_argument("--list", action="store_true", help="show the mutations and exit")
    ap.add_argument("--timeout", type=float, default=900.0)
    ap.add_argument(
        "--strict",
        action="store_true",
        help="also fail if a mutation could not be applied, so the checker "
        "cannot quietly rot as the code it points at moves",
    )
    args = ap.parse_args()

    selected = [m for m in MUTATIONS if not args.filter or args.filter in m.name]
    if args.list:
        for m in selected:
            print(f"{m.name:28s} {m.file:32s} {m.breaks}")
        return

    have_cuda = _cuda_available()
    baseline = {m.file: (ROOT / m.file).read_bytes() for m in selected}

    caught: list[tuple[Mutation, str]] = []
    survived: list[tuple[Mutation, str]] = []
    stale: list[Mutation] = []
    unsupported: list[Mutation] = []

    for i, mutation in enumerate(selected, 1):
        print(f"[{i}/{len(selected)}] {mutation.name} ... ", end="", flush=True)
        if mutation.requires == "cuda" and not have_cuda:
            # Its detecting test would skip, which is not the same as failing to
            # catch anything.
            print("skipped (needs a CUDA device)")
            unsupported.append(mutation)
            continue
        start = time.perf_counter()
        patch = Patch(mutation)
        try:
            with patch as applied:
                if not applied:
                    print("STALE (anchor text not found; the code moved)")
                    stale.append(mutation)
                    continue
                passed, summary = run_tests(mutation.tests, args.timeout)
        except subprocess.TimeoutExpired:
            passed, summary = False, "timed out (a hang counts as caught)"
        elapsed = time.perf_counter() - start
        if passed:
            label = "SURVIVED (expected)" if mutation.expect_survival else "SURVIVED"
            note = "" if mutation.expect_survival else " <- the tests did not notice"
            print(f"{label} ({elapsed:.0f}s){note}")
            survived.append((mutation, summary))
        else:
            print(f"caught ({elapsed:.0f}s)")
            caught.append((mutation, summary))

    # The whole point is that this leaves nothing behind.
    for file, content in baseline.items():
        current = (ROOT / file).read_bytes()
        if current != content:
            print(f"\nFATAL: {file} was not restored. Run `git checkout -- {file}`.")
            raise SystemExit(2)
    print("\nall files restored byte-for-byte.")

    real = [m for m in selected if not m.expect_survival and m not in unsupported]
    detected = [m for m, _ in caught if not m.expect_survival]
    print(f"\ncaught {len(detected)}/{len(real)} real mutations")
    if unsupported:
        print(f"  {len(unsupported)} skipped for missing hardware: "
              f"{', '.join(m.name for m in unsupported)}")
    if caught:
        print("\n| mutation | what it breaks | caught by |")
        print("|---|---|---|")
        for m, _ in caught:
            target = m.tests[0].split("::")[-1] if m.tests else ""
            print(f"| `{m.name}` | {m.breaks} | `{target}` |")

    unexpected = [(m, s) for m, s in survived if not m.expect_survival]
    expected = [m for m, _ in survived if m.expect_survival]
    if expected:
        print(f"\ncontrol survived as designed: {', '.join(m.name for m in expected)}")
        print("  (a checker that never reports a survivor cannot be trusted to)")
    if unexpected:
        print("\nSURVIVORS (each is an untested line of reasoning):")
        for m, summary in unexpected:
            print(f"  {m.name}: {m.breaks}")
            print(f"    ran {m.tests} -> {summary}")
    if stale:
        print("\nSTALE (the code moved; update the anchor or drop the mutation):")
        for m in stale:
            print(f"  {m.name} in {m.file}")

    failures = len(unexpected) + (len(stale) if args.strict else 0)
    raise SystemExit(1 if failures else 0)


def _cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


if __name__ == "__main__":
    main()
