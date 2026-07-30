"""Smoke-run the examples, because they are the first thing anyone runs.

The README opens with a list of commands. If one of them is broken, that is the
first impression the project makes, and nothing else in the suite would notice:
the examples exercise wiring the unit tests do not (argument parsing, stage
construction, the mesh factorisation, loading a corpus, sampling text).

This project has already refactored every module onto a shared protocol, moved
gradient averaging into a helper, and changed the pipeline's constructor. Any of
those could have broken an example silently, so the check belongs in CI rather
than in my memory of having run them once.

These are smoke tests, deliberately. They assert the command exits cleanly and
prints the one line that proves it did the work, not that training converged;
the parity tests elsewhere cover correctness. Everything is sized for speed, so
a failure means "this example is broken", not "this example is slow".
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
CORPUS = ROOT / "data" / "tinyshakespeare.txt"

# Every example takes the same shrink-to-nothing treatment.
TINY = "--steps 2 --n-layer 2 --n-embd 32 --batch-size 4 --block-size 16"

EXAMPLES = [
    pytest.param(
        f"examples/train_gpt.py --world-size 2 {TINY}",
        "final loss:",
        id="train_gpt_ddp",
    ),
    pytest.param(
        f"examples/train_gpt.py --world-size 2 --fsdp {TINY}",
        "FSDP peak total:",
        id="train_gpt_fsdp",
    ),
    pytest.param(
        f"examples/tensor_parallel_gpt.py --world-size 2 {TINY}",
        "max disagreement:",
        id="tensor_parallel",
    ),
    pytest.param(
        f"examples/pipeline_gpt.py --world-size 2 --microbatches 2 {TINY}",
        "microbatches in flight per stage",
        id="pipeline",
    ),
    pytest.param(
        f"examples/two_dimensional_gpt.py --world-size 2 --tp 2 {TINY}",
        "loss across the tensor group",
        id="two_dimensional",
    ),
    # The 3D example is covered by test_three_dimensional_mesh_is_orthogonal
    # below, which runs it once and checks something specific rather than
    # spawning four processes twice for the same coverage.
]


@pytest.fixture(scope="module", autouse=True)
def corpus() -> None:
    """Stand in for tiny shakespeare so the examples never hit the network.

    ``load_corpus`` only downloads when the file is missing, so writing a small
    synthetic corpus here makes these tests hermetic and fast. A real corpus, if
    one is already present, is left alone.
    """
    if CORPUS.exists():
        return
    CORPUS.parent.mkdir(parents=True, exist_ok=True)
    # Enough distinct characters for a vocabulary, and long enough to sample
    # blocks from without running off the end.
    text = ("To be, or not to be, that is the question:\n"
            "Whether 'tis nobler in the mind to suffer\n") * 200
    CORPUS.write_text(text, encoding="utf-8")


def run_example(command: str, expect: str, timeout: float = 420.0) -> str:
    proc = subprocess.run(
        [sys.executable, *command.split()],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if proc.returncode != 0:
        tail = "\n".join((proc.stdout + proc.stderr).splitlines()[-25:])
        pytest.fail(f"`{command}` exited {proc.returncode}:\n{tail}")
    if expect not in proc.stdout:
        tail = "\n".join(proc.stdout.splitlines()[-25:])
        pytest.fail(f"`{command}` ran but never printed {expect!r}:\n{tail}")
    return proc.stdout


@pytest.mark.examples
@pytest.mark.parametrize(("command", "expect"), EXAMPLES)
def test_example_runs(command: str, expect: str) -> None:
    run_example(command, expect)


@pytest.mark.examples
def test_desync_demo_produces_a_diagnosis() -> None:
    """The demo has to actually demonstrate something, not just exit cleanly."""
    out = run_example("examples/desync_demo.py", "DESYNC at collective")
    # The point of the demo is naming the rank that went its own way.
    assert "rank 2: barrier" in out, out[-800:]
    assert "CollectiveTimeoutError" in out, out[-800:]


@pytest.mark.examples
def test_three_dimensional_mesh_is_orthogonal() -> None:
    """Runs the 3D example, and checks the mesh it reports actually is a mesh.

    pp=2 x tp=2 leaves dp=1, which keeps this to four processes instead of eight
    while still exercising all three dimensions.
    """
    out = run_example(
        f"examples/three_dimensional_gpt.py --world-size 4 --pp 2 --tp 2 "
        f"--microbatches 2 {TINY}",
        "pipeline group",
    )
    # Match on "pipeline group": the loss-spread line also mentions the tensor
    # group, and output from several ranks arrives interleaved.
    line = next(line for line in out.splitlines() if "pipeline group" in line)
    # literal_eval, not eval: this is parsing a printed list, and a test has no
    # business executing whatever the string happens to contain.
    groups = [
        set(ast.literal_eval(part.split("group")[1].strip()))
        for part in line.split("|")
    ]
    shared = set.intersection(*groups)
    assert len(shared) == 1, f"mesh dimensions overlap by more than one rank: {line}"
