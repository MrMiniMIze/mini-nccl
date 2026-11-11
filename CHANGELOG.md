# Changelog

Newest first. Each entry says what changed and why it matters; the commit
history has the detail.

## Unreleased

- A peer whose connection drops or resets now raises `PeerClosedError`
  carrying the same local state a timeout does. Before, only timeouts were
  translated, and a dead peer surfaced as a bare `ConnectionResetError`.
- Rendezvous validates every handshake: magic number, protocol version, the
  claimed rank and channel, and a fingerprint of the job configuration. A
  refused dialer is told why. A silent connection is dropped after 5 s instead
  of holding the acceptor for the 300 s operation timeout, and a stray's bytes
  can no longer be filed as a rank. Mismatched host lists now fail at startup
  by name.
- Check the README's numbers instead of maintaining them.
  `scripts/check_readme_claims.py` derives the test, module, mutation and
  protocol-member counts and the version from the code, fails naming the line
  of any that drifted, and runs in CI. It has its own mutation, so a checker
  that stops checking is caught.
- Single-source the version: `pyproject.toml` reads `mini_nccl.__version__`.
- Declare `numpy` as a runtime dependency. Every send goes through
  `Tensor.numpy()`, but numpy was only listed under the `dev` extra, so a plain
  `pip install` could not move a byte. A new CI job installs the built wheel
  with no extras and runs a real all-reduce, which is the check that would have
  caught it.
- `mypy` and `coverage` are now in the `dev` extra, so the README's own
  commands work after `pip install -e .[dev]`. The mypy settings live in
  `pyproject.toml`, and the package ships `py.typed`.
- CI tests Python 3.10, 3.12 and 3.13, adds macOS beside Linux and Windows, and
  enforces at least 90% coverage instead of only reporting it in the README.
- Add a contents index to the README and say what the project demonstrates.

## 0.9.0 (2026-07-30)

- Put every Quickstart command under test: the examples are smoke-run in CI on
  a synthetic corpus, so the first thing a visitor runs cannot silently break.
- Extend the mutation checker to the diagnostic tooling (desync and straggler
  analysis) and the launcher.

## 0.8.0 (2026-07-29)

- Verify the tests by breaking the library on purpose: `scripts/mutation_check.py`
  applies one plausible mistake at a time and requires the suite to notice. It
  runs in CI, and it caught the device stream-ordering test passing only when
  the timing cooperated, so that test now asserts the mechanism instead.

## 0.7.0 (2026-07-29)

- Fix FSDP's backward recompute drawing a different dropout mask than its
  forward, which gave silently wrong gradients for the sharded units only.
- Fix a failed rendezvous leaking its listening socket, so a same-process retry
  died with "address already in use" on its own port.
- Make the interface claim checkable: `Communicator` is a `Protocol`, and both
  `ProcessGroup` and `SubGroup` must satisfy it. mypy is clean and runs in CI.

## 0.6.1 (2026-07-29)

- Assert that DDP dispatches a bucket reduction during backward, rather than
  that the OS scheduled it there; the old test failed on a two-core CI runner.

## 0.6.0 (2026-07-29)

- Add 3D parallelism (data x pipeline x tensor) on eight ranks, verified
  gradient by gradient against single-process training.

## 0.5.0 (2026-07-29)

- Compose the strategies: `SubGroup` communicators as rank-translating views
  over the parent's sockets, and `ParallelMesh` for named 2D layouts.

## 0.4.0 (2026-07-29)

- Add pipeline parallelism with GPipe and 1F1B schedules, completing all four
  strategies.
- Add a device path for accelerator tensors: pinned host staging and a chunked
  copy/network pipeline. Verified on a real GPU, which found a cross-stream
  ordering race no CPU test could reach, and bounded by measurement: pipelining
  the copy can buy at most about 1.1x over loopback.

## 0.3.0 (2026-07-29)

- Add fully sharded data parallelism and tensor parallelism on the same
  primitives.
- Add a bfloat16 wire option, and document the measured result: the error
  comes from re-rounding at each hop, so tree beats ring on accuracy.

## 0.2.0 (2026-07-29)

First tagged release.

- TCP full-mesh transport, process group, and a multi-process launcher.
- Ring, binomial-tree, recursive halving-doubling and naive all-reduce, plus
  reduce-scatter, all-gather, broadcast, all-to-all and barrier.
- Channels, with thresholds tuned from an ablation and an alpha-beta cost model.
- DistributedDataParallel with gradient bucketing, overlap, and `no_sync()`.
- A flight recorder (bounded ring buffer) with desync diagnosis and Perfetto
  traces, multi-node rendezvous from environment variables, and a
  character-level GPT trained entirely through the library.
