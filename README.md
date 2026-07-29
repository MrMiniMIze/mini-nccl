# mini-nccl

[![ci](https://github.com/MrMiniMIze/mini-nccl/actions/workflows/ci.yml/badge.svg)](https://github.com/MrMiniMIze/mini-nccl/actions/workflows/ci.yml)

**Collective communication from first principles.** A small, readable
reimplementation of the machinery inside libraries like NCCL: ring,
binomial-tree, and recursive halving-doubling all-reduce, reduce-scatter,
all-gather, broadcast, and all-to-all, built on nothing but TCP sockets and
PyTorch tensors. On top of that sits a bucketed-overlap
`DistributedDataParallel`, a flight recorder that turns a hung job into a
named culprit, and a character-level GPT whose every gradient byte moves
through this library rather than `torch.distributed`.

No `torch.distributed`, no MPI, no NCCL underneath. The goal is to make the
machinery of distributed training small enough to read in an afternoon and
measured well enough to trust, including where it loses.

![all-reduce bus bandwidth](docs/img/allreduce_busbw.png)

## What's inside

| Layer | File | What it does |
|---|---|---|
| Transport | `mini_nccl/transport.py` | Full-mesh TCP rendezvous, N connections per peer ("channels"), zero-copy receives straight into tensor storage, bounded operation timeouts |
| Process group | `mini_nccl/process_group.py` | `send` / `recv` / full-duplex `send_recv` / sliced `send_recv`: the primitives everything else is built from |
| Collectives | `mini_nccl/collectives.py` | `all_reduce` (ring, tree, halving-doubling, naive), `reduce_scatter`, `all_gather`, `broadcast`, `all_to_all`, `barrier` |
| DDP | `mini_nccl/ddp.py` | Gradient bucketing, grads-as-bucket-views, overlap on a reducer thread, `no_sync()` accumulation |
| Flight recorder | `mini_nccl/recorder.py`, `diagnose.py` | Sequence-numbered collective log, Perfetto traces, desync diagnosis |
| Launcher | `mini_nccl/launcher.py` | `mn.run(fn, world_size)`: spawn, rendezvous, collect results, notice dead ranks |
| Example | `examples/train_gpt.py` | Char-level GPT trained data-parallel through mini-nccl |
| Benchmarks | `benchmarks/` | nccl-tests-style sweep, tuning ablation, alpha-beta cost model fit |

## Quickstart

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e .[dev]

pytest -q                                        # 18 tests, world sizes 2-4

python examples/train_gpt.py --world-size 4 --steps 200 --sample
python examples/desync_demo.py                   # a hang, diagnosed
python benchmarks/bench_allreduce.py --world-sizes 2,4 --gloo
python benchmarks/bench_ablation.py              # channel + pipeline tuning
python benchmarks/fit_cost_model.py              # alpha-beta fit
```

The API mirrors `torch.distributed`:

```python
import mini_nccl as mn

def worker(pg):
    t = torch.ones(1024) * pg.rank
    mn.collectives.all_reduce(pg, t)          # in place; algorithm="auto"
    mn.collectives.all_reduce(pg, t, algorithm="halving")
    mn.collectives.broadcast(pg, t, src=0)

mn.run(worker, world_size=4)
```

## Results

`bench_allreduce.py` follows nccl-tests conventions: per-config time is the
max across ranks (the slowest rank defines collective latency), and bus
bandwidth for all-reduce is `algbw · 2(W-1)/W`. Loopback TCP, CPU tensors,
time per all-reduce in milliseconds, best in bold:

**2 ranks**

| size | ring | tree | halving | naive | gloo |
|---|---|---|---|---|---|
| 64 KiB | 0.55 | **0.24** | 0.80 | 0.26 | 2.41 |
| 256 KiB | 1.17 | **0.46** | 0.84 | 0.47 | 1.28 |
| 1 MiB | **1.64** | 1.64 | 2.19 | 1.83 | 8.76 |
| 16 MiB | **40.9** | 50.5 | 45.2 | 48.1 | 89.6 |
| 64 MiB | **97.0** | 107.1 | 101.8 | 114.3 | 447.6 |

**4 ranks**

| size | ring | tree | halving | naive | gloo |
|---|---|---|---|---|---|
| 64 KiB | 1.41 | 0.57 | 1.11 | **0.34** | 1.58 |
| 1 MiB | 3.16 | 3.08 | 3.01 | **2.89** | 3.72 |
| 4 MiB | **10.8** | 25.9 | 16.1 | 26.0 | 14.3 |
| 16 MiB | **49.9** | 104.6 | 91.7 | 87.1 | 58.0 |
| 64 MiB | 257.5 | 321.9 | 255.4 | 293.3 | **222.2** |

Against `torch.distributed`'s gloo backend, mini-nccl's ring is faster at
every size at 2 ranks (**4.6x** at 64 MiB) and at 4 ranks up to 16 MiB;
gloo takes the largest 4-rank case, where its chunk pipelining pays off.

![all-reduce latency](docs/img/allreduce_latency.png)

### Tuning, not guessing: the channel ablation

A "channel" is an independent connection between the same pair of ranks, the
mechanism NCCL uses to drive one collective over several parallel paths. The
tensor is split across channels, each with its own thread and socket. How
many channels, and whether to also pipeline the reduction inside each ring
step, are empirical questions, so `bench_ablation.py` answers them
(ring all-reduce, 4 ranks, speedup against 1 channel):

| configuration | 1 MiB | 4 MiB | 16 MiB | 64 MiB |
|---|---|---|---|---|
| 1 channel | 5 ms | 20 ms | 94 ms | 301 ms |
| **2 channels** | 6 ms (0.79x) | 19 ms (1.06x) | 65 ms (**1.45x**) | 259 ms (1.16x) |
| 4 channels | 8 ms (0.60x) | 30 ms (0.67x) | 77 ms (1.22x) | 259 ms (1.16x) |
| 8 channels | 7 ms (0.63x) | 26 ms (0.77x) | 86 ms (1.10x) | 252 ms (1.20x) |

Two channels won, and only above a few MiB, so the defaults became two
channels with an 8 MiB split threshold. Past two, extra threads compete for
the same cores that are doing the reduction, and below 8 MiB the thread
handoff costs more than the parallel socket gains.

The second optimization did not survive contact with measurement:

| configuration | 1 MiB | 4 MiB | 16 MiB | 64 MiB |
|---|---|---|---|---|
| pipeline off | 7 ms | 26 ms | 86 ms | 252 ms |
| pipeline on | 8 ms (0.95x) | 29 ms (0.90x) | 99 ms (0.86x) | 427 ms (0.59x) |

Reducing each slice while the next arrives is a real technique, but it only
pays when the reduction runs on hardware the transfer is not using. That is
exactly NCCL's situation (GPU kernels while the NIC does DMA) and exactly not
this one, where "the network" is a memory copy performed by the same cores.
So the mechanism ships **off** by default, still reachable through
`MINI_NCCL_MAX_SLICES=16` for the transport where it belongs.

### The alpha-beta cost model

Fitting `t(n) = α·steps + β·bytes_on_critical_path(n)` to the measurements
recovers each algorithm's latency and per-byte cost from data. Because every
size from 4 KiB to 64 MiB should carry equal weight, the fit minimizes
*relative* error (an absolute-error fit is decided entirely by the largest
points, and the latency term ends up fitted to noise).

4 ranks:

| algorithm | steps | payload moved | α per step | β per byte | implied bandwidth | mean error |
|---|---|---|---|---|---|---|
| ring | 6 | 1.50n | 197 µs | 1.63 ns | 0.61 GB/s | 20% |
| tree | 4 | 4.00n | 82 µs | 0.79 ns | 1.27 GB/s | 29% |
| halving | 4 | 1.50n | 221 µs | 2.16 ns | 0.46 GB/s | 16% |
| naive | 6 | 6.00n | 35 µs | 0.50 ns | 1.98 GB/s | 29% |

![cost model vs measurement](docs/img/cost_model.png)

Three things fall out of this table, and the third is the one worth
remembering:

1. **Tree wins small messages** because its `α·steps` product is the
   smallest, and the fitted model puts the tree/ring crossover at 1.1 MiB
   against a measured 4 MiB. Same order of magnitude from a two-parameter
   model, which is about all one should claim for it.
2. **The naive baseline has the *best* per-byte efficiency** (1.98 GB/s),
   because rank 0 does long uninterrupted transfers with no dependency
   stalls. It still loses at scale, because it moves 4x more data (`6n`
   against ring's `1.5n`).
3. **Ring wins by moving less data, not by moving data faster.** Its
   per-byte efficiency is the *worst* of the four, since every step waits on
   the previous one and the socket never gets to stream. That is the whole
   argument for ring in one number: it is a bandwidth-*volume* optimization,
   and it is why ring's advantage should widen on a real fabric where bytes
   actually cost something, which is what `docs/multinode.md` sets up.

## How it works

### Ring all-reduce

The tensor is split into `W` blocks. In `W-1` **reduce-scatter** steps, each
rank sends the block it just accumulated to its right neighbor while
receiving and reducing the next block from its left. After that phase every
rank holds one fully reduced block. `W-1` **all-gather** steps then circulate
the finished blocks around the ring.

Every rank sends `2(W-1)/W · n` bytes total, asymptotically independent of
world size: add workers and per-rank traffic stays flat. The cost is
`2(W-1)` serialized steps of latency.

Both directions of every step run concurrently. `send_recv` pushes to the
right neighbor from a send thread while the calling thread blocks on the
receive from the left (CPython releases the GIL inside socket syscalls, so
this is real full duplex, not time slicing).

### Binomial tree all-reduce

Latency-optimal: reduce up a binomial tree to rank 0 in `⌈log₂W⌉` steps,
broadcast back down in `⌈log₂W⌉` more. Interior ranks forward the full
payload, so bytes moved grow with `log W`: worse than ring for large tensors,
unbeatable for small ones where per-message latency dominates. Works for any
world size, not just powers of two, via virtual rank renumbering.

### Recursive halving-doubling

Rabenseifner's algorithm, and the best of both: reduce-scatter by recursively
halving the working segment with the partner at distance `W/2, W/4, …, 1`,
then all-gather by recursively doubling back out. That is `2log₂W` steps
(tree's latency) while moving `2(W-1)/W · n` bytes (ring's volume). The
segment a rank ends up owning is exactly its own index, because the splits
walk from the most significant bit down, which is what makes the doubling
phase a simple reversal.

It requires a power-of-two world size. Rather than silently doing something
different, non-power-of-two world sizes fall back to ring; the general
version needs an extra fold-in step for the remainder ranks.

### Algorithm selection

`algorithm="auto"` picks tree at or below 1 MiB and ring above it, the same
latency-versus-bandwidth decision NCCL's tuner makes, with the threshold
taken from the table above rather than from intuition.

Note what `auto` deliberately does *not* do: on loopback the naive baseline
is fastest at small sizes (0.34 ms against tree's 0.57 ms at 4 ranks), but
its step count grows linearly with world size while tree's grows
logarithmically, so that ordering reverses well before real cluster scale.
Tuning to a two-rank laptop measurement would be tuning to the wrong machine.

### The naive baseline

`algorithm="naive"` is the parameter-server pattern: every rank sends to rank
0, which reduces and sends results back, moving `O(W · n)` bytes through one
rank. It exists so the benchmarks can show *why* the other three do what they
do, and it is honest about looking good on loopback at small sizes.

### DDP: bucketing, gradient views, overlap

`mini_nccl.DistributedDataParallel` reimplements the architecture of
PyTorch's DDP:

- **Buckets.** Parameters are grouped in reverse registration order
  (approximately the order gradients are produced in backward) into
  ~`bucket_cap_mb` flat buffers. One all-reduce per bucket instead of one per
  tensor amortizes latency.
- **Gradient views.** Each `param.grad` is a view into its bucket's flat
  buffer, so autograd accumulates directly into the communication buffer,
  with no flatten/unflatten copies.
- **Overlap.** A post-accumulate-grad hook marks buckets ready; a dedicated
  reducer thread all-reduces each as it completes, while backward is still
  computing earlier layers.
- **Determinism invariant.** The reducer walks buckets in fixed index order
  regardless of readiness order, so every rank issues an identical collective
  sequence. This is the same invariant NCCL communicators require, and
  violating it deadlocks rather than fails.
- **`no_sync()`** accumulates gradients across microbatches without touching
  the wire, then one reduction covers them all.

Correctness is enforced by the strictest test a DDP can face: training on `W`
processes with `1/W` of the batch each must produce the same parameters as
single-process full-batch training, step for step, including with 1 KiB
buckets, with overlap on and off, and across `no_sync()` boundaries.

**What overlap buys here: about 2%.** That is the correct answer for this
environment and worth understanding rather than hiding. Overlap pays when
communication uses a resource distinct from compute (a NIC with DMA, a GPU
copy engine). On loopback, the transfer *is* memcpy on the cores backward
needs, so there is nothing independent to overlap with. The mechanism is what
matters; the payoff appears when the transport stops sharing silicon with the
model.

## Reliability: a hang you can debug

The worst outcome in distributed training is not a crash, it is a hang: the
allocation burns while every rank sits in `recv` and no log says why. Two
mechanisms turn that into a diagnosis.

**Bounded waits.** Every socket carries an operation timeout, so a dead or
desynchronized peer raises `CollectiveTimeoutError` carrying the local rank's
view of what it was waiting for.

**A flight recorder.** With `MINI_NCCL_TRACE=1` (or `run(..., trace_dir=...)`)
every collective is sequence-numbered per channel and timed. Because ranks
must issue the same collectives in the same order, comparing those streams
finds the divergence exactly. `examples/desync_demo.py` injects the classic
bug, a rank-dependent branch around a collective, and gets:

```
mini_nccl.errors.CollectiveTimeoutError: rank 2: recv on channel 0 to/from rank 0 exceeded 5.0s.
  local state: rank 2 | in flight: barrier[tree] seq=2 channel=-1 PENDING after 5002.8ms |
               last completed: all_reduce[tree] seq=1 channel=-1 bytes=16384 completed after 0.4ms
  likely cause: rank 0 is dead, stuck, or issuing a different collective

$ python -m mini_nccl.diagnose desync_trace
DESYNC at collective #2: ranks issued *different* collectives.
    rank 0: all_reduce (16384 bytes)
    rank 1: all_reduce (16384 bytes)
    rank 2: barrier (0 bytes)
  Ranks must issue identical collective sequences. Look at what the odd rank
  out did differently just before this point (a conditional branch, an early
  return, an unequal batch count).
```

`--trace merged.json` merges every rank's events into Trace Event Format for
Perfetto or `chrome://tracing`, one process per rank and one track per
channel, which is how you *see* DDP buckets reducing while backward is still
running. `train_gpt.py --trace-dir` produces one from a real training run.

The launcher also distinguishes a dead job from a hung one: a rank killed
hard (`os._exit`, OOM killer, segfault) never reports, so `run()` watches
process exits and says `rank 1 exited with code 9 without reporting` in
about a second instead of waiting out the timeout.

## The proof: a GPT trained entirely through mini-nccl

`examples/train_gpt.py` trains a ~1M-parameter character-level GPT on tiny
shakespeare, data-parallel, with gradients averaged by this library's own
ring all-reduce. `torch.distributed` is never imported. Two ranks on a laptop
CPU:

```
step    0 | loss 4.1891 | 21,516 tok/s
step  100 | loss 2.6260 | 17,121 tok/s
step  299 | loss 2.4133 | 24,631 tok/s

--- sample ---
LAMIS:
On by s sith me haro gasthe he busal this thend ary my thye I ke.
```

Early-training Shakespeare gibberish, produced by gradients that traveled
through hand-rolled reduce-scatter rings. The DDP parity tests are the
rigorous version of this demonstration; this is the fun one.

## Testing

```
pytest -q     # 18 tests, ~80 s (process spawn dominates)
```

- **Collectives:** every collective x every algorithm x sum/max/min/prod x
  float32/int64 x sizes chosen to hit edge cases (one element, fewer elements
  than ranks, non-divisible sizes, multi-buffer messages), at world sizes 2,
  3, and 4, with expected values recomputed independently on every rank from
  seeds. Plus explicit single-channel and all-channels-in-use cases.
- **DDP:** parity against single-process training (overlap on/off,
  multi-bucket, `no_sync()` accumulation), and initial-weight broadcast.
- **Transport:** full-duplex ring rotation deadlock test, zero-copy
  invariants.
- **Faults:** a rank dying mid-collective must surface in seconds, and a
  desync must produce a timeout plus a diagnosis naming the divergent
  collective.

## Running across machines

Every number here is loopback, which compresses the differences between
algorithms. `docs/multinode.md` covers running across hosts (each rank reads
`MINI_NCCL_HOSTS`, `RANK`, `WORLD_SIZE`; `scripts/launch_multinode.sh` does
the ssh fan-out), along with the specific predictions to check on a real
fabric: naive should collapse, ring's advantage should widen with world size,
more channels should start paying, and slice pipelining may flip from a loss
to a win.

## Limitations (deliberate)

- **CPU tensors, TCP transport.** The algorithms are transport-agnostic; the
  socket layer is the reference implementation. Device buffers would stage
  through pinned host memory.
- **A process group is not thread-safe per channel.** Collectives must be
  issued in identical order on every rank, so callers serialize per channel.
  This is NCCL's contract too, and the DDP reducer is built around it.
- **Halving-doubling needs a power-of-two world size** and falls back to ring
  otherwise.
- **No unused-parameter detection in DDP**, matching PyTorch's default.
- Equal tensor shapes are required on all ranks, as in NCCL.

## Roadmap

- Chunk pipelining *across* ring steps (independent chunks flowing through
  the ring simultaneously, rather than slicing within one step), which is how
  gloo takes the 64 MiB 4-rank case.
- The general non-power-of-two halving-doubling with the remainder fold-in.
- CUDA-aware path: device buffers staged through pinned host memory with a
  copy/communication pipeline, where slice pipelining should finally pay.
- FSDP-style parameter sharding on top of the existing reduce-scatter and
  all-gather.

## Layout

```
mini_nccl/
  transport.py       # sockets, mesh rendezvous, channels, timeouts
  process_group.py   # send / recv / full-duplex and sliced send_recv
  collectives.py     # ring, tree, halving-doubling, naive; auto selection
  ddp.py             # buckets, gradient views, overlap reducer, no_sync
  recorder.py        # flight recorder
  diagnose.py        # desync analysis and Perfetto trace merging
  launcher.py        # local multi-process runner
tests/               # collectives, DDP parity, transport, fault injection
benchmarks/          # sweep, tuning ablation, cost model fit, charts
examples/            # char-GPT, desync demo, multi-node benchmark
docs/multinode.md    # running on real hardware
```

## License

MIT
