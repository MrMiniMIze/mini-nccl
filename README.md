# mini-nccl

[![ci](https://github.com/MrMiniMIze/mini-nccl/actions/workflows/ci.yml/badge.svg)](https://github.com/MrMiniMIze/mini-nccl/actions/workflows/ci.yml)

**Collective communication from first principles.** A small, readable
reimplementation of the algorithms inside libraries like NCCL (ring and
binomial-tree all-reduce, reduce-scatter, all-gather, broadcast), built on
nothing but TCP sockets and PyTorch tensors, plus a bucketed-overlap
`DistributedDataParallel` built on top of it, and a character-level GPT
whose every gradient byte moves through this library rather than
`torch.distributed`.

No `torch.distributed`, no MPI, no NCCL underneath. The point is to make the
machinery of distributed training small enough to read in an afternoon and
measured enough to trust.

![all-reduce bus bandwidth](docs/img/allreduce_busbw.png)

## What's inside

| Layer | File | What it does |
|---|---|---|
| Transport | `mini_nccl/transport.py` | Full-mesh TCP rendezvous, zero-copy receives (`socket.recv_into` straight into tensor storage) |
| Process group | `mini_nccl/process_group.py` | `send` / `recv` / full-duplex `send_recv`: the three primitives everything else is built from |
| Collectives | `mini_nccl/collectives.py` | `all_reduce` (ring / binomial tree / naive), `reduce_scatter`, `all_gather`, `broadcast`, `barrier` |
| DDP | `mini_nccl/ddp.py` | Gradient bucketing, grads-as-bucket-views, communication/compute overlap on a dedicated reducer thread |
| Launcher | `mini_nccl/launcher.py` | `mn.run(fn, world_size)`: spawn, rendezvous, collect results, propagate worker tracebacks |
| Example | `examples/train_gpt.py` | Char-level GPT trained data-parallel on tiny shakespeare |
| Benchmarks | `benchmarks/` | nccl-tests-style sweep vs `torch.distributed` gloo, overlap timing, chart generation |

## Quickstart

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e .[dev]

pytest -q                                        # correctness: 12 tests, world sizes 2-4

python examples/train_gpt.py --world-size 4 --steps 200 --sample
python benchmarks/bench_allreduce.py --world-sizes 2,4 --gloo
python benchmarks/plot_results.py
```

Using the library directly looks like `torch.distributed`:

```python
import mini_nccl as mn

def worker(pg, _):
    t = torch.ones(1024) * pg.rank
    mn.collectives.all_reduce(pg, t)          # in place, algorithm="auto"
    mn.collectives.broadcast(pg, t, src=0)

mn.run(worker, world_size=4, None)
```

## How it works

### Ring all-reduce

The bandwidth-optimal algorithm, and the reason data-parallel training
scales. The tensor is split into `W` blocks. In `W-1` **reduce-scatter**
steps, each rank sends the block it just accumulated to its right neighbor
while receiving (and reducing into) the next block from its left neighbor.
After that phase, every rank holds one *fully reduced* block. `W-1`
**all-gather** steps then circulate the finished blocks around the ring.

Every rank sends `2 (W-1)/W · n` bytes total, asymptotically independent
of world size. That is the whole magic: add more workers and per-rank
traffic stays flat. The cost is latency: `2(W-1)` serialized steps.

Both directions of every step run concurrently: `send_recv` pushes to the
right neighbor from a send thread while the calling thread blocks on the
receive from the left (CPython releases the GIL inside socket syscalls, so
this is real full-duplex, not time-slicing).

### Binomial tree all-reduce

Latency-optimal: reduce up a binomial tree to rank 0 in `⌈log₂ W⌉` steps,
broadcast back down in `⌈log₂ W⌉` more. Interior ranks forward the full
payload, so total bytes moved grow with `log W`: worse than ring for large
tensors, unbeatable for small ones where per-message latency dominates.

### Algorithm selection

`algorithm="auto"` picks tree at or below `RING_THRESHOLD_BYTES` (256 KiB)
and ring above it, the same latency-vs-bandwidth decision NCCL's tuner
makes. The measured crossover on loopback TCP sits right in that window:

![all-reduce latency](docs/img/allreduce_latency.png)

The threshold is a constant in `mini_nccl/collectives.py`; re-derive it for
your own fabric from the benchmark CSV.

### The naive baseline

`algorithm="naive"` is the parameter-server pattern: every rank sends its
tensor to rank 0, which reduces and sends results back. Rank 0 moves
`O(W · n)` bytes. It's included because the benchmark charts make the point
better than prose: it looks fine at small scale and falls behind as world
size and message size grow, which is exactly why ring algorithms exist.

### DDP: bucketing, gradient views, overlap

`mini_nccl.DistributedDataParallel` reimplements the architecture of
PyTorch's DDP:

- **Buckets.** Parameters are grouped in reverse registration order
  (approximately the order gradients are produced during backward) into
  ~`bucket_cap_mb` flat buffers. One all-reduce per bucket instead of one
  per tensor amortizes latency.
- **Gradient views.** Each `param.grad` is a view into its bucket's flat
  buffer, so autograd accumulates directly into the communication buffer,
  with zero flatten/unflatten copies.
- **Overlap.** A post-accumulate-grad hook marks buckets ready; a dedicated
  reducer thread all-reduces each bucket as it completes, while backward is
  still computing earlier layers' gradients.
- **Determinism invariant.** The reducer processes buckets in fixed index
  order regardless of readiness order, so every rank issues the identical
  collective sequence, the same invariant NCCL communicators require to
  avoid cross-rank deadlock.

Correctness is enforced by the strictest test a DDP can face: training on
`W` processes (each with `1/W` of the batch) must produce the same
parameters as single-process full-batch training, step for step, to
floating-point tolerance, including with 1 KiB buckets and overlap enabled
(`tests/test_ddp.py`).

### What overlap buys: an honest number

On this benchmark's CPU-loopback setup, overlap hides only ~2% of step time
(`benchmarks/bench_ddp_overlap.py`). That is the *correct* result for the
environment, and worth understanding: overlap pays when communication uses
a resource distinct from compute: a NIC with DMA, a GPU copy engine, a
dedicated fabric. On loopback, "network transfer" is memcpy executed by the
same CPU cores that backward needs, so there is nothing independent to
overlap with. The mechanism is what matters; the payoff appears the moment
the transport stops sharing silicon with the model.

## The proof: a GPT trained entirely through mini-nccl

`examples/train_gpt.py` trains a ~1M-parameter character-level GPT on tiny
shakespeare, data-parallel, with gradients averaged by this library's own
ring all-reduce; `torch.distributed` is never imported. Two ranks on a
laptop CPU:

```
step    0 | loss 4.1891 | 21,516 tok/s
step  100 | loss 2.6260 | 17,121 tok/s
step  299 | loss 2.4133 | 24,631 tok/s

--- sample ---
LAMIS:
On by s sith me haro gasthe he busal this thend ary my thye I ke.
```

Early-training Shakespeare gibberish, produced by gradients that traveled
through hand-rolled reduce-scatter rings. The DDP parity tests below are
the rigorous version of this demonstration.

## Benchmarks

`bench_allreduce.py` follows nccl-tests conventions: per-config time is the
max across ranks (the slowest rank defines collective latency), and bus
bandwidth for all-reduce is `algbw · 2(W-1)/W`. Run on loopback TCP,
4 ranks, CPU tensors:

| size | ring | tree | naive | gloo |
|---|---|---|---|---|
| 4 KiB | 1.65 ms | **0.42 ms** | 0.35 ms | 3.12 ms |
| 1 MiB | **2.4 ms** | 3.5 ms | 3.8 ms | 4.0 ms |
| 16 MiB | **64 ms** | 91 ms | 77 ms | 64 ms |
| 64 MiB | 253 ms | 334 ms | 310 ms | **218 ms** |

mini-nccl's ring beats `torch.distributed`'s gloo backend across the small
and mid range and ties it at 16 MiB; gloo takes the largest size (its
chunk pipelining starts paying; see roadmap). Loopback numbers compress
real-network differences: on a physical fabric, naive's central bottleneck
and ring's bandwidth optimality both separate much harder.

## Testing

```
pytest -q     # 12 tests, ~50 s (process spawn dominates)
```

- Every collective × every algorithm × sum/max/min/prod × float32/int64 ×
  sizes chosen to hit edge cases (1 element, fewer elements than ranks,
  non-divisible sizes, multi-buffer messages), at world sizes 2, 3, and 4,
  with expected values recomputed independently on every rank from seeds.
- DDP parity vs single-process training (overlap on/off, multi-bucket).
- Transport: full-duplex ring rotation deadlock test, zero-copy invariants,
  worker exception propagation.

## Limitations (deliberate)

- **CPU tensors, TCP transport.** The algorithms are transport-agnostic;
  the socket layer is the reference implementation. Device buffers would
  stage through pinned host memory (see roadmap).
- **A process group is not thread-safe.** Collectives must be issued in
  identical order on every rank; callers serialize. This is NCCL's contract
  too, and the DDP reducer thread is built around it.
- **No unused-parameter detection, no gradient accumulation across
  backwards**, the same defaults as `torch.nn.parallel.DistributedDataParallel`,
  kept out to keep the reducer readable.
- Equal tensor shapes are required on all ranks (NCCL's contract as well).

## Roadmap

- Chunk pipelining within ring steps (split blocks into slices so a rank
  starts forwarding a slice while the rest is still arriving); closes the
  64 MiB gap to gloo.
- Recursive halving-doubling all-reduce, for a better latency×bandwidth product
  in the mid range.
- Multi-node support: hostfile-based rendezvous (the address book is
  already a `(host, port)` list; only launch tooling is missing).
- CUDA-aware path: device buffers staged through pinned host memory with a
  copy/communication pipeline.

## Layout

```
mini_nccl/
  transport.py       # sockets, mesh rendezvous, zero-copy framing
  process_group.py   # send / recv / full-duplex send_recv
  collectives.py     # ring, binomial tree, naive; auto selection
  ddp.py             # buckets, gradient views, overlap reducer
  launcher.py        # local multi-process runner
tests/               # correctness: collectives battery + DDP parity
benchmarks/          # sweep, overlap timing, charts
examples/            # char-GPT trained through mini-nccl
```

## License

MIT
