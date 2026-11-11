# Emulated links: testing what loopback hides

The README's section [What loopback teaches, and what it
hides](../README.md#what-loopback-teaches-and-what-it-hides) explains four
negative results with one cause: on loopback, "the network" is a memory copy
performed by the same cores that do the reduction, so bytes are never the
expensive part. [docs/multinode.md](multinode.md) then lists what *should*
happen on a real link. None of that had been tested, because it needs a
network.

`mini_nccl.LinkModel` emulates one. This document writes down, with numbers,
what the emulated link should show, **before** anything was measured on it.
The measurements arrive in the next commit, so the history shows that these
predictions were not fitted to the results. The pass/fail tolerance is fixed
here too, for the same reason.

## The emulator

A shaped rank sleeps instead of copying, which gives the two properties
loopback lacks: bytes cost time, and the CPU is idle while they move.

- **Rate** (`mbps`): per rank, per direction, shared by every connection and
  channel the rank has. One full-duplex NIC per host.
- **Per-flow rate** (`flow_mbps`): an additional cap per connection, the only
  setting under which "do more channels help?" is a fair question.
- **Latency** (`latency_us`): a fixed delay per send. Not used below; the
  software's own per-step cost (the alpha in the model) is already 60 to
  170 µs, more than a LAN's propagation delay.

Each direction of a link is a clock that books every chunk's serialization
time. An idle link banks no credit, so a small message pays its own delay.
Arrivals are booked as having just finished, so one flow is not charged twice,
but several senders converging on one rank queue on its ingress.

**Calibration** (point-to-point, 2 ranks, best of 5): the emulator delivers
99.5% of nominal at 100 Mb/s, 99.6% at 1 Gb/s and 99.7% at 10 Gb/s. A 1 KiB
message plus a 4-byte acknowledgement takes 173 µs at 100 Mb/s against 82 µs of
serialization, so shaping adds roughly **45 µs per message** of timer and
scheduling overhead. That figure enters the model below as `δ`.

## What this can show, and what it cannot

It can test every prediction that depends on bytes being expensive while the
CPU is free, on links well below what loopback itself can carry.

**Loopback on this machine is not the README's loopback.** The README's
numbers come from a machine whose loopback moved about 0.5 GB/s. Here (Apple
M5, 10 cores, macOS 27.2, Python 3.12, torch 2.15) a 2-rank point-to-point
send moves **5.7 GB/s at 16 MiB and 14.8 GB/s at 64 MiB**, ten to thirty times
more. So the README's premise that loopback is about 22x slower than PCIe is a
property of that machine, not of loopback in general. Every experiment below
therefore also runs unshaped on this machine as a control, and none of the
predictions uses the README's loopback numbers.

It **cannot** test the README's strongest prediction, that pipelining the GPU
staging copies pays once the fabric is within reach of PCIe (about 100 Gb/s).
This machine has no CUDA device, and 100 Gb/s is the speed of loopback itself
here, so there is nothing to emulate it with. That prediction stays on the
roadmap.

Also not modeled: switch contention beyond each host's own link, packet loss
and retransmission, TCP congestion control, latency that overlaps with
transmission, and several ranks sharing one NIC (each rank here has its own).

## The model

Every prediction comes from one formula, with inputs measured unshaped on this
machine:

```
t = steps · (α + δ)  +  bytes_on_critical_path · (β + 1/R)
```

- `α`, `β`: the alpha-beta fit of the unshaped all-reduce sweep
  (`fit_cost_model.py` on `bench_allreduce.py --world-sizes 2,4`):

  | world | algorithm | α per step | β per byte |
  |---|---|---|---|
  | 2 | ring | 166 µs | 0.26 ns |
  | 2 | naive | 50 µs | 0.16 ns |
  | 4 | ring | 148 µs | 0.35 ns |
  | 4 | tree | 66 µs | 0.17 ns |
  | 4 | naive | 26 µs | 0.14 ns |

  There is no 8-rank fit, so 8 ranks reuse the 4-rank values. At 1 Gb/s the
  link term is about 25 times β, so this barely matters.
- `δ` = 45 µs per send, from the calibration above.
- `1/R`: 80 ns per byte at 100 Mb/s, 8 ns per byte at 1 Gb/s.
- Bytes on the critical path, with every rank's link capped in both
  directions: ring `2(W-1)/W · n`; tree `2⌈log₂W⌉ · n`; naive `2(W-1) · n`,
  because rank 0 takes in `W-1` tensors and sends out `W-1` copies of the
  result through one link.

Unless stated otherwise: 4 ranks, ring all-reduce, 16 MiB of float32.

## The predictions

### 1. Naive collapses, by a factor of the world size

Ring moves `2(W-1)/W · n` per rank, all ranks at once; naive moves
`2(W-1) · n` through rank 0. Their ratio is `W`.

| link | ranks | ring | naive | naive / ring |
|---|---|---|---|---|
| 1 Gb/s | 2 | 139 ms | 274 ms | **1.97x** |
| 1 Gb/s | 4 | 211 ms | 820 ms | **3.88x** |
| 1 Gb/s | 8 | 248 ms | 1913 ms | **7.72x** |
| 100 Mb/s | 4 | 2023 ms | 8068 ms | **3.99x** |

On this machine's loopback the same ratio was 1.28x at 2 ranks and 1.10x at 4
(the unshaped sweep above), which is why the README could only say naive
"looks good on loopback".

### 2. The tree/ring crossover moves down by two to three orders of magnitude

Tree wins while its smaller step count matters more than its larger byte
count. Making bytes expensive moves the crossover from megabytes (the model
put it at 3.9 MiB on loopback here, measured 1 MiB) to:

| link | model crossover | first measured size where ring should win |
|---|---|---|
| 100 Mb/s | **3.5 KiB** | **4 KiB**, narrowly: the model gives ring only a 6% edge there |
| 1 Gb/s | **35 KiB** | **64 KiB** (16 KiB: tree ahead, tree/ring 0.72) |

The consequence is the interesting part. `algorithm="auto"` picks tree at or
below 1 MiB, a threshold tuned on loopback. On either link that sends every
message between the crossover and 1 MiB down the slower algorithm. At 1 MiB
the model puts tree at **2.64x** ring's time at 100 Mb/s and **2.43x** at
1 Gb/s.

### 3. A bfloat16 wire finally halves the time

Halving the wire bytes halves the link term. Assuming the per-hop casts cost
about what the saved copying did (which is what the near-1.0x loopback result
implies), the speedup at 16 MiB is:

| link | float32 | bfloat16 wire | speedup |
|---|---|---|---|
| 100 Mb/s | 2023 ms | 1017 ms | **1.99x** |
| 1 Gb/s | 211 ms | 111 ms | **1.91x** |

The README measured 0.95x to 1.28x on its loopback.

### 4. DDP overlap pays most when communication and backward are balanced

Overlap can hide at most the shorter of backward and communication, minus the
last bucket, which only becomes ready when backward ends. Inputs, measured
unshaped on 4 ranks with overlap off (the README's GPT: 4.8M parameters,
0.5 MiB buckets, which gives 37 buckets and 19.15 MB of gradients): forward
55 ms, backward 100 ms, optimizer 16 ms.

Communication per step is `37 · 6 · (α + δ) + 1.5 · 19.15 MB · (β + 1/R)`. The
two are balanced at 4.7 Gb/s, which was computed from those inputs and is
included as a third rate for that reason:

| link | communication | step, no overlap | step, overlap | gain |
|---|---|---|---|---|
| 100 Mb/s | 2351 ms | 2522 ms | 2422 ms | **4.0%** |
| 1 Gb/s | 283 ms | 454 ms | 354 ms | **22%** |
| 4.7 Gb/s | 102 ms | 273 ms | 173 ms | **37%** |

The gain should rise and then fall again. At 100 Mb/s communication dwarfs
everything, so hiding all of backward behind it saves little. This is the same
balanced-stages argument that bounds the device pipelining in the README,
applied to a different pair of stages. The README measured +2% on its
loopback; the unshaped run here is a control, with no prediction made.

### 5. Channels help only when the limit is per flow

| cap at 1 Gb/s | 2 channels | 4 channels |
|---|---|---|
| per rank (one NIC) | **1.00x** | **1.00x** |
| per flow | **1.91x to 1.99x** | **3.50x to 3.94x** |

Under a per-rank cap, extra connections divide the same link, so nothing
changes. Under a per-flow cap each channel brings its own bandwidth. The range
depends on whether the CPU work (`β`) also splits across channel threads
(upper end) or stays serial (lower end). docs/multinode.md's advice that a NIC
"should prefer 4-8" channels is really a statement about per-flow limits.

### 6. Slice pipelining stays useless, for the opposite reason

Pipelining hides the reduction behind the transfer. One core reduces float32
at about 30 GB/s here, so the reduction on ring's critical path is 0.42 ms out
of 211 ms at 1 Gb/s. That makes the ceiling **1.002x at 1 Gb/s and 1.0002x at
100 Mb/s**. On loopback the mechanism lost because the reduction and the
transfer fought over the same cores. On a shaped link the transfer uses no
cores, so it should stop losing, but there is nothing left to win either.

Prediction: **1.00x** at both rates, no gain and no loss. This disagrees with
docs/multinode.md, which says the mechanism "may flip from a loss to a win" on
a NIC. On a slow link it cannot, because the stage it hides is too small to
matter. It could only win on a link fast enough for the reduction to be a
large share of the step time, which is closer to the GPU case this machine
cannot test.

## How the verdicts will be judged

Fixed now, before measuring:

- A ratio or speedup **holds** if the measurement is within **±15%** of the
  predicted value, and **misses** otherwise. For predictions 5 (per-rank cap)
  and 6, which predict 1.00x, it holds if the measurement is within
  **0.95x to 1.05x**.
- The crossover holds if ring first beats tree at the predicted grid size. One
  grid step either way counts as a near miss, and is reported as such. The
  100 Mb/s case is flagged above as marginal.
- For the DDP gain, the prediction that matters is the shape: lowest at
  100 Mb/s, highest at the balanced rate. Each gain is also judged against its
  number at ±15% relative.

Every miss gets an explanation of how big it was and why, rather than a
re-tuned model. The experiments are `benchmarks/bench_emulated_link.py`
(added with the measurements); the unshaped inputs come from its
`--unshaped-inputs` mode and from `bench_allreduce.py` plus
`fit_cost_model.py`.
