# Collectives on device tensors

A socket can only send host memory, so a tensor on an accelerator has to be
staged through the host and back:

```
device buffer -> host buffer -> socket -> host buffer -> device buffer
```

`mini_nccl/device.py` implements that three ways so they can be compared
instead of assumed.

| entry point | what it does |
|---|---|
| `_naive_all_reduce` (in the benchmark) | `tensor.cpu()`, reduce, copy back. Pageable host memory, so the driver stages through its own pinned buffer and the copies are synchronous. |
| `all_reduce_staged` | The same shape of work through an explicitly **pinned** host buffer, letting the copy engine DMA directly. Inherits every CPU-side feature: channels, algorithm selection, the narrow wire. |
| `all_reduce_pipelined` | A ring run on the device tensor. The payload is **chunked** so chunk *k* is on the wire while chunk *k+1* is still being copied off the device. The reduction stays on the device. |

Three details separate the fast version from the slow one:

1. **Pinned host memory.** A copy out of pageable memory cannot be done by
   DMA, so the driver stages it through an internal pinned buffer: an extra
   copy, and a synchronous one. `pin_memory=True` removes both.
2. **A separate copy stream.** Issued on their own stream, device copies do
   not serialize behind whatever the compute stream is doing, and the CPU is
   free while they are in flight.
3. **Overlapping copies with the network.** Double-buffered staging slots with
   CUDA events, so the copy engine and the CPU writing to the socket run at
   once rather than taking turns.

## Why this is the interesting case for the pipelining question

The chunked-overlap idea already appears in this repo for CPU tensors, and it
**lost** there: 5-40% slower on loopback TCP (see the ablation table in the
README), because "the network" on loopback is a memory copy performed by the
very cores that would be doing the copying. There is nothing to overlap when
both halves want the same hardware.

A device tensor is the case the optimization was always designed for. The copy
engine is separate silicon from the CPU running the socket write, so the two
can genuinely proceed in parallel. `benchmarks/bench_device.py` is set up to
answer whether that holds, and it prints the CPU null result too, which is
worth seeing side by side:

```
all-reduce of cpu tensors, world_size=2

| size | naive | staged (pinned) | pipelined | pipelined vs staged |
|---|---|---|---|---|
| 16 MiB | 67.3 ms | 58.2 ms | 127.3 ms | 0.46x |
| 64 MiB | 207.2 ms | 150.4 ms | 458.2 ms | 0.33x |
```

On CPU the pipelined path pays for two extra passes over the data and buys
nothing, exactly as predicted. The prediction to test on a GPU is that the last
column crosses 1.0.

## What is verified, and what is not

Being precise about this, because it is the difference between a measurement
and a hope:

- **Verified on CPU** (`tests/test_device.py`, runs in CI): the chunking covers
  the payload exactly with no gaps or overlaps, the double-buffered pipeline
  produces the same answer as the plain ring for chunk sizes from 64 bytes to
  1 MiB, payloads smaller than one chunk work, and non-contiguous input is
  rejected. All the control flow is shared between the CPU and CUDA paths, so
  this is the bulk of the logic.
- **Verified without a device**: the CUDA API surface the code calls (`Stream`,
  `Event`, `record`, `synchronize`, `wait_event`, `pin_memory=`,
  `non_blocking=`) exists in the installed torch. Shallow, but it catches the
  failure mode that actually bites untested code.
- **Not verified**: actual execution on a GPU. The machine this was developed
  on has an RTX 2070 with a 2020-era driver (451.67, CUDA 11.0 maximum), and
  every PyTorch CUDA build for Python 3.14 requires CUDA 12.6+, which needs
  driver 527 or newer. Two tests in `tests/test_device.py` cover the device
  path and currently skip. They are the check to run first once a GPU is
  available.

## Enabling it

**1. Update the NVIDIA driver** (needs administrator rights and a reboot).
Anything 527 or newer works; current drivers fully support Turing. Confirm with:

```
nvidia-smi
```

The `CUDA Version` in the header must read 12.6 or higher.

**2. Install a CUDA build of torch**, replacing the CPU-only one:

```
pip install --force-reinstall torch --index-url https://download.pytorch.org/whl/cu126
python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"
```

**3. Run the tests that were skipping**, then the benchmark:

```
pytest tests/test_device.py -v
python benchmarks/bench_device.py --world-size 2
```

## What a single GPU can and cannot tell you

Several ranks can share one GPU, and that is enough to prove correctness: the
device tests do exactly that. It says very little about performance, because
the ranks contend for one copy engine and one set of SMs, so the overlap the
pipelining depends on is partly serialized anyway.

A real answer to the performance question wants one GPU per rank. On a rented
multi-GPU node the useful comparison is against **NCCL itself**, through
`torch.distributed` with the `nccl` backend, which the existing benchmark
harness already knows how to run for gloo. That comparison is the honest
finish line for this part of the project: NCCL is heavily tuned C++ with
kernel-driven transfers and peer-to-peer paths, so the expectation is that it
wins comfortably. Knowing *by how much*, and which of its techniques account
for the gap, is the point.
