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

## Results on an RTX 2070

### Pinned staging is a real win

`bench_copy_ceiling.py` measures the two halves separately:

| size | D2H pinned | H2D pinned | D2H pageable |
|---|---|---|---|
| 4 MiB | 10.9 GB/s | 10.5 GB/s | 4.9 GB/s |
| 16 MiB | 11.2 GB/s | 11.3 GB/s | 6.5 GB/s |
| 64 MiB | 10.5 GB/s | 11.3 GB/s | 6.5 GB/s |

Pinned host memory is worth about **1.7x on the copy**, which is exactly what
the mechanism is for: pageable memory cannot be DMA'd, so the driver stages it
through its own pinned buffer and pays for an extra copy along the way.

### Pipelining those copies cannot pay here, and the bound says why

PCIe moves 11 GB/s. The loopback transport moves about 0.5 GB/s. So the copy is
a small fraction of the total, and since pipelining can only hide the copy,
that fraction is the entire prize:

| size | copy round trip | network time | copy share | ceiling on pipelining |
|---|---|---|---|---|
| 4 MiB | 0.7 to 0.9 ms | 8 ms | 8.6 to 10.2% | **1.09 to 1.11x** |
| 16 MiB | 2.8 to 3.4 ms | 31 ms | 8.2 to 9.9% | **1.09 to 1.11x** |
| 64 MiB | 11.4 to 13.2 ms | 125 ms | 8.4 to 9.5% | **1.09 to 1.11x** |

(Two runs, shown as a range, because the run-to-run variance on a display GPU
is larger than the difference being discussed.)

Measured end to end, the pipelined path lands between 0.44x and 1.35x against
pinned staging, below its own ~1.1x ceiling once per-chunk overhead and
run-to-run variance are accounted for. This machine is a laptop GPU driving a
display under WDDM with both ranks sharing it, so the variance is large; but the
ceiling calculation does not depend on any of that. **No chunk size fixes a 10%
ceiling**, which makes this a structural result rather than a tuning failure.

The useful part is the condition it implies. Pipelining rewards *balanced*
stages: at 100 Gb/s (12.5 GB/s) a fabric is within 15% of PCIe, the copy share
approaches half, and the ceiling approaches 2x. That is why NCCL pipelines
aggressively on the hardware it targets, and why the same idea measured
*negative* on loopback with CPU tensors (0.59x to 0.95x, see the README's
ablation table). The technique is not wrong; the ratio it needs is absent here.

## The bug a GPU found immediately

Worth recording, because it is the argument for running the device tests rather
than trusting the CPU ones.

The pipeline logic is shared between the CPU and CUDA paths, and all of it
passed on CPU, where CUDA streams are no-ops. The first execution on a real GPU
failed within seconds: a ring step's reduction is queued on the **compute**
stream, while the next step copies that same block off the device on the
**copy** stream, and nothing ordered the two. The copy was free to read the
block before the addition landed, so the peer received a partially reduced
value. Rank 0 saw 2.0 where it expected 3.0, precisely the un-reduced number.

The fix is one `wait_stream` per exchange (`Staging.order_copies_after_compute`).
The class of bug (a missing cross-stream dependency) is invisible to any test
that runs without a device.

## What is verified

- **On CPU, in CI** (`tests/test_device.py`): chunking covers the payload with
  no gaps or overlaps, the double-buffered pipeline agrees with the plain ring
  for chunk sizes from 64 bytes to 1 MiB, sub-chunk payloads work,
  non-contiguous input is rejected, and every CUDA API the code calls exists in
  the installed torch.
- **On a GPU** (`nvidia-smi` driver 610.88, CUDA 12.6 build of torch, RTX 2070
  sm_75): all 12 device tests pass with nothing skipped. Both device paths produce
  the correct result for payloads from 1 element to 250k, and the staging really
  is pinned with a separate stream and two buffer slots.
- **Not verified**: anything about multi-GPU performance, and any comparison
  against NCCL. The `nccl` backend in `torch.distributed` is Linux-only, so that
  comparison needs a rented Linux box, one GPU per rank.

## Enabling it elsewhere

**1. Update the NVIDIA driver** (needs administrator rights, and usually a
reboot). Anything 527 or newer works; current drivers fully support Turing.
Confirm with:

```
nvidia-smi
```

The `CUDA Version` in the header must read 12.6 or higher.

**2. Install a CUDA build of torch**, replacing the CPU-only one:

```
pip install --force-reinstall torch --index-url https://download.pytorch.org/whl/cu126
python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"
```

**3. Run the tests that were skipping**, then the benchmarks:

```
pytest tests/test_device.py -v
python benchmarks/bench_copy_ceiling.py --transport-gbps 0.5
python benchmarks/bench_device.py --world-size 2 --channels 1
```

Run `bench_copy_ceiling.py` before drawing conclusions from
`bench_device.py`. It tells you what the ceiling on pipelining is for *your*
copy and transport bandwidths, and therefore whether a disappointing result is
worth chasing. Pass your own transport number: the default 0.5 GB/s is what the
loopback ring in this repo measures, and a real fabric will be different.

`bench_device.py` defaults to one channel so the pipelined path (single socket)
and the staged path are compared on equal footing; the staged path would
otherwise also pick up the CPU collective's channel parallelism, which has
nothing to do with pipelining.

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
