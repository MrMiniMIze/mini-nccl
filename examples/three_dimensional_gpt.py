"""A GPT trained across all three parallelism dimensions at once.

    python examples/three_dimensional_gpt.py --world-size 8 --pp 2 --tp 2

Eight ranks as a 2x2x2 mesh. Each rank holds **one stage** of the model, **one
tensor shard** of that stage's layers, and belongs to **one of two replicas**.
This is the shape large-model training actually uses, and each dimension is
chosen for a different reason:

- **tensor** splits a layer, so a matrix multiply too large for one device fits.
  It costs two all-reduces per block, so it wants the fastest links.
- **pipeline** splits the stack, moving only one activation per boundary. It is
  the cheapest in bytes and the fussiest in scheduling, and it buys depth.
- **data** replicates the whole arrangement to raise throughput.

For the mechanics of each, read ``tensor_parallel_gpt.py`` and
``pipeline_gpt.py``. What this example adds is the composition, and two rules
about how the dimensions interact:

1. **Gradients are averaged along the data dimension only.** Tensor-parallel
   ranks own different slices and pipeline stages own different layers, so
   neither axis has anything to average.
2. **That averaging happens after the pipeline drains, not in a backward hook.**
   The pipeline runs backward once per microbatch, so a DDP-style hook would
   fire ``M`` times per step and reduce partial gradients.
   ``average_gradients`` is the explicit form for exactly this case.
"""

from __future__ import annotations

import argparse
import sys
import time
import urllib.request
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

sys.path.insert(0, str(Path(__file__).parent))

import mini_nccl as mn
from mini_nccl import collectives
from mini_nccl.ddp import average_gradients
from mini_nccl.mesh import ParallelMesh
from mini_nccl.tensor_parallel import ParallelMLP, ParallelSelfAttention

DATA_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/"
    "data/tinyshakespeare/input.txt"
)
DATA_PATH = Path(__file__).parent.parent / "data" / "tinyshakespeare.txt"


class Block(nn.Module):
    """A transformer block whose attention and MLP are tensor-parallel."""

    def __init__(self, width: int, n_head: int, tp_group) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(width)
        self.attn = ParallelSelfAttention(width, n_head, tp_group)
        self.ln2 = nn.LayerNorm(width)
        self.mlp = ParallelMLP(width, tp_group)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        return x + self.mlp(self.ln2(x))


class FirstStage(nn.Module):
    def __init__(self, vocab: int, cfg: dict, tp_group, depth: int) -> None:
        super().__init__()
        width = cfg["n_embd"]
        self.tok_emb = nn.Embedding(vocab, width)
        self.pos_emb = nn.Parameter(torch.zeros(1, cfg["block_size"], width))
        self.blocks = nn.Sequential(
            *[Block(width, cfg["n_head"], tp_group) for _ in range(depth)]
        )
        nn.init.normal_(self.tok_emb.weight, std=0.02)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        x = self.tok_emb(idx) + self.pos_emb[:, : idx.shape[1]]
        return self.blocks(x)


class MiddleStage(nn.Module):
    def __init__(self, cfg: dict, tp_group, depth: int) -> None:
        super().__init__()
        self.blocks = nn.Sequential(
            *[Block(cfg["n_embd"], cfg["n_head"], tp_group) for _ in range(depth)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x)


class LastStage(nn.Module):
    def __init__(self, vocab: int, cfg: dict, tp_group, depth: int) -> None:
        super().__init__()
        width = cfg["n_embd"]
        self.blocks = nn.Sequential(
            *[Block(width, cfg["n_head"], tp_group) for _ in range(depth)]
        )
        self.ln_f = nn.LayerNorm(width)
        self.head = nn.Linear(width, vocab, bias=False)
        nn.init.normal_(self.head.weight, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.ln_f(self.blocks(x)))


def build_stage(mesh: ParallelMesh, vocab: int, cfg: dict, tp_group) -> nn.Module:
    stage_index = mesh.coordinate("pp")
    depth = cfg["n_layer"] // mesh.size("pp")
    if stage_index == 0:
        return FirstStage(vocab, cfg, tp_group, depth)
    if stage_index == mesh.size("pp") - 1:
        return LastStage(vocab, cfg, tp_group, depth)
    return MiddleStage(cfg, tp_group, depth)


def load_corpus() -> str:
    if not DATA_PATH.exists():
        DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(DATA_URL, DATA_PATH)
    return DATA_PATH.read_text(encoding="utf-8")


def cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))


def worker(pg, data: torch.Tensor, vocab_size: int, cfg: dict) -> float:
    mesh = ParallelMesh(pg, dp=cfg["dp"], pp=cfg["pp"], tp=cfg["tp"])
    tp, pp, dp = mesh.group("tp"), mesh.group("pp"), mesh.group("dp")

    # Same seed everywhere: replicas of the same stage and shard must start
    # identical, and there is no data-parallel broadcast here to fix it up.
    torch.manual_seed(7)
    stage = build_stage(mesh, vocab_size, cfg, tp)

    micro = cfg["batch_size"] // cfg["microbatches"]
    pipeline = mn.PipelineParallel(
        stage,
        pg=pp,
        activation_shape=(micro, cfg["block_size"], cfg["n_embd"]),
        loss_fn=cross_entropy,
        schedule="1f1b",
    )
    opt = torch.optim.AdamW(stage.parameters(), lr=cfg["lr"], betas=(0.9, 0.95))

    local = sum(p.numel() for p in stage.parameters())
    totals = collectives.all_gather(pg, torch.tensor([float(local)]))
    if pg.rank == 0:
        whole = sum(t.item() for t in totals) / mesh.size("dp")
        print(
            f"mesh dp={mesh.size('dp')} x pp={mesh.size('pp')} x tp={mesh.size('tp')}"
            f"  ({pg.world_size} ranks)\n"
            f"  {mesh.describe()}\n"
            f"  tensor group {mesh.ranks_along('tp')} | "
            f"pipeline group {mesh.ranks_along('pp')} | "
            f"data group {mesh.ranks_along('dp')}\n"
            f"  per-rank params {local / 1e6:.2f}M of {whole / 1e6:.2f}M total "
            f"({whole / local:.1f}x smaller per rank)"
        )

    block, batch = cfg["block_size"], cfg["batch_size"]
    is_last = mesh.coordinate("pp") == mesh.size("pp") - 1
    final_loss = 0.0
    window_start, window_tokens = time.perf_counter(), 0

    for step in range(cfg["steps"]):
        # Replicas take different data; every rank inside a replica shares it.
        gen = torch.Generator().manual_seed(step * mesh.size("dp") + mesh.coordinate("dp"))
        ix = torch.randint(len(data) - block - 1, (batch,), generator=gen)
        x = torch.stack([data[i : i + block] for i in ix])
        y = torch.stack([data[i + 1 : i + block + 1] for i in ix])

        opt.zero_grad(set_to_none=True)
        loss = pipeline.step(
            inputs=x if mesh.coordinate("pp") == 0 else None,
            targets=y if is_last else None,
            n_microbatches=cfg["microbatches"],
        )
        average_gradients(dp, list(stage.parameters()))
        opt.step()

        window_tokens += batch * block * mesh.size("dp")
        if (step % 20 == 0 or step == cfg["steps"] - 1) and is_last:
            final_loss = loss
            if mesh.coordinate("dp") == 0 and mesh.coordinate("tp") == 0:
                elapsed = time.perf_counter() - window_start
                print(
                    f"step {step:4d} | loss {loss:.4f} | "
                    f"{window_tokens / elapsed:,.0f} tok/s"
                )
                window_start, window_tokens = time.perf_counter(), 0

    if is_last:
        # Ranks differing only in tensor shard ran identical data through a
        # split model, so their losses must agree exactly.
        same_tp = collectives.all_gather(tp, torch.tensor([final_loss]))
        spread = max(t.item() for t in same_tp) - min(t.item() for t in same_tp)
        if mesh.coordinate("dp") == 0 and mesh.coordinate("tp") == 0:
            print(f"\nloss spread across the tensor group: {spread:.2e}")
            print(f"in-flight microbatches on this stage: {pipeline.in_flight_peak}")
    return final_loss


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--world-size", type=int, default=8)
    ap.add_argument("--pp", type=int, default=2, help="pipeline stages")
    ap.add_argument("--tp", type=int, default=2, help="tensor-parallel width")
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=8, help="per replica")
    ap.add_argument("--microbatches", type=int, default=4)
    ap.add_argument("--block-size", type=int, default=64)
    ap.add_argument("--n-layer", type=int, default=4)
    ap.add_argument("--n-head", type=int, default=4)
    ap.add_argument("--n-embd", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    args = ap.parse_args()

    if args.world_size % (args.pp * args.tp):
        raise SystemExit(
            f"world_size={args.world_size} must divide by pp*tp={args.pp * args.tp}"
        )
    if args.n_layer % args.pp:
        raise SystemExit(f"n_layer={args.n_layer} must divide by pp={args.pp}")
    if args.n_head % args.tp:
        raise SystemExit(f"n_head={args.n_head} must divide by tp={args.tp}")
    if args.batch_size % args.microbatches:
        raise SystemExit("batch_size must divide by microbatches")

    text = load_corpus()
    vocab = "".join(sorted(set(text)))
    stoi = {ch: i for i, ch in enumerate(vocab)}
    data = torch.tensor([stoi[ch] for ch in text], dtype=torch.long)
    print(f"corpus: {len(text):,} chars, vocab {len(vocab)}")

    cfg = vars(args).copy()
    world = cfg.pop("world_size")
    cfg["dp"] = world // (args.pp * args.tp)
    losses = mn.run(worker, world, data, len(vocab), cfg, timeout=3600.0)
    print(f"final loss: {max(losses):.4f}")


if __name__ == "__main__":
    main()
