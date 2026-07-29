"""A GPT split by depth across ranks, trained with a 1F1B pipeline.

    python examples/pipeline_gpt.py --world-size 4 --steps 60

Each rank owns a contiguous slice of the transformer blocks. Activations flow
forward through the stages and gradients flow back, with only one tensor
crossing each rank boundary in each direction. That makes pipeline parallelism
by far the cheapest model parallelism in bytes moved, and the fussiest to
schedule.

The run prints the measured pipeline depth per stage, which is the whole point
of 1F1B over GPipe: stage ``s`` holds at most ``W-s`` microbatches in flight
rather than all ``M``, so activation memory stops depending on how many
microbatches you picked. Pass ``--schedule gpipe`` to watch that bound
disappear.

The bubble (the fraction of time a stage sits idle) is ``(W-1)/(M+W-1)`` for
both schedules, so more microbatches means better utilization either way; 1F1B
just stops charging memory for it.
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

from model import Block

import mini_nccl as mn
from mini_nccl import collectives

DATA_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/"
    "data/tinyshakespeare/input.txt"
)
DATA_PATH = Path(__file__).parent.parent / "data" / "tinyshakespeare.txt"


class FirstStage(nn.Module):
    """Embeddings plus this rank's share of the blocks."""

    def __init__(self, vocab: int, width: int, block_size: int, n_head: int, depth: int):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab, width)
        self.pos_emb = nn.Parameter(torch.zeros(1, block_size, width))
        self.blocks = nn.Sequential(*[Block(width, n_head) for _ in range(depth)])
        nn.init.normal_(self.tok_emb.weight, std=0.02)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        x = self.tok_emb(idx) + self.pos_emb[:, : idx.shape[1]]
        return self.blocks(x)


class MiddleStage(nn.Module):
    def __init__(self, width: int, n_head: int, depth: int):
        super().__init__()
        self.blocks = nn.Sequential(*[Block(width, n_head) for _ in range(depth)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x)


class LastStage(nn.Module):
    """This rank's blocks, then the norm and the vocabulary projection."""

    def __init__(self, vocab: int, width: int, n_head: int, depth: int):
        super().__init__()
        self.blocks = nn.Sequential(*[Block(width, n_head) for _ in range(depth)])
        self.ln_f = nn.LayerNorm(width)
        self.head = nn.Linear(width, vocab, bias=False)
        nn.init.normal_(self.head.weight, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.ln_f(self.blocks(x)))


def build_stage(rank: int, world: int, vocab: int, cfg: dict) -> nn.Module:
    depth = cfg["n_layer"] // world
    if rank == 0:
        return FirstStage(vocab, cfg["n_embd"], cfg["block_size"], cfg["n_head"], depth)
    if rank == world - 1:
        return LastStage(vocab, cfg["n_embd"], cfg["n_head"], depth)
    return MiddleStage(cfg["n_embd"], cfg["n_head"], depth)


def load_corpus() -> str:
    if not DATA_PATH.exists():
        DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(DATA_URL, DATA_PATH)
    return DATA_PATH.read_text(encoding="utf-8")


def cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))


def worker(pg, data: torch.Tensor, vocab_size: int, cfg: dict) -> float:
    torch.manual_seed(1234 + pg.rank)  # each stage owns different layers
    stage = build_stage(pg.rank, pg.world_size, vocab_size, cfg)
    micro = cfg["batch_size"] // cfg["microbatches"]

    pipeline = mn.PipelineParallel(
        stage,
        pg,
        activation_shape=(micro, cfg["block_size"], cfg["n_embd"]),
        loss_fn=cross_entropy,
        schedule=cfg["schedule"],
    )
    opt = torch.optim.AdamW(stage.parameters(), lr=cfg["lr"], betas=(0.9, 0.95))

    local_params = sum(p.numel() for p in stage.parameters())
    if pg.rank == 0:
        print(
            f"{pg.world_size} stages, {cfg['n_layer'] // pg.world_size} blocks each | "
            f"schedule={cfg['schedule']} | {cfg['microbatches']} microbatches of {micro}\n"
            f"stage 0 holds {local_params / 1e6:.2f}M params"
        )

    block, batch = cfg["block_size"], cfg["batch_size"]
    final_loss = 0.0
    window_start, window_tokens = time.perf_counter(), 0

    for step in range(cfg["steps"]):
        # Every stage draws the same indices so the data lines up end to end.
        gen = torch.Generator().manual_seed(step)
        ix = torch.randint(len(data) - block - 1, (batch,), generator=gen)
        x = torch.stack([data[i : i + block] for i in ix])
        y = torch.stack([data[i + 1 : i + block + 1] for i in ix])

        opt.zero_grad(set_to_none=True)
        loss = pipeline.step(
            inputs=x if pg.rank == 0 else None,
            targets=y if pg.rank == pg.world_size - 1 else None,
            n_microbatches=cfg["microbatches"],
        )
        opt.step()

        window_tokens += batch * block
        if (step % 20 == 0 or step == cfg["steps"] - 1) and pg.rank == pg.world_size - 1:
            final_loss = loss
            elapsed = time.perf_counter() - window_start
            print(
                f"step {step:4d} | loss {loss:.4f} | "
                f"{window_tokens / elapsed:,.0f} tok/s"
            )
            window_start, window_tokens = time.perf_counter(), 0

    depths = collectives.all_gather(pg, torch.tensor([float(pipeline.in_flight_peak)]))
    if pg.rank == 0:
        measured = [int(d.item()) for d in depths]
        bound = [pg.world_size - s for s in range(pg.world_size)]
        print(f"\nmicrobatches in flight per stage: {measured}")
        if cfg["schedule"] == "1f1b":
            print(f"1F1B bound (W-s):                {bound}")
            print(
                f"GPipe would hold {cfg['microbatches']} on every stage, so peak "
                f"activation memory here is {cfg['microbatches'] / max(measured):.1f}x lower "
                f"on the busiest stage."
            )
        bubble = (pg.world_size - 1) / (cfg["microbatches"] + pg.world_size - 1)
        print(f"idle fraction (bubble): {bubble:.1%}")
    return final_loss


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--world-size", type=int, default=4)
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=16, help="global batch per step")
    ap.add_argument("--microbatches", type=int, default=8)
    ap.add_argument("--block-size", type=int, default=64)
    ap.add_argument("--n-layer", type=int, default=8)
    ap.add_argument("--n-head", type=int, default=4)
    ap.add_argument("--n-embd", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--schedule", choices=["1f1b", "gpipe"], default="1f1b")
    args = ap.parse_args()

    if args.n_layer % args.world_size:
        raise SystemExit(f"n_layer={args.n_layer} must divide by world_size={args.world_size}")
    if args.batch_size % args.microbatches:
        raise SystemExit("batch_size must divide by microbatches")

    text = load_corpus()
    vocab = "".join(sorted(set(text)))
    stoi = {ch: i for i, ch in enumerate(vocab)}
    data = torch.tensor([stoi[ch] for ch in text], dtype=torch.long)
    print(f"corpus: {len(text):,} chars, vocab {len(vocab)}")

    cfg = vars(args).copy()
    world = cfg.pop("world_size")
    losses = mn.run(worker, world, data, len(vocab), cfg, timeout=3600.0)
    print(f"final loss: {losses[-1]:.4f}")


if __name__ == "__main__":
    main()
