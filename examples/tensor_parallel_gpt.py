"""A GPT whose attention and MLP weights are split across ranks.

    python examples/tensor_parallel_gpt.py --world-size 4 --steps 100

This is not data parallelism. Every rank sees the *same* batch and holds a
*different slice of every block*, so a model wider than one device can hold
still trains. Attention splits by head, the MLP splits by hidden unit, and
each block costs exactly two all-reduces (one forward, one backward).

Embeddings, norms, and the tied head stay replicated. They need no gradient
synchronization at all, which is worth understanding: the backward pass
all-reduces the activation gradient at each block boundary, so every rank
sees an identical gradient arriving at the replicated layers. Ring and tree
all-reduce both produce bitwise-identical results on every rank (each output
element is reduced on one rank, then copied), so the replicated weights stay
exactly in step instead of slowly drifting apart.
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
from mini_nccl.tensor_parallel import ParallelMLP, ParallelSelfAttention

DATA_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/"
    "data/tinyshakespeare/input.txt"
)
DATA_PATH = Path(__file__).parent.parent / "data" / "tinyshakespeare.txt"


class ParallelBlock(nn.Module):
    def __init__(self, width: int, n_head: int, pg) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(width)
        self.attn = ParallelSelfAttention(width, n_head, pg)
        self.ln2 = nn.LayerNorm(width)
        self.mlp = ParallelMLP(width, pg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        return x + self.mlp(self.ln2(x))


class ParallelGPT(nn.Module):
    def __init__(
        self, vocab_size: int, pg, block_size: int, n_layer: int, n_head: int, n_embd: int
    ) -> None:
        super().__init__()
        self.block_size = block_size
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Parameter(torch.zeros(1, block_size, n_embd))
        self.blocks = nn.ModuleList(
            ParallelBlock(n_embd, n_head, pg) for _ in range(n_layer)
        )
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight  # tied
        self.apply(self._init)

    @staticmethod
    def _init(module: nn.Module) -> None:
        # Only the replicated layers: the parallel layers initialize their own
        # shards, and isinstance skips them since they are not nn.Linear.
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        x = self.tok_emb(idx) + self.pos_emb[:, : idx.shape[1]]
        for block in self.blocks:
            x = block(x)
        logits = self.head(self.ln_f(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), targets.reshape(-1))
        return logits, loss


def load_corpus() -> str:
    if not DATA_PATH.exists():
        DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(DATA_URL, DATA_PATH)
    return DATA_PATH.read_text(encoding="utf-8")


def worker(pg, data: torch.Tensor, vocab_size: int, cfg: dict) -> float:
    # Identical seed everywhere: replicated layers must start in step, and
    # each rank's shard of the split layers is initialized independently.
    torch.manual_seed(0)
    model = ParallelGPT(
        vocab_size,
        pg,
        block_size=cfg["block_size"],
        n_layer=cfg["n_layer"],
        n_head=cfg["n_head"],
        n_embd=cfg["n_embd"],
    )
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], betas=(0.9, 0.95))

    local = sum(p.numel() for p in model.parameters())
    block_local = sum(p.numel() for b in model.blocks for p in b.parameters())
    if pg.rank == 0:
        print(
            f"per-rank params: {local / 1e6:.2f}M "
            f"(block weights held locally: {block_local / 1e6:.2f}M, "
            f"~1/{pg.world_size} of the full blocks) | world_size={pg.world_size}"
        )

    block, batch = cfg["block_size"], cfg["batch_size"]
    final_loss = 0.0
    window_start, window_tokens = time.perf_counter(), 0

    for step in range(cfg["steps"]):
        # Same batch on every rank: the model is split, not the data.
        gen = torch.Generator().manual_seed(step)
        ix = torch.randint(len(data) - block - 1, (batch,), generator=gen)
        x = torch.stack([data[i : i + block] for i in ix])
        y = torch.stack([data[i + 1 : i + block + 1] for i in ix])

        opt.zero_grad(set_to_none=True)
        _, loss = model(x, y)
        loss.backward()
        opt.step()

        window_tokens += batch * block
        if step % 20 == 0 or step == cfg["steps"] - 1:
            final_loss = loss.item()
            if pg.rank == 0:
                elapsed = time.perf_counter() - window_start
                print(
                    f"step {step:4d} | loss {final_loss:.4f} | "
                    f"{window_tokens / elapsed:,.0f} tok/s"
                )
                window_start, window_tokens = time.perf_counter(), 0

    # Every rank must agree on the loss, since they all ran the same batch
    # through the same (distributed) model.
    spread = torch.tensor([final_loss])
    gathered = collectives.all_gather(pg, spread)
    losses = [g.item() for g in gathered]
    if pg.rank == 0 and pg.world_size > 1:
        print(f"\nfinal loss per rank: {[f'{v:.6f}' for v in losses]}")
        print(f"max disagreement: {max(losses) - min(losses):.2e}")
    return final_loss


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--world-size", type=int, default=2)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--block-size", type=int, default=64)
    ap.add_argument("--n-layer", type=int, default=4)
    ap.add_argument("--n-head", type=int, default=4)
    ap.add_argument("--n-embd", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    args = ap.parse_args()

    text = load_corpus()
    vocab = "".join(sorted(set(text)))
    stoi = {ch: i for i, ch in enumerate(vocab)}
    data = torch.tensor([stoi[ch] for ch in text], dtype=torch.long)
    print(f"corpus: {len(text):,} chars, vocab {len(vocab)}")

    cfg = vars(args).copy()
    world_size = cfg.pop("world_size")
    if args.n_head % world_size:
        raise SystemExit(f"n_head={args.n_head} must be divisible by world_size={world_size}")
    losses = mn.run(worker, world_size, data, len(vocab), cfg, timeout=3600.0)
    print(f"final loss: {losses[0]:.4f}")


if __name__ == "__main__":
    main()
