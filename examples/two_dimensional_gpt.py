"""A GPT trained with tensor parallelism *and* data parallelism at once.

    python examples/two_dimensional_gpt.py --world-size 4 --tp 2

Four ranks arranged as a 2x2 mesh. Each transformer block's attention and MLP
are split across the tensor dimension, and gradients are averaged across the
data dimension, so the run is two replicas of a half-width-per-rank model.

This is what the strategies composing looks like in practice, and the reason it
needs sub-group communicators: a tensor-parallel all-reduce must reach only the
ranks sharing that layer, while a data-parallel all-reduce must reach only the
corresponding ranks of each replica. With ``dp=2, tp=2`` the tensor groups are
``[0,1]`` and ``[2,3]`` and the data groups are ``[0,2]`` and ``[1,3]``: two
orthogonal partitions, so the two kinds of traffic never share a socket.

The gradient rule is worth stating because it is easy to get wrong. Sharded
weights are reduced **only** along the data dimension: each tensor-parallel rank
owns a different slice, so there is nothing to average along that axis.
Replicated weights are also reduced only along the data dimension, because the
backward all-reduce inside the tensor-parallel layers has already made their
gradients identical within a tensor group.
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
from mini_nccl.mesh import ParallelMesh
from mini_nccl.tensor_parallel import ParallelMLP, ParallelSelfAttention

DATA_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/"
    "data/tinyshakespeare/input.txt"
)
DATA_PATH = Path(__file__).parent.parent / "data" / "tinyshakespeare.txt"


class Block(nn.Module):
    def __init__(self, width: int, n_head: int, tp_group) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(width)
        self.attn = ParallelSelfAttention(width, n_head, tp_group)
        self.ln2 = nn.LayerNorm(width)
        self.mlp = ParallelMLP(width, tp_group)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        return x + self.mlp(self.ln2(x))


class GPT(nn.Module):
    def __init__(self, vocab: int, tp_group, cfg: dict) -> None:
        super().__init__()
        width = cfg["n_embd"]
        self.tok_emb = nn.Embedding(vocab, width)
        self.pos_emb = nn.Parameter(torch.zeros(1, cfg["block_size"], width))
        self.blocks = nn.ModuleList(
            Block(width, cfg["n_head"], tp_group) for _ in range(cfg["n_layer"])
        )
        self.ln_f = nn.LayerNorm(width)
        self.head = nn.Linear(width, vocab, bias=False)
        self.head.weight = self.tok_emb.weight
        self.apply(self._init)

    @staticmethod
    def _init(module: nn.Module) -> None:
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
    mesh = ParallelMesh(pg, dp=cfg["dp"], tp=cfg["tp"])
    tp_group, dp_group = mesh.group("tp"), mesh.group("dp")

    # Replicated layers must start identical across the whole world; DDP over
    # the data dimension only syncs within a replica column, so the seed does
    # the rest.
    torch.manual_seed(0)
    model = GPT(vocab_size, tp_group, cfg)
    ddp = mn.DistributedDataParallel(model, dp_group, bucket_cap_mb=0.5)
    opt = torch.optim.AdamW(ddp.parameters(), lr=cfg["lr"], betas=(0.9, 0.95))

    local = sum(p.numel() for p in model.parameters())
    if pg.rank == 0:
        print(
            f"mesh dp={mesh.size('dp')} x tp={mesh.size('tp')} | "
            f"{mesh.describe()}\n"
            f"tensor groups: {mesh.ranks_along('tp')} for this rank | "
            f"data groups: {mesh.ranks_along('dp')}\n"
            f"per-rank params: {local / 1e6:.2f}M"
        )

    block, batch = cfg["block_size"], cfg["batch_size"]
    final_loss = 0.0
    window_start, window_tokens = time.perf_counter(), 0

    for step in range(cfg["steps"]):
        # Replicas take different data; ranks within a tensor group share it.
        seed = step * mesh.size("dp") + mesh.coordinate("dp")
        gen = torch.Generator().manual_seed(seed)
        ix = torch.randint(len(data) - block - 1, (batch,), generator=gen)
        x = torch.stack([data[i : i + block] for i in ix])
        y = torch.stack([data[i + 1 : i + block + 1] for i in ix])

        ddp.zero_grad()
        _, loss = ddp(x, y)
        loss.backward()
        ddp.sync()
        opt.step()

        window_tokens += batch * block * mesh.size("dp")
        if step % 20 == 0 or step == cfg["steps"] - 1:
            # Average across the data dimension for a global figure.
            global_loss = loss.detach().clone().view(1)
            collectives.all_reduce(dp_group, global_loss)
            global_loss /= mesh.size("dp")
            final_loss = global_loss.item()
            if pg.rank == 0:
                elapsed = time.perf_counter() - window_start
                print(
                    f"step {step:4d} | loss {final_loss:.4f} | "
                    f"{window_tokens / elapsed:,.0f} tok/s"
                )
                window_start, window_tokens = time.perf_counter(), 0

    # Ranks in the same tensor group ran the same data, so their losses must
    # agree exactly; ranks in different replicas need not.
    same_tp = collectives.all_gather(tp_group, torch.tensor([final_loss]))
    if pg.rank == 0:
        values = [f"{t.item():.6f}" for t in same_tp]
        print(f"\nloss across the tensor group (same data, split model): {values}")
        print(f"spread: {max(t.item() for t in same_tp) - min(t.item() for t in same_tp):.2e}")
    return final_loss


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--world-size", type=int, default=4)
    ap.add_argument("--tp", type=int, default=2, help="tensor-parallel width")
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=8, help="per replica")
    ap.add_argument("--block-size", type=int, default=64)
    ap.add_argument("--n-layer", type=int, default=4)
    ap.add_argument("--n-head", type=int, default=4)
    ap.add_argument("--n-embd", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    args = ap.parse_args()

    if args.world_size % args.tp:
        raise SystemExit(f"world_size={args.world_size} must divide by tp={args.tp}")
    if args.n_head % args.tp:
        raise SystemExit(f"n_head={args.n_head} must divide by tp={args.tp}")

    text = load_corpus()
    vocab = "".join(sorted(set(text)))
    stoi = {ch: i for i, ch in enumerate(vocab)}
    data = torch.tensor([stoi[ch] for ch in text], dtype=torch.long)
    print(f"corpus: {len(text):,} chars, vocab {len(vocab)}")

    cfg = vars(args).copy()
    world = cfg.pop("world_size")
    cfg["dp"] = world // args.tp
    losses = mn.run(worker, world, data, len(vocab), cfg, timeout=3600.0)
    print(f"final loss: {losses[0]:.4f}")


if __name__ == "__main__":
    main()
