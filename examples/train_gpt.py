"""Train a character-level GPT data-parallel — every gradient byte moves
through mini-nccl's own ring all-reduce, not torch.distributed.

    python examples/train_gpt.py --world-size 4 --steps 200 --sample

Each rank samples its own batches; gradients are averaged across ranks by
the bucketed-overlap DDP wrapper. Run with --world-size 1 to compare
single-process throughput and loss curves.
"""

from __future__ import annotations

import argparse
import sys
import time
import urllib.request
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))

import mini_nccl as mn
from mini_nccl import collectives
from model import GPT

DATA_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/"
    "data/tinyshakespeare/input.txt"
)
DATA_PATH = Path(__file__).parent.parent / "data" / "tinyshakespeare.txt"


def load_corpus() -> str:
    if not DATA_PATH.exists():
        DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
        print(f"downloading tiny shakespeare -> {DATA_PATH}")
        urllib.request.urlretrieve(DATA_URL, DATA_PATH)
    return DATA_PATH.read_text(encoding="utf-8")


def worker(pg, data: torch.Tensor, vocab: str, cfg: dict) -> float:
    torch.manual_seed(0)  # identical init everywhere; DDP broadcast enforces it anyway
    model = GPT(
        vocab_size=len(vocab),
        block_size=cfg["block_size"],
        n_layer=cfg["n_layer"],
        n_head=cfg["n_head"],
        n_embd=cfg["n_embd"],
    )
    ddp = mn.DistributedDataParallel(
        model,
        pg,
        bucket_cap_mb=cfg["bucket_mb"],
        overlap=not cfg["no_overlap"],
        algorithm=cfg["algorithm"],
    )
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], betas=(0.9, 0.95), weight_decay=0.1)

    n_params = sum(p.numel() for p in model.parameters())
    if pg.rank == 0:
        print(f"model: {n_params / 1e6:.2f}M params | world_size={pg.world_size} "
              f"| per-rank batch={cfg['batch_size']} | overlap={not cfg['no_overlap']}")

    block, batch = cfg["block_size"], cfg["batch_size"]
    window_start = time.perf_counter()
    window_tokens = 0
    final_loss = 0.0

    for step in range(cfg["steps"]):
        # Per-rank generator: every rank trains on different data each step.
        g = torch.Generator().manual_seed(step * pg.world_size + pg.rank)
        ix = torch.randint(len(data) - block - 1, (batch,), generator=g)
        x = torch.stack([data[i : i + block] for i in ix])
        y = torch.stack([data[i + 1 : i + block + 1] for i in ix])

        ddp.zero_grad()
        _, loss = ddp(x, y)
        loss.backward()
        ddp.sync()
        opt.step()

        window_tokens += batch * block * pg.world_size
        if step % 20 == 0 or step == cfg["steps"] - 1:
            # Average the loss across ranks for a global picture.
            global_loss = loss.detach().clone().view(1)
            collectives.all_reduce(pg, global_loss)
            global_loss /= pg.world_size
            final_loss = global_loss.item()
            if pg.rank == 0:
                elapsed = time.perf_counter() - window_start
                tok_s = window_tokens / elapsed if elapsed > 0 else 0.0
                print(f"step {step:4d} | loss {final_loss:.4f} | {tok_s:,.0f} tok/s")
                window_start = time.perf_counter()
                window_tokens = 0

    if pg.rank == 0 and cfg["sample"]:
        stoi = {ch: i for i, ch in enumerate(vocab)}
        prompt = torch.tensor([[stoi["\n"]]], dtype=torch.long)
        out = model.generate(prompt, max_new_tokens=300)
        text = "".join(vocab[i] for i in out[0].tolist())
        print("\n--- sample ---" + text + "\n--------------")
    return final_loss


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--world-size", type=int, default=2)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=16, help="per-rank batch size")
    ap.add_argument("--block-size", type=int, default=128)
    ap.add_argument("--n-layer", type=int, default=4)
    ap.add_argument("--n-head", type=int, default=4)
    ap.add_argument("--n-embd", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--bucket-mb", type=float, default=0.5)
    ap.add_argument("--no-overlap", action="store_true")
    ap.add_argument("--algorithm", choices=["ring", "tree", "naive", "auto"], default="ring")
    ap.add_argument("--sample", action="store_true", help="print generated text at the end")
    args = ap.parse_args()

    text = load_corpus()
    vocab = "".join(sorted(set(text)))
    stoi = {ch: i for i, ch in enumerate(vocab)}
    data = torch.tensor([stoi[ch] for ch in text], dtype=torch.long)
    print(f"corpus: {len(text):,} chars, vocab {len(vocab)}")

    cfg = vars(args).copy()
    world_size = cfg.pop("world_size")
    losses = mn.run(worker, world_size, data, vocab, cfg, timeout=3600.0)
    print(f"final loss: {losses[0]:.4f}")


if __name__ == "__main__":
    main()
