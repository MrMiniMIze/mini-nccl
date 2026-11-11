"""Prove an installed mini-nccl works with nothing but its declared dependencies.

    pip install dist/mini_nccl-*.whl          # no extras
    cd /somewhere/else && python /path/to/scripts/smoke_install.py

The test suite always runs with the ``dev`` extra installed, so it cannot
notice a runtime dependency that is only declared there. That is exactly how
``numpy`` went missing: every send goes through ``Tensor.numpy()``, numpy was
listed only under ``dev``, and a plain install could not move a single byte
(``RuntimeError: Numpy is not available``) while every test passed.

This runs one real 2-rank all-reduce against the *installed* package. It
refuses to run against a source checkout, because importing the working tree
would prove nothing about what was installed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

import mini_nccl as mn
from mini_nccl import collectives as c

REPO = Path(__file__).resolve().parent.parent


def _worker(pg) -> float:
    tensor = torch.full((4096,), float(pg.rank + 1))
    c.all_reduce(pg, tensor, algorithm="ring")
    return float(tensor[0])


def main() -> None:
    installed = Path(mn.__file__).resolve()
    if REPO in installed.parents:
        sys.exit(
            f"smoke_install: imported mini_nccl from the checkout ({installed}). "
            "Install the wheel and run this from another directory."
        )
    print(f"mini_nccl {mn.__version__} from {installed.parent}")
    results = mn.run(_worker, 2, timeout=120.0)
    if results != [3.0, 3.0]:
        sys.exit(f"smoke_install: expected [3.0, 3.0], got {results}")
    print("2-rank all-reduce OK:", results)


if __name__ == "__main__":
    main()
