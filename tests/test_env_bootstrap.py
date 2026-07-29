"""The environment-variable bootstrap, which is the multi-node contract.

``mn.run`` spawns and wires up ranks itself, but on a real cluster each rank
is launched independently (ssh, Slurm, a container per rank) and discovers
the others from the environment. That path has no launcher to lean on, so it
is tested here the way it is actually used: separate processes that only
share environment variables.
"""

from __future__ import annotations

import os
import subprocess
import sys

import pytest

from mini_nccl import _hosts_from_env
from mini_nccl.launcher import _free_ports

WORKER = """
import torch
import mini_nccl as mn

pg = mn.init_process_group()
try:
    assert pg.world_size == 2, pg.world_size
    t = torch.full((256,), float(pg.rank + 1))
    mn.all_reduce(t)
    torch.testing.assert_close(t, torch.full((256,), 3.0))
    gathered = mn.all_gather(torch.tensor([float(pg.rank)]))
    assert [g.item() for g in gathered] == [0.0, 1.0], gathered
    mn.barrier()
finally:
    mn.destroy_process_group()
print("OK", pg.rank)
"""


def test_hosts_from_env_parsing(monkeypatch) -> None:
    monkeypatch.setenv("MINI_NCCL_HOSTS", "10.0.0.1:29500, host-b:1234 ,")
    assert _hosts_from_env() == [("10.0.0.1", 29500), ("host-b", 1234)]

    monkeypatch.setenv("MINI_NCCL_HOSTS", "missing-port")
    with pytest.raises(ValueError, match="host:port"):
        _hosts_from_env()

    monkeypatch.delenv("MINI_NCCL_HOSTS")
    assert _hosts_from_env() is None


def test_two_independent_processes_rendezvous() -> None:
    ports = _free_ports(2)
    hosts = ",".join(f"127.0.0.1:{p}" for p in ports)
    procs = []
    for rank in range(2):
        env = dict(
            os.environ, MINI_NCCL_HOSTS=hosts, WORLD_SIZE="2", RANK=str(rank)
        )
        procs.append(
            subprocess.Popen(
                [sys.executable, "-c", WORKER],
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
        )
    outputs = []
    for proc in procs:
        stdout, _ = proc.communicate(timeout=180)
        outputs.append((proc.returncode, stdout))
    for rank, (code, stdout) in enumerate(outputs):
        assert code == 0, f"rank {rank} failed:\n{stdout}"
        assert f"OK {rank}" in stdout, stdout
