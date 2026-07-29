# Running across machines

Every number in the README was measured on loopback TCP, where "the network"
is a memory copy performed by the same cores that run the reduction. That
compresses the differences between algorithms: it understates ring's
advantage (its whole point is moving fewer bytes, which only matters when
bytes are expensive) and flatters the naive parameter-server pattern (whose
weakness is a central bottleneck that loopback barely penalizes).

Nothing in the algorithms is loopback-specific. The address book is a list of
`(host, port)` pairs, so the same code runs across machines once each rank
knows the list.

## The contract

Each rank is its own process and learns who it is from the environment:

| variable | meaning |
|---|---|
| `MINI_NCCL_HOSTS` | comma-separated `host:port`, in rank order, identical on every rank |
| `RANK` | this process's rank, `0 .. WORLD_SIZE-1` |
| `WORLD_SIZE` | total rank count |
| `MINI_NCCL_CHANNELS` | connections per peer (optional; default 2) |
| `MINI_NCCL_TRACE` | `1` to enable the flight recorder |

`mn.init_process_group()` with no arguments reads all of them. Rank `r` binds
the `r`-th entry in the host list, so the hostnames must be the addresses
peers can actually reach (not `127.0.0.1`), and the port must be open between
hosts.

## Two hosts by hand

```bash
# host A (10.0.0.1)
MINI_NCCL_HOSTS=10.0.0.1:29500,10.0.0.2:29500 WORLD_SIZE=2 RANK=0 \
  python examples/multinode_allreduce.py

# host B (10.0.0.2)
MINI_NCCL_HOSTS=10.0.0.1:29500,10.0.0.2:29500 WORLD_SIZE=2 RANK=1 \
  python examples/multinode_allreduce.py
```

Rank 0 prints the table. Ranks may start in any order: dialing retries until
the rendezvous timeout.

## Fanning out over ssh

```bash
cat > hosts.txt <<'EOF'
10.0.0.1:29500
10.0.0.2:29500
EOF

./scripts/launch_multinode.sh hosts.txt examples/multinode_allreduce.py --iters 20
```

The script needs passwordless ssh and the repo at the same path on each host
(override with `REMOTE_DIR`). Rank 0 runs locally so its output stays in your
terminal.

## What to expect, and what to re-tune

On a real link the shape of the results should change in three ways. They are
worth checking explicitly, because each one is a prediction the design makes:

1. **Naive should fall apart.** Rank 0 moves `2(W-1)n` bytes through one
   link, so its time should grow with world size while ring's stays flat.
2. **Ring's advantage should grow with `W`.** Per-rank traffic is
   `2(W-1)/W · n`, which approaches `2n` and stops growing; tree keeps
   forwarding `2⌈log₂W⌉ · n`.
3. **More channels should help more.** Two was optimal on loopback because
   extra threads competed for the same cores. A NIC that needs several flows
   to saturate should prefer 4-8. Sweep it:

   ```bash
   for c in 1 2 4 8; do
     MINI_NCCL_CHANNELS=$c ./scripts/launch_multinode.sh hosts.txt \
       examples/multinode_allreduce.py
   done
   ```

4. **Slice pipelining may flip from a loss to a win.** It is off by default
   (`MINI_NCCL_MAX_SLICES=1`) because on loopback the reduction and the
   transfer contend for the same cores. Where the transport is a NIC doing
   DMA, the reduction runs on hardware the transfer is not using, and
   overlapping them should pay. Test with `MINI_NCCL_MAX_SLICES=16`.

Then re-derive the two thresholds in `mini_nccl/collectives.py`
(`RING_THRESHOLD_BYTES`, `CHANNEL_MIN_BYTES`) from your own numbers, the same
way `benchmarks/bench_ablation.py` derived the loopback ones.

## Debugging a run that will not start

- **Rendezvous timeout naming specific ranks**: those hosts could not be
  reached. Check the port is open (`nc -vz host 29500`) and that the host
  list uses routable addresses.
- **`CollectiveTimeoutError`**: the mesh formed but a peer stopped
  participating. Re-run with `MINI_NCCL_TRACE=1`, have each rank dump its
  recorder (`pg.recorder.dump(dir)`), collect the files, and run
  `python -m mini_nccl.diagnose <dir>` to find which rank diverged.
- **Mismatched host lists**: ranks that disagree about the list will connect
  to the wrong peers. The list must be byte-identical everywhere.
