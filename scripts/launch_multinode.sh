#!/usr/bin/env bash
# Launch one rank per host over ssh.
#
#   ./scripts/launch_multinode.sh hosts.txt examples/multinode_allreduce.py
#
# hosts.txt holds one "host:port" per line, in rank order:
#
#   10.0.0.1:29500
#   10.0.0.2:29500
#
# Requirements: passwordless ssh to each host, the repo at the same path
# everywhere (or set REMOTE_DIR), and the port reachable between hosts.
# Rank 0 runs locally so its output stays in this terminal.
set -euo pipefail

HOSTFILE=${1:?usage: launch_multinode.sh <hostfile> <script> [args...]}
SCRIPT=${2:?usage: launch_multinode.sh <hostfile> <script> [args...]}
shift 2

REMOTE_DIR=${REMOTE_DIR:-$(pwd)}
PYTHON=${PYTHON:-python}

mapfile -t ENTRIES < <(grep -v '^[[:space:]]*\(#\|$\)' "$HOSTFILE")
WORLD_SIZE=${#ENTRIES[@]}
HOSTS=$(IFS=,; echo "${ENTRIES[*]}")

echo "world_size=$WORLD_SIZE"
echo "hosts=$HOSTS"

pids=()
for rank in "${!ENTRIES[@]}"; do
  host=${ENTRIES[$rank]%%:*}
  env_vars="MINI_NCCL_HOSTS=$HOSTS WORLD_SIZE=$WORLD_SIZE RANK=$rank"
  if [[ $rank -eq 0 ]]; then
    echo "rank 0 -> local"
    env MINI_NCCL_HOSTS="$HOSTS" WORLD_SIZE="$WORLD_SIZE" RANK=0 \
      "$PYTHON" "$SCRIPT" "$@" &
  else
    echo "rank $rank -> $host"
    # shellcheck disable=SC2029  # deliberate client-side expansion
    ssh "$host" "cd $REMOTE_DIR && $env_vars $PYTHON $SCRIPT $*" &
  fi
  pids+=($!)
done

status=0
for pid in "${pids[@]}"; do
  wait "$pid" || status=$?
done
exit "$status"
