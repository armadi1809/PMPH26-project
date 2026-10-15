#!/usr/bin/env bash
set -euo pipefail

HOST="${HENDRIX_HOST:-rsc784@hendrixfut03fl}"
REMOTE_DIR="PMPH26-project"
LOCAL_DIR="$(cd "$(dirname "$0")" && pwd)"
SSH_OPTS=(-o ControlMaster=auto -o ControlPath="/tmp/cm-hendrix-$$" -o ControlPersist=10m)
CMD="${*:-make -C cuda bench}"

trap 'ssh "${SSH_OPTS[@]}" -O exit "$HOST" 2>/dev/null || true' EXIT

ssh "${SSH_OPTS[@]}" "$HOST" "mkdir -p ~/$REMOTE_DIR"
rsync -az --exclude .git --exclude results --exclude-from "$LOCAL_DIR/.gitignore" \
  -e "ssh ${SSH_OPTS[*]}" "$LOCAL_DIR/" "$HOST:~/$REMOTE_DIR/"

ssh "${SSH_OPTS[@]}" "$HOST" bash -l -s <<EOF
set -eo pipefail
module load cuda/12.8 futhark
cd ~/$REMOTE_DIR
$CMD
EOF

for dir in cuda futhark; do
  rsync -az -e "ssh ${SSH_OPTS[*]}" "$HOST:~/$REMOTE_DIR/$dir/results/" "$LOCAL_DIR/$dir/results/" 2>/dev/null || true
done
