#!/usr/bin/env bash
# Sync this project to spark2, run track.py there with uv, copy results back.
# ./run_spark2.sh data/albatross.mp4 --point 470,260 --out results/birds_s.mp4
set -euo pipefail
cd "$(dirname "$0")"
HOST=${HOST:-spark2}
REMOTE=GWorkspace/TrackAnything
ssh "$HOST" mkdir -p "$REMOTE"
rsync -az --delete --exclude .venv --exclude results/ --exclude __pycache__ ./ "$HOST:$REMOTE/"
# HF_TOKEN (needed for gated facebook/sam3) goes over stdin, not argv, so it stays out of remote `ps`.
printf '%s\n' "${HF_TOKEN:-}" | ssh "$HOST" "read -r t; [ -n \"\$t\" ] && export HF_TOKEN=\$t; cd $REMOTE && mkdir -p results && ~/.local/bin/uv run track.py $(printf '%q ' "$@") </dev/null"
mkdir -p results && rsync -az "$HOST:$REMOTE/results/" results/
