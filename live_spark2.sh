#!/usr/bin/env bash
# Run the live tracker on spark2 and open it on this Mac through an SSH tunnel.
# ./live_spark2.sh [--model sam2|sam3|etam-s|etam-ti]
# Stop: Ctrl-C, or `pkill -f live_spark2.sh` from another shell. Either one also stops the server on spark2.
set -euo pipefail
cd "$(dirname "$0")"
HOST=${HOST:-spark2}; REMOTE=GWorkspace/TrackAnything; PORT=${PORT:-8765}
ARGS=""; [ $# -gt 0 ] && ARGS=$(printf '%q ' "$@")
ssh "$HOST" mkdir -p "$REMOTE"
rsync -az --delete --exclude .venv --exclude results/ --exclude __pycache__ ./ "$HOST:$REMOTE/"
# NO_OPEN=1: no browser tab (tests/page_check.py); a second viewer would drive the same shared tracker.
( until curl -sf -o /dev/null "http://localhost:$PORT/"; do sleep 1; done; [ -n "${NO_OPEN:-}" ] || open "http://localhost:$PORT/" ) &
WAITER=$!
# Line 1 of stdin is HF_TOKEN (kept off argv). The local sleep keeps the pipe open. When ssh dies, the remote
# `cat` sees EOF and kills the server.
{ printf '%s\n' "${HF_TOKEN:-}"; exec sleep 2147483647; } | ssh -o ExitOnForwardFailure=yes -L "$PORT:127.0.0.1:$PORT" "$HOST" \
  "read -r t; [ -n \"\$t\" ] && export HF_TOKEN=\$t; cd $REMOTE && ~/.local/bin/uv sync -q && { .venv/bin/python live.py --port $PORT $ARGS & pid=\$!; cat >/dev/null; kill \$pid; }" &
SSH=$!
# ssh stays up while the browser holds forwarded connections, so kill it explicitly.
trap 'kill $SSH $WAITER 2>/dev/null; pkill -P $$ sleep 2>/dev/null; exit' EXIT INT TERM
wait $SSH
