#!/usr/bin/env bash
# Sync this project to spark2 and run the GPU test suite there (tests/, stdlib unittest).
# ./test_spark2.sh [-k pattern]          TEST_MODEL=sam2 ./test_spark2.sh -k Text          HOST=spark3 ./test_spark2.sh
set -euo pipefail
cd "$(dirname "$0")"
HOST=${HOST:-spark2}; REMOTE=GWorkspace/TrackAnything; MODEL=${TEST_MODEL:-etam-s}
ARGS=""; [ $# -gt 0 ] && ARGS=$(printf '%q ' "$@")
ssh "$HOST" mkdir -p "$REMOTE"
rsync -az --delete --exclude .venv --exclude results/ --exclude __pycache__ ./ "$HOST:$REMOTE/"
# HF_TOKEN (only TEST_MODEL=sam3 needs it) goes over stdin, not argv. imageio-ffmpeg: the hosts have no ffmpeg.
# spark3 has a root-owned ~/.triton/cache, so Triton gets a cache of its own there.
printf '%s\n' "${HF_TOKEN:-}" | ssh "$HOST" "read -r t; [ -n \"\$t\" ] && export HF_TOKEN=\$t; cd $REMOTE && \
  { [ -w ~/.triton/cache ] || export TRITON_CACHE_DIR=~/.cache/triton-trackanything; } && \
  TEST_MODEL=$(printf '%q' "$MODEL") ~/.local/bin/uv run --with imageio-ffmpeg python -m unittest discover -s tests -v $ARGS </dev/null"
