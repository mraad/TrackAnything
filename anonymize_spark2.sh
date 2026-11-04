#!/usr/bin/env bash
# Remove personal data from a clip on spark2 (manual 5.5): copy the clip there, run tools/anonymize.py, copy the
# result back. The original goes into a private (0700) directory on the host, which is removed afterwards.
# ./anonymize_spark2.sh my_clip.mov data/garbage_cans.mp4 --subject "garbage can . trash can"
set -euo pipefail
cd "$(dirname "$0")"
[ $# -ge 3 ] || { echo "usage: $0 SRC OUT --subject NAMES [tools/anonymize.py options]" >&2; exit 2; }
SRC=$1 OUT=$2; shift 2
[ -f "$SRC" ] || { echo "no such file: $SRC" >&2; exit 2; }
if [ -e "$OUT" ] && [ "$SRC" -ef "$OUT" ]; then echo "OUT is the source clip: refusing to overwrite it" >&2; exit 2; fi
HOST=${HOST:-spark2}
REMOTE=GWorkspace/TrackAnything
SRC_ABS=$(cd "$(dirname "$SRC")" && pwd -P)/$(basename "$SRC")
EXCL=()
case "$SRC_ABS" in "$(pwd -P)"/*)  # never sync the original itself; escape rsync's pattern characters in its name
  REL=${SRC_ABS#"$(pwd -P)"/}
  # rsync reads \ as an escape only in a pattern with * ? or [; escape only such names
  case "$REL" in *[*?[]*) PAT=$(printf '%s' "$REL" | sed 's/[][*?\\]/\\&/g') ;; *) PAT=$REL ;; esac
  EXCL=(--exclude "/$PAT") ;;
esac
ssh "$HOST" mkdir -p "$REMOTE/results"
rsync -az --delete --exclude .venv --exclude results/ --exclude __pycache__ ${EXCL[@]+"${EXCL[@]}"} ./ "$HOST:$REMOTE/"
# Private per-run directory (mktemp -d makes it 0700). A host-side timer removes it after 6 h even if this script and
# its SSH connection die; the normal path removes it at the end.
D=$(ssh "$HOST" "d=\$(mktemp -d \$HOME/$REMOTE/results/anon.XXXXXX) && (nohup sh -c 'sleep 21600; rm -rf \"\$0\"' \"\$d\" </dev/null >/dev/null 2>&1 &) && echo \$d")
cleanup() {
  ssh "$HOST" "rm -rf '$D'" 2>/dev/null \
    || echo "WARNING: could not confirm cleanup on $HOST; delete $D there (a timer removes it within 6 h)" >&2
}
trap cleanup EXIT
scp -q "$SRC" "$HOST:$D/src"
# The host deletes the original as soon as the tool ends, also on hang-up or interrupt (connection loss).
# TRITON_CACHE_DIR: spark3's ~/.triton/cache belongs to root (manual 11); harmless elsewhere.
ssh "$HOST" "trap 'rm -f $D/src' EXIT HUP INT TERM; cd $REMOTE && TRITON_CACHE_DIR=\$HOME/.cache/triton-trackanything \
  ~/.local/bin/uv run --with imageio-ffmpeg python tools/anonymize.py $D/src --out $D/out.mp4 $(printf '%q ' "$@")"
mkdir -p "$(dirname "$OUT")" results
scp -q "$HOST:$D/out.mp4" "$OUT"
SHEET="results/$(basename "${OUT%.*}")_sheet.jpg"  # in results/ (git-ignored), not next to OUT in data/
scp -q "$HOST:$D/out_sheet.jpg" "$SHEET"
echo "wrote $OUT; review $SHEET (and the video) before you commit it"
