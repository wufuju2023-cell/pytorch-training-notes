#!/bin/bash
# Runs on the cloud GPU container (as root), detached.
set -u
BASE=/mnt/workspace/alphaproof-learn
LOG=$BASE/exec_all.log
cd "$BASE" || exit 1
rm -rf stage && mkdir -p stage
tar xzf v1-stage.tgz -C stage
ROOT="$BASE/stage/v1-stage"
: > "$LOG"
echo "START $(date -Is)" >> "$LOG"
find "$ROOT/notebooks" -name 'N[0-9]*.ipynb' | sort | while read -r nb; do
  echo "=== EXEC ${nb#$ROOT/}" >> "$LOG"
  timeout 1200 jupyter nbconvert --to notebook --execute --inplace "$nb" \
      --ExecutePreprocessor.timeout=900 >> "$LOG" 2>&1
  echo "RC=$? ${nb#$ROOT/}" >> "$LOG"
done
echo "ALL_DONE $(date -Is)" >> "$LOG"
