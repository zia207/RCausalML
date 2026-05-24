#!/usr/bin/env bash
set -o pipefail

TUTORIALS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG="${TUTORIALS_DIR}/_render_all.log"
SUMMARY="${TUTORIALS_DIR}/_render_all_summary.txt"

cd "$TUTORIALS_DIR"
export CAUSALML_FAST_RENDER="${CAUSALML_FAST_RENDER:-TRUE}"

: > "$LOG"
: > "$SUMMARY"

echo "Started: $(date -Iseconds)" | tee -a "$LOG" "$SUMMARY"
echo "CAUSALML_FAST_RENDER=$CAUSALML_FAST_RENDER" | tee -a "$LOG"

success=0
fail=0
mapfile -t files < <(find "$TUTORIALS_DIR" -maxdepth 1 -name '*.qmd' -printf '%f\n' | sort)

for f in "${files[@]}"; do
  echo "" | tee -a "$LOG"
  echo "=== Rendering $f at $(date -Iseconds) ===" | tee -a "$LOG"
  if quarto render "$f" >> "$LOG" 2>&1; then
    echo "OK: $f" | tee -a "$LOG" "$SUMMARY"
    success=$((success + 1))
  else
    echo "FAIL: $f" | tee -a "$LOG" "$SUMMARY"
    fail=$((fail + 1))
  fi
done

echo "" | tee -a "$LOG" "$SUMMARY"
echo "Finished: $(date -Iseconds)" | tee -a "$LOG" "$SUMMARY"
echo "Results: $success succeeded, $fail failed, $((success + fail)) total" | tee -a "$LOG" "$SUMMARY"

exit "$fail"
