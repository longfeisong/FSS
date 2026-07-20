#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
FABLE_RUN_DIR="${1:?usage: run_sigma_input_bridge.sh <fable-run-directory>}"
PORT="${SIGMA_INPUT_BRIDGE_PORT:-18820}"
OUT_DIR="$FABLE_RUN_DIR/sigma-input"
MASK_DIR="$OUT_DIR/dealer"

for file in "$FABLE_RUN_DIR/P0-share.u64" "$FABLE_RUN_DIR/P1-share.u64"; do
    [[ -f "$file" ]] || { printf 'Missing %s\n' "$file" >&2; exit 1; }
done
if ss -ltn | awk '{print $4}' | rg -q ":${PORT}$"; then
    printf 'Port %d is already in use.\n' "$PORT" >&2
    exit 1
fi

mkdir -p "$OUT_DIR"
conda run -n sigma-fable python "$SCRIPT_DIR/generate_sigma_input_masks.py" \
    --output-dir "$MASK_DIR" >"$OUT_DIR/dealer.log"

conda run -n sigma-fable python "$SCRIPT_DIR/share_to_sigma_masked.py" \
    --party 0 --port "$PORT" \
    --share "$FABLE_RUN_DIR/P0-share.u64" \
    --mask-share "$MASK_DIR/P0-mask-share.u64" \
    --output "$OUT_DIR/P0-masked-input.u64" >"$OUT_DIR/P0.log" 2>&1 &
p0_pid=$!
trap 'kill "$p0_pid" 2>/dev/null || true' EXIT INT TERM

conda run -n sigma-fable python "$SCRIPT_DIR/share_to_sigma_masked.py" \
    --party 1 --port "$PORT" \
    --share "$FABLE_RUN_DIR/P1-share.u64" \
    --mask-share "$MASK_DIR/P1-mask-share.u64" \
    --output "$OUT_DIR/P1-masked-input.u64" >"$OUT_DIR/P1.log" 2>&1
wait "$p0_pid"
trap - EXIT INT TERM

conda run -n sigma-fable python "$SCRIPT_DIR/verify_sigma_input_bridge.py" \
    --share0 "$FABLE_RUN_DIR/P0-share.u64" \
    --share1 "$FABLE_RUN_DIR/P1-share.u64" \
    --dealer-mask "$MASK_DIR/dealer-input-mask.u64" \
    --masked0 "$OUT_DIR/P0-masked-input.u64" \
    --masked1 "$OUT_DIR/P1-masked-input.u64" \
    --json-output "$OUT_DIR/verification.json"

printf 'SIGMA masked-input bridge completed: %s\n' "$OUT_DIR"
