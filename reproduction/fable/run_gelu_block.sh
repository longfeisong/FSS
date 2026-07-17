#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
FABLE_DIR="$ROOT_DIR/src/FABLE"
BINARY="$FABLE_DIR/build-gelu/bin/fable"
CONDA_PREFIX_PATH="${CONDA_PREFIX:-/home/slf/anaconda3/envs/sigma-fable}"
LIBOTE_PREFIX="$ROOT_DIR/third_party/libOTe/install"
RESULTS_DIR="${FABLE_RESULTS_DIR:-$SCRIPT_DIR/results}"

CHUNKS="${1:-16}"
BATCH_SIZE="${2:-4096}"
THREADS="${3:-16}"
PORT_BASE="${FABLE_PORT_BASE:-18200}"
RUN_ID="$(date +%Y%m%dT%H%M%S)-fable-gelu-block"
RAW_DIR="$RESULTS_DIR/raw/$RUN_ID"

if [[ ! -x "$BINARY" ]]; then
    printf 'FABLE binary not found: %s\nRun %s first.\n' "$BINARY" "$SCRIPT_DIR/build_gelu.sh" >&2
    exit 1
fi

cleanup() {
    local pid exe
    while read -r pid; do
        [[ -n "$pid" ]] || continue
        exe="$(readlink -f "/proc/$pid/exe" 2>/dev/null || true)"
        if [[ "$exe" == "$(readlink -f "$BINARY")" ]]; then
            kill "$pid" 2>/dev/null || true
        fi
    done < <(pgrep -f "$BINARY" || true)
}

cleanup
trap cleanup EXIT INT TERM
mkdir -p "$RAW_DIR"
export LD_LIBRARY_PATH="$CONDA_PREFIX_PATH/lib:$LIBOTE_PREFIX/lib:${LD_LIBRARY_PATH:-}"

start_ms="$(date +%s%3N)"
for chunk in $(seq 1 "$CHUNKS"); do
    port=$((PORT_BASE + chunk))
    if ss -ltn | awk '{print $4}' | rg -q ":${port}$"; then
        printf 'Port %d is already in use.\n' "$port" >&2
        exit 1
    fi

    server_log="$RAW_DIR/chunk-$(printf '%02d' "$chunk")-server.log"
    client_log="$RAW_DIR/chunk-$(printf '%02d' "$chunk")-client.log"

    "$BINARY" 127.0.0.1 r=1 p="$port" bs="$BATCH_SIZE" db=65536 \
        l=4 par=1 thr="$THREADS" >"$server_log" 2>&1 &
    server_pid=$!
    sleep 0.2
    "$BINARY" 127.0.0.1 r=2 p="$port" bs="$BATCH_SIZE" db=65536 \
        l=4 par=1 thr="$THREADS" >"$client_log" 2>&1
    wait "$server_pid"

    rg -q '\[FABLE\] Test passed' "$server_log"
    rg -q '\[FABLE\] Test passed' "$client_log"
    printf 'FABLE GELU chunk %d/%d passed\n' "$chunk" "$CHUNKS"
done
wall_ms=$(($(date +%s%3N) - start_ms))

python3 "$SCRIPT_DIR/summarize.py" \
    --run-id "$RUN_ID" \
    --raw-dir "$RAW_DIR" \
    --results-dir "$RESULTS_DIR" \
    --chunks "$CHUNKS" \
    --batch-size "$BATCH_SIZE" \
    --threads "$THREADS" \
    --wall-ms "$wall_ms"

printf 'Saved FABLE result under %s\n' "$RAW_DIR"
