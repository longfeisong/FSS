#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
FABLE_DIR="$ROOT_DIR/src/FABLE"
BINARY="$FABLE_DIR/build-xlmr/bin/xlmr_embedding"
ENV_NAME="${FABLE_CONDA_ENV:-sigma-fable}"
if [[ "${CONDA_DEFAULT_ENV:-}" == "$ENV_NAME" && -n "${CONDA_PREFIX:-}" ]]; then
    CONDA_PREFIX_PATH="$CONDA_PREFIX"
elif command -v conda >/dev/null 2>&1; then
    CONDA_PREFIX_PATH="$(conda run -n "$ENV_NAME" bash -c 'printf %s "$CONDA_PREFIX"')"
else
    printf 'Activate Conda environment %s or make conda available in PATH.\n' "$ENV_NAME" >&2
    exit 1
fi
LIBOTE_PREFIX="$ROOT_DIR/third_party/libOTe/install"
MODEL_ARTIFACTS="${XLMR_ARTIFACTS_DIR:-$SCRIPT_DIR/artifacts}"
TABLE="${XLMR_FABLE_TABLE:-$MODEL_ARTIFACTS/fable-table/xlmr_word_embedding_scale12.i16}"
QUERIES="${XLMR_TOKEN_IDS:-$MODEL_ARTIFACTS/plaintext-bridge-smoke/token_ids.u32}"
CHUNK_START="${1:-0}"
CHUNK_COUNT="${2:-1}"
THREADS="${FABLE_THREADS:-32}"
P0_CPUSET="${FABLE_P0_CPUSET:-}"
P1_CPUSET="${FABLE_P1_CPUSET:-}"
PORT="${FABLE_XLMR_PORT:-18800}"
RUN_ID="$(date +%Y%m%dT%H%M%S)-fable-xlmr-c${CHUNK_START}-n${CHUNK_COUNT}"
RUN_DIR="$SCRIPT_DIR/artifacts/fable-xlmr/$RUN_ID"

if [[ ! -x "$BINARY" ]]; then
    printf 'Missing binary: %s\nRun %s first.\n' "$BINARY" "$SCRIPT_DIR/build_fable_xlmr.sh" >&2
    exit 1
fi
if [[ ! -f "$TABLE" || ! -f "$QUERIES" ]]; then
    printf 'Missing table or query artifact. Follow reproduction/xlmr_sigma/README.md.\n' >&2
    exit 1
fi
if ss -ltn | awk '{print $4}' | rg -q ":${PORT}$"; then
    printf 'Port %d is already in use.\n' "$PORT" >&2
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
mkdir -p "$RUN_DIR"
export LD_LIBRARY_PATH="$CONDA_PREFIX_PATH/lib:$LIBOTE_PREFIX/lib:${LD_LIBRARY_PATH:-}"

SERVER_SHARE="$RUN_DIR/P0-share.u64"
CLIENT_SHARE="$RUN_DIR/P1-share.u64"
SERVER_LOG="$RUN_DIR/P0.log"
CLIENT_LOG="$RUN_DIR/P1.log"

start_ms="$(date +%s%3N)"
pin() {
    local cpuset="$1"
    shift
    if [[ -n "$cpuset" ]]; then
        taskset -c "$cpuset" "$@"
    else
        "$@"
    fi
}
pin "$P0_CPUSET" "$BINARY" 127.0.0.1 r=1 p="$PORT" par=1 thr="$THREADS" \
    chunk_start="$CHUNK_START" chunk_count="$CHUNK_COUNT" \
    table="$TABLE" queries="$QUERIES" output="$SERVER_SHARE" \
    >"$SERVER_LOG" 2>&1 &
server_pid=$!
sleep 0.5
pin "$P1_CPUSET" "$BINARY" 127.0.0.1 r=2 p="$PORT" par=1 thr="$THREADS" \
    chunk_start="$CHUNK_START" chunk_count="$CHUNK_COUNT" \
    table="$TABLE" queries="$QUERIES" output="$CLIENT_SHARE" \
    >"$CLIENT_LOG" 2>&1
wait "$server_pid"
wall_ms=$(($(date +%s%3N) - start_ms))

rg -q 'FABLE-to-SIGMA share bridge passed' "$SERVER_LOG"
rg -q 'FABLE-to-SIGMA share bridge passed' "$CLIENT_LOG"

conda run -n "$ENV_NAME" python "$SCRIPT_DIR/verify_fable_sigma_shares.py" \
    --table "$TABLE" --queries "$QUERIES" \
    --share0 "$SERVER_SHARE" --share1 "$CLIENT_SHARE" \
    --chunk-start "$CHUNK_START" --chunk-count "$CHUNK_COUNT" \
    --json-output "$RUN_DIR/verification.json"

printf '{\n  "run_id": "%s",\n  "chunk_start": %s,\n  "chunk_count": %s,\n  "wall_ms": %s,\n  "threads": %s\n}\n' \
    "$RUN_ID" "$CHUNK_START" "$CHUNK_COUNT" "$wall_ms" "$THREADS" \
    >"$RUN_DIR/metadata.json"
printf 'FABLE XLM-R share bridge completed: %s ms\n' "$wall_ms"
printf 'Artifacts: %s\n' "$RUN_DIR"
