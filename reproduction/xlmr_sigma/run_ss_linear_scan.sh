#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
TABLE_SHARES="${1:?usage: run_ss_linear_scan.sh <table-share-dir> <triple-dir> [output-dir]}"
TRIPLES="${2:?usage: run_ss_linear_scan.sh <table-share-dir> <triple-dir> [output-dir]}"
PYTHON="${SS_LINEAR_SCAN_PYTHON:-python3}"
QUERIES="${XLMR_TOKEN_IDS:-$SCRIPT_DIR/artifacts/plaintext-bridge-smoke/token_ids.u32}"
CLEAR_TABLE="${XLMR_FABLE_TABLE:-$SCRIPT_DIR/artifacts/fable-table/xlmr_word_embedding_scale12.i16}"
PORT="${SS_LINEAR_SCAN_PORT:-18840}"
UNIX_SOCKET="${SS_LINEAR_SCAN_UNIX_SOCKET:-}"
BACKEND="${SS_LINEAR_SCAN_BACKEND:-torch-u50}"
P0_GPU="${SS_LINEAR_SCAN_P0_GPU:-0}"
P1_GPU="${SS_LINEAR_SCAN_P1_GPU:-4}"
P0_CPUSET="${SS_LINEAR_SCAN_P0_CPUSET:-}"
P1_CPUSET="${SS_LINEAR_SCAN_P1_CPUSET:-}"
RUN_ID="$(date +%Y%m%dT%H%M%S)-ss-linear-scan"
RUN_DIR="${3:-$SCRIPT_DIR/artifacts/ss-linear-scan/runs/$RUN_ID}"

for file in "$TABLE_SHARES/manifest.json" "$TRIPLES/manifest.json" "$QUERIES"; do
    [[ -f "$file" ]] || { printf 'Missing %s\n' "$file" >&2; exit 1; }
done
if [[ -z "$UNIX_SOCKET" ]] && ss -ltn | awk '{print $4}' | rg -q ":${PORT}$"; then
    printf 'Port %d is already in use.\n' "$PORT" >&2
    exit 1
fi
if [[ -n "$UNIX_SOCKET" && -e "$UNIX_SOCKET" ]]; then
    printf 'Unix socket path already exists: %s\n' "$UNIX_SOCKET" >&2
    exit 1
fi

read -r logical_n queries_count output_dim ring_bits < <(
    "$PYTHON" -c 'import json,sys; p=json.load(open(sys.argv[1]))["profile"]; print(p["logical_n"],p["queries"],p["output_dim"],p["ring_bits"])' \
        "$TRIPLES/manifest.json"
)
expected_query_bytes=$((queries_count * 4))
actual_query_bytes="$(stat -c %s "$QUERIES")"
if [[ "$actual_query_bytes" -ne "$expected_query_bytes" ]]; then
    printf 'Query file has %s bytes; profile requires %s.\n' \
        "$actual_query_bytes" "$expected_query_bytes" >&2
    exit 1
fi

mkdir -p "$RUN_DIR"
transport_args=(--port "$PORT")
if [[ -n "$UNIX_SOCKET" ]]; then
    transport_args=(--unix-socket "$UNIX_SOCKET")
fi
if [[ "$BACKEND" == "torch-u50" ]]; then
    nvidia-smi --query-gpu=index,name,memory.used,memory.free,utilization.gpu \
        --format=csv,noheader,nounits \
        | awk -F', ' -v a="$P0_GPU" -v b="$P1_GPU" '$1==a || $1==b' \
        >"$RUN_DIR/gpu-before.csv"
    (
        while true; do
            timestamp="$(date +%s%3N)"
            nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu \
                --format=csv,noheader,nounits \
                | awk -F', ' -v t="$timestamp" -v a="$P0_GPU" -v b="$P1_GPU" \
                    '$1==a || $1==b {print t ", " $0}'
            sleep 1
        done
    ) >"$RUN_DIR/gpu-during.csv" &
    monitor_pid=$!
else
    monitor_pid=""
fi
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
pin "$P0_CPUSET" "$PYTHON" "$SCRIPT_DIR/ss_linear_scan_party.py" \
    --party 0 "${transport_args[@]}" \
    --backend "$BACKEND" --gpu "$P0_GPU" \
    --table-shares "$TABLE_SHARES" --triples "$TRIPLES" \
    --output "$RUN_DIR/P0-share.u64" --metadata "$RUN_DIR/P0-metadata.json" \
    >"$RUN_DIR/P0.log" 2>&1 &
p0_pid=$!
cleanup() {
    kill "$p0_pid" 2>/dev/null || true
    [[ -z "$monitor_pid" ]] || kill "$monitor_pid" 2>/dev/null || true
    [[ -z "$UNIX_SOCKET" ]] || rm -f -- "$UNIX_SOCKET"
}
trap cleanup EXIT INT TERM

pin "$P1_CPUSET" "$PYTHON" "$SCRIPT_DIR/ss_linear_scan_party.py" \
    --party 1 "${transport_args[@]}" \
    --backend "$BACKEND" --gpu "$P1_GPU" \
    --table-shares "$TABLE_SHARES" --triples "$TRIPLES" --queries "$QUERIES" \
    --output "$RUN_DIR/P1-share.u64" --metadata "$RUN_DIR/P1-metadata.json" \
    >"$RUN_DIR/P1.log" 2>&1
wait "$p0_pid"
trap - EXIT INT TERM
if [[ -n "$monitor_pid" ]]; then
    kill "$monitor_pid" 2>/dev/null || true
    wait "$monitor_pid" 2>/dev/null || true
    nvidia-smi --query-gpu=index,name,memory.used,memory.free,utilization.gpu \
        --format=csv,noheader,nounits \
        | awk -F', ' -v a="$P0_GPU" -v b="$P1_GPU" '$1==a || $1==b' \
        >"$RUN_DIR/gpu-after.csv"
fi
[[ -z "$UNIX_SOCKET" ]] || rm -f -- "$UNIX_SOCKET"
wall_ms=$(($(date +%s%3N) - start_ms))

if [[ -f "$CLEAR_TABLE" ]]; then
    "$PYTHON" "$SCRIPT_DIR/verify_ss_linear_scan.py" \
        --table "$CLEAR_TABLE" --queries "$QUERIES" \
        --share0 "$RUN_DIR/P0-share.u64" --share1 "$RUN_DIR/P1-share.u64" \
        --logical-n "$logical_n" --queries-count "$queries_count" \
        --output-dim "$output_dim" --ring-bits "$ring_bits" \
        --json-output "$RUN_DIR/verification.json"
else
    printf 'Skipping debug reconstruction: clear table %s is absent.\n' "$CLEAR_TABLE"
fi

"$PYTHON" "$SCRIPT_DIR/summarize_ss_linear_scan_run.py" \
    --run-dir "$RUN_DIR" --table-shares "$TABLE_SHARES" --triples "$TRIPLES" \
    --queries "$QUERIES" --wall-ms "$wall_ms" >"$RUN_DIR/summary.log"

printf 'SS-LinearScan completed in %s ms: %s\n' "$wall_ms" "$RUN_DIR"
if [[ "$queries_count" -eq 512 && "$output_dim" -eq 768 ]]; then
    printf 'Next: %s/run_sigma_input_bridge.sh %s\n' "$SCRIPT_DIR" "$RUN_DIR"
fi
