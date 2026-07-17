#!/usr/bin/env bash
set -euo pipefail

# Run the two SIGMA parties on two currently available local GPUs.
# GPUs 0 and 1 are reserved until 2026-07-29, so the default allowlist is 2-7.

MODEL="${1:-bert-tiny}"
SEQ_LEN="${2:-128}"
CPU_THREADS="${3:-4}"
POOL_GB="${SIGMA_GPU_POOL_GB:-0}"
ALLOWED_GPUS="${SIGMA_ALLOWED_GPUS:-2,3,4,5,6,7}"
MAX_UTIL="${SIGMA_MAX_GPU_UTIL:-100}"
WARN_UTIL="${SIGMA_WARN_GPU_UTIL:-80}"
MIN_FREE_MIB="${SIGMA_MIN_FREE_MIB:-8192}"
ALLOW_SAME_GPU="${SIGMA_ALLOW_SAME_GPU:-0}"
AUTO_CLEAN="${SIGMA_AUTO_CLEAN:-1}"
RECORD_RESULTS="${SIGMA_RECORD_RESULTS:-1}"
SCHEME="${SIGMA_SCHEME:-SIGMA}"
VARIANT="${SIGMA_VARIANT:-baseline-zero}"
NETWORK="${SIGMA_NETWORK:-local}"
CORRECTNESS="${SIGMA_CORRECTNESS:-zero-smoke-test}"
RUN_NOTES="${SIGMA_RUN_NOTES:-}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
RESULTS_DIR="${SIGMA_RESULTS_DIR:-$SCRIPT_DIR/results}"
SIGMA_BIN="$SCRIPT_DIR/sigma"
CUDA_LIB="/usr/local/cuda-12.1/lib64"
CONDA_LIB="/home/slf/anaconda3/envs/sigma-fable/lib"

if [[ ! -x "$SIGMA_BIN" ]]; then
    echo "SIGMA executable not found: $SIGMA_BIN" >&2
    exit 1
fi

cleanup_previous_runs() {
    local sigma_real script_real pid exe cwd arg candidate
    local -a old_launchers=() old_sigma=()
    sigma_real="$(readlink -f "$SIGMA_BIN")"
    script_real="$(readlink -f "${BASH_SOURCE[0]}")"

    # Stop older copies of this launcher first. SIGTERM runs their cleanup
    # trap; SIGCONT is needed when an older run was suspended with Ctrl+Z.
    while read -r pid; do
        [[ -z "$pid" || "$pid" == "$$" ]] && continue
        [[ -r "/proc/$pid/cmdline" ]] || continue
        cwd="$(readlink -f "/proc/$pid/cwd" 2>/dev/null || true)"
        while IFS= read -r -d '' arg; do
            [[ "$arg" == *"run_local_auto.sh" ]] || continue
            if [[ "$arg" == /* ]]; then
                candidate="$(readlink -f "$arg" 2>/dev/null || true)"
            else
                candidate="$(readlink -f "$cwd/$arg" 2>/dev/null || true)"
            fi
            if [[ "$candidate" == "$script_real" ]]; then
                old_launchers+=("$pid")
                break
            fi
        done < "/proc/$pid/cmdline"
    done < <(pgrep -u "$UID" -x bash || true)

    if (( ${#old_launchers[@]} > 0 )); then
        echo "Cleaning previous SIGMA launcher(s): ${old_launchers[*]}"
        kill -TERM "${old_launchers[@]}" 2>/dev/null || true
        kill -CONT "${old_launchers[@]}" 2>/dev/null || true
        sleep 0.2
    fi

    # Match by /proc/<pid>/exe so only this workspace's SIGMA binary is
    # terminated. This does not touch unrelated jobs owned by the same user.
    while read -r pid; do
        [[ -z "$pid" ]] && continue
        exe="$(readlink "/proc/$pid/exe" 2>/dev/null || true)"
        exe="${exe% (deleted)}"
        if [[ "$exe" == "$sigma_real" ]]; then
            old_sigma+=("$pid")
        fi
    done < <(pgrep -u "$UID" -x sigma || true)

    if (( ${#old_sigma[@]} > 0 )); then
        echo "Cleaning previous SIGMA process(es): ${old_sigma[*]}"
        kill -TERM "${old_sigma[@]}" 2>/dev/null || true
        kill -CONT "${old_sigma[@]}" 2>/dev/null || true
        sleep 0.5
        for pid in "${old_sigma[@]}"; do
            if kill -0 "$pid" 2>/dev/null; then
                kill -KILL "$pid" 2>/dev/null || true
            fi
        done
    fi
}

if [[ "$AUTO_CLEAN" == "1" ]]; then
    cleanup_previous_runs
fi

is_allowed() {
    local gpu="$1"
    [[ ",$ALLOWED_GPUS," == *",$gpu,"* ]]
}

mapfile -t CANDIDATES < <(
    nvidia-smi \
        --query-gpu=index,memory.free,utilization.gpu,compute_mode \
        --format=csv,noheader,nounits |
    while IFS=',' read -r index free_mib util mode; do
        index="${index//[[:space:]]/}"
        free_mib="${free_mib//[[:space:]]/}"
        util="${util//[[:space:]]/}"
        mode="${mode#${mode%%[![:space:]]*}}"
        if is_allowed "$index" &&
           [[ "$mode" == "Default" ]] &&
           (( free_mib >= MIN_FREE_MIB )) &&
           (( util <= MAX_UTIL )); then
            printf '%s %s %s\n' "$util" "$free_mib" "$index"
        fi
    done |
    sort -k1,1n -k2,2nr |
    awk '{print $3}' |
    head -n 2
)

if (( ${#CANDIDATES[@]} == 1 )) && [[ "$ALLOW_SAME_GPU" == "1" ]]; then
    CANDIDATES+=("${CANDIDATES[0]}")
fi

if (( ${#CANDIDATES[@]} < 2 )); then
    echo "Need two eligible GPUs, found ${#CANDIDATES[@]}." >&2
    echo "Constraints: allowed=$ALLOWED_GPUS, max_util=$MAX_UTIL%, min_free=${MIN_FREE_MIB}MiB" >&2
    nvidia-smi \
        --query-gpu=index,memory.free,utilization.gpu,compute_mode \
        --format=csv,noheader
    exit 2
fi

GPU_P0="${CANDIDATES[0]}"
GPU_P1="${CANDIDATES[1]}"
RUN_ID="$(date +%Y%m%dT%H%M%S)-$$"
RUN_TIMESTAMP="$(date --iso-8601=seconds)"
echo "Selected GPUs: P0=$GPU_P0, P1=$GPU_P1"
echo "Model=$MODEL, seq_len=$SEQ_LEN, pool=${POOL_GB}GiB"

gpu_snapshot() {
    nvidia-smi --id="$1" \
        --query-gpu=name,utilization.gpu,memory.free \
        --format=csv,noheader,nounits
}

IFS=',' read -r GPU_P0_NAME GPU_P0_UTIL GPU_P0_FREE < <(gpu_snapshot "$GPU_P0")
IFS=',' read -r GPU_P1_NAME GPU_P1_UTIL GPU_P1_FREE < <(gpu_snapshot "$GPU_P1")
GPU_P0_NAME="${GPU_P0_NAME#${GPU_P0_NAME%%[![:space:]]*}}"
GPU_P1_NAME="${GPU_P1_NAME#${GPU_P1_NAME%%[![:space:]]*}}"
GPU_P0_UTIL="${GPU_P0_UTIL//[[:space:]]/}"
GPU_P1_UTIL="${GPU_P1_UTIL//[[:space:]]/}"
GPU_P0_FREE="${GPU_P0_FREE//[[:space:]]/}"
GPU_P1_FREE="${GPU_P1_FREE//[[:space:]]/}"
TIMING_VALID=yes

for gpu_util in "$GPU_P0:$GPU_P0_UTIL" "$GPU_P1:$GPU_P1_UTIL"; do
    gpu="${gpu_util%%:*}"
    util="${gpu_util##*:}"
    if (( util >= WARN_UTIL )); then
        echo "Warning: GPU $gpu utilization is ${util}%; timing results will not be reliable." >&2
        TIMING_VALID=no
    fi
done

if [[ "$GPU_P0" == "$GPU_P1" ]]; then
    echo "Warning: P0 and P1 are sharing GPU $GPU_P0; use this only for functional testing." >&2
fi

cleanup() {
    for pid in "${P0_PID:-}" "${P1_PID:-}"; do
        if [[ -n "$pid" ]]; then
            kill "$pid" 2>/dev/null || true
            wait "$pid" 2>/dev/null || true
        fi
    done
}
trap cleanup EXIT INT TERM HUP
trap 'cleanup; exit 148' TSTP

cd "$SCRIPT_DIR"

if ss -ltnH | awk '$4 ~ /:42002$/ { found=1 } END { exit !found }'; then
    echo "SIGMA port 42002 is already in use. Clean up an earlier run first:" >&2
    ss -ltnp | awk 'NR == 1 || $4 ~ /:42002$/' >&2
    exit 3
fi

SIGMA_GPU_POOL_GB="$POOL_GB" \
CUDA_VISIBLE_DEVICES="$GPU_P0" \
LD_LIBRARY_PATH="$CONDA_LIB:$CUDA_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
"$SIGMA_BIN" "$MODEL" "$SEQ_LEN" 0 127.0.0.1 "$CPU_THREADS" &
P0_PID=$!

for _ in {1..100}; do
    if ss -ltnH | awk '$4 ~ /:42002$/ { found=1 } END { exit !found }'; then
        break
    fi
    if ! kill -0 "$P0_PID" 2>/dev/null; then
        wait "$P0_PID"
        exit $?
    fi
    sleep 0.1
done

if ! ss -ltnH | awk '$4 ~ /:42002$/ { found=1 } END { exit !found }'; then
    echo "P0 did not open port 42002 within 10 seconds." >&2
    exit 4
fi

SIGMA_GPU_POOL_GB="$POOL_GB" \
CUDA_VISIBLE_DEVICES="$GPU_P1" \
LD_LIBRARY_PATH="$CONDA_LIB:$CUDA_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
"$SIGMA_BIN" "$MODEL" "$SEQ_LEN" 1 127.0.0.1 "$CPU_THREADS" &
P1_PID=$!

wait "$P1_PID"
wait "$P0_PID"

if [[ "$RECORD_RESULTS" == "1" ]]; then
    python3 "$SCRIPT_DIR/record_result.py" \
        --scheme "$SCHEME" \
        --variant "$VARIANT" \
        --model "$MODEL" \
        --seq-len "$SEQ_LEN" \
        --run-id "$RUN_ID" \
        --timestamp "$RUN_TIMESTAMP" \
        --network "$NETWORK" \
        --gpu-p0 "$GPU_P0" \
        --gpu-p0-name "$GPU_P0_NAME" \
        --gpu-p0-util "$GPU_P0_UTIL" \
        --gpu-p0-free "$GPU_P0_FREE" \
        --gpu-p1 "$GPU_P1" \
        --gpu-p1-name "$GPU_P1_NAME" \
        --gpu-p1-util "$GPU_P1_UTIL" \
        --gpu-p1-free "$GPU_P1_FREE" \
        --cpu-threads "$CPU_THREADS" \
        --pool-gib "$POOL_GB" \
        --timing-valid "$TIMING_VALID" \
        --correctness "$CORRECTNESS" \
        --notes "$RUN_NOTES" \
        --results-dir "$RESULTS_DIR"
fi

trap - EXIT INT TERM HUP TSTP
