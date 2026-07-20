#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
LOOKUP_RUN_DIR="${1:?usage: run_sigma_xlmr.sh <verified-lookup-run-dir> [layers] [sequence-index]}"
LAYERS="${2:-1}"
SEQUENCE_INDEX="${3:-0}"
P0_GPU="${SIGMA_P0_GPU:-7}"
P1_GPU="${SIGMA_P1_GPU:-2}"
THREADS="${SIGMA_CPU_THREADS:-4}"
BIN="$ROOT_DIR/third_party/EzPC/GPU-MPC/experiments/sigma/sigma_xlmr"
WEIGHT_DIR="$SCRIPT_DIR/artifacts/sigma-weights-${LAYERS}layer"
OUT_DIR="$LOOKUP_RUN_DIR/sigma-${LAYERS}layer-seq${SEQUENCE_INDEX}"
KEY_PREFIX="$OUT_DIR/keys/xlmr${LAYERS}-seq${SEQUENCE_INDEX}"
ENV_NAME="${SIGMA_CONDA_ENV:-sigma-fable}"
CUDA_ROOT="${CUDA_HOME:-/usr/local/cuda}"
if [[ "${CONDA_DEFAULT_ENV:-}" == "$ENV_NAME" && -n "${CONDA_PREFIX:-}" ]]; then
    SIGMA_CONDA_PREFIX="$CONDA_PREFIX"
elif command -v conda >/dev/null 2>&1; then
    SIGMA_CONDA_PREFIX="$(conda run -n "$ENV_NAME" bash -c 'printf %s "$CONDA_PREFIX"')"
else
    printf 'Activate Conda environment %s or make conda available in PATH.\n' "$ENV_NAME" >&2
    exit 1
fi
LIBS="$SIGMA_CONDA_PREFIX/lib:$CUDA_ROOT/lib64:$ROOT_DIR/third_party/EzPC/GPU-MPC/ext/sytorch/build:$ROOT_DIR/third_party/EzPC/GPU-MPC/ext/sytorch/build/ext/cryptoTools:$ROOT_DIR/third_party/EzPC/GPU-MPC/ext/sytorch/build/ext/llama:$ROOT_DIR/third_party/EzPC/GPU-MPC/ext/sytorch/build/ext/bitpack"

[[ -x "$BIN" ]] || { printf 'Run %s first.\n' "$SCRIPT_DIR/build_sigma_xlmr.sh" >&2; exit 1; }
[[ -f "$LOOKUP_RUN_DIR/verification.json" ]] || { printf 'Lookup run has not been verified.\n' >&2; exit 1; }
if [[ "$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["dimensions"][1])' "$LOOKUP_RUN_DIR/verification.json")" != 768 ]]; then
    printf 'Refusing partial lookup input: a SIGMA recorded run requires all 768 dimensions.\n' >&2
    exit 1
fi
if [[ ! -f "$LOOKUP_RUN_DIR/sigma-input/verification.json" ]]; then
    "$SCRIPT_DIR/run_sigma_input_bridge.sh" "$LOOKUP_RUN_DIR"
fi
LOOKUP_BACKEND="$(python -c 'import json,sys; from pathlib import Path; p=Path(sys.argv[1]); m=p/"metadata.json"; value=json.load(open(m)).get("scheme") if m.exists() else None; print(value or ("fable" if "fable" in p.name.lower() else "unknown-lookup"))' "$LOOKUP_RUN_DIR")"
for file in "$WEIGHT_DIR/dealer-weight-mask.u64" "$WEIGHT_DIR/evaluator-masked-weights.u64"; do
    [[ -f "$file" ]] || { printf 'Missing %s; export and prepare %s-layer weights first.\n' "$file" "$LAYERS" >&2; exit 1; }
done
if ss -ltn | awk '{print $4}' | rg -q ':42002$'; then
    printf 'SIGMA port 42002 is already in use.\n' >&2
    exit 1
fi

mkdir -p "$OUT_DIR/keys"
gpu_state="$(nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu --format=csv,noheader,nounits | awk -F', ' -v a="$P0_GPU" -v b="$P1_GPU" '$1==a || $1==b')"
printf '%s\n' "$gpu_state" >"$OUT_DIR/gpu-before.csv"
export SIGMA_GPU_POOL_GB=0
export SIGMA_XLMR_KEY_GB="${SIGMA_XLMR_KEY_GB:-$((LAYERS == 1 ? 8 : 6 * LAYERS))}"
export LD_LIBRARY_PATH="$LIBS:${LD_LIBRARY_PATH:-}"

dealer_input="$LOOKUP_RUN_DIR/sigma-input/dealer/dealer-input-mask.u64"
dealer_weights="$WEIGHT_DIR/dealer-weight-mask.u64"
for party in 0 1; do
    CUDA_VISIBLE_DEVICES="$P0_GPU" "$BIN" dealer "$LAYERS" "$party" \
        "$SEQUENCE_INDEX" "$dealer_input" "$dealer_weights" "$KEY_PREFIX" \
        >"$OUT_DIR/dealer-P${party}.log" 2>&1
done

masked_weights="$WEIGHT_DIR/evaluator-masked-weights.u64"
(
    CUDA_VISIBLE_DEVICES="$P0_GPU" "$BIN" online "$LAYERS" 0 \
        "$SEQUENCE_INDEX" "$LOOKUP_RUN_DIR/sigma-input/P0-masked-input.u64" \
        "$masked_weights" "$KEY_PREFIX" 127.0.0.1 "$THREADS" \
        "$OUT_DIR/P0-output.u64" >"$OUT_DIR/online-P0.log" 2>&1
) &
p0_pid=$!
trap 'kill "$p0_pid" 2>/dev/null || true' EXIT INT TERM
sleep 1
CUDA_VISIBLE_DEVICES="$P1_GPU" "$BIN" online "$LAYERS" 1 \
    "$SEQUENCE_INDEX" "$LOOKUP_RUN_DIR/sigma-input/P1-masked-input.u64" \
    "$masked_weights" "$KEY_PREFIX" 127.0.0.1 "$THREADS" \
    "$OUT_DIR/P1-output.u64" >"$OUT_DIR/online-P1.log" 2>&1
wait "$p0_pid"
trap - EXIT INT TERM

cmp "$OUT_DIR/P0-output.u64" "$OUT_DIR/P1-output.u64"
output_sha="$(sha256sum "$OUT_DIR/P0-output.u64" | awk '{print $1}')"
elapsed_p0="$(rg -o 'elapsed_us=[0-9]+' "$OUT_DIR/online-P0.log" | cut -d= -f2)"
elapsed_p1="$(rg -o 'elapsed_us=[0-9]+' "$OUT_DIR/online-P1.log" | cut -d= -f2)"
communication="$(rg -o 'communication_bytes=[0-9]+' "$OUT_DIR/online-P0.log" | cut -d= -f2)"
key_bytes="$(stat -c %s "${KEY_PREFIX}_0.dat")"
timing_reliable=true
if awk -F', ' '$4 > 30 {found=1} END {exit !found}' "$OUT_DIR/gpu-before.csv"; then
    timing_reliable=false
fi
cat >"$OUT_DIR/result.json" <<EOF
{
  "status": "pass",
  "scheme": "$LOOKUP_BACKEND -> arithmetic shares -> SIGMA masked input -> XLM-R encoder",
  "lookup_backend": "$LOOKUP_BACKEND",
  "layers": $LAYERS,
  "sequence_index": $SEQUENCE_INDEX,
  "lookup_dimensions": 768,
  "p0_gpu": $P0_GPU,
  "p1_gpu": $P1_GPU,
  "timing_reliable": $timing_reliable,
  "online_elapsed_us_p0": $elapsed_p0,
  "online_elapsed_us_p1": $elapsed_p1,
  "online_communication_bytes_per_party": $communication,
  "offline_key_bytes_per_party": $key_bytes,
  "party_outputs_identical": true,
  "output_sha256": "$output_sha"
}
EOF
cat "$OUT_DIR/result.json"
