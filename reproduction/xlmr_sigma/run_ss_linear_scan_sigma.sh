#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
TABLE="${XLMR_FABLE_TABLE:-$SCRIPT_DIR/artifacts/fable-table/xlmr_word_embedding_scale12.i16}"
QUERIES="${XLMR_TOKEN_IDS:-$SCRIPT_DIR/artifacts/plaintext-bridge-smoke/token_ids.u32}"
TABLE_SHARES="${SS_LINEAR_SCAN_TABLE_SHARES:-$SCRIPT_DIR/artifacts/ss-linear-scan/xlmr-table-b4096}"
DEALER_GPU="${SS_LINEAR_SCAN_DEALER_GPU:-0}"
P0_GPU="${SS_LINEAR_SCAN_P0_GPU:-0}"
P1_GPU="${SS_LINEAR_SCAN_P1_GPU:-4}"
LAYERS="${1:-1}"
RUN_ID="$(date +%Y%m%dT%H%M%S)-ss-linear-scan-sigma"
TRIPLES="$SCRIPT_DIR/artifacts/ss-linear-scan/triples/$RUN_ID"
RUN_DIR="$SCRIPT_DIR/artifacts/ss-linear-scan/runs/$RUN_ID"

for file in "$TABLE" "$QUERIES"; do
    [[ -f "$file" ]] || { printf 'Missing %s\n' "$file" >&2; exit 1; }
done
if [[ ! -f "$TABLE_SHARES/manifest.json" ]]; then
    python3 "$SCRIPT_DIR/prepare_ss_table_shares.py" \
        --table "$TABLE" --output-dir "$TABLE_SHARES" \
        --logical-n 250002 --output-dim 768 --block-size 4096
fi

python3 "$SCRIPT_DIR/generate_ss_beaver_triples.py" \
    --backend torch-u50 --gpu "$DEALER_GPU" \
    --output-dir "$TRIPLES" --run-id "$RUN_ID" \
    --logical-n 250002 --queries 512 --output-dim 768 --block-size 4096

SS_LINEAR_SCAN_BACKEND=torch-u50 \
SS_LINEAR_SCAN_P0_GPU="$P0_GPU" SS_LINEAR_SCAN_P1_GPU="$P1_GPU" \
XLMR_TOKEN_IDS="$QUERIES" XLMR_FABLE_TABLE="$TABLE" \
    "$SCRIPT_DIR/run_ss_linear_scan.sh" "$TABLE_SHARES" "$TRIPLES" "$RUN_DIR"

"$SCRIPT_DIR/run_sigma_input_bridge.sh" "$RUN_DIR"
for sequence in 0 1 2 3; do
    SIGMA_P0_GPU="$P1_GPU" SIGMA_P1_GPU="$P0_GPU" \
        "$SCRIPT_DIR/run_sigma_xlmr.sh" "$RUN_DIR" "$LAYERS" "$sequence"
done

record_args=(
    --run-dir "$RUN_DIR" --layers "$LAYERS"
    --output "$SCRIPT_DIR/results/ss_linear_scan_sigma_full.json"
)
reference="$SCRIPT_DIR/artifacts/fable-xlmr/20260718T084846-fable-xlmr-c0-n24"
if [[ -f "$reference/sigma-${LAYERS}layer-seq0/result.json" ]]; then
    record_args+=(--reference-fable-run "$reference")
fi
python3 "$SCRIPT_DIR/record_ss_linear_scan_sigma.py" "${record_args[@]}"

printf 'Full SS-LinearScan + SIGMA run completed: %s\n' "$RUN_DIR"
