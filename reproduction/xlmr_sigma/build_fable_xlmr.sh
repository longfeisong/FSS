#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
FABLE_DIR="$ROOT_DIR/src/FABLE"
LIBOTE_PREFIX="$ROOT_DIR/third_party/libOTe/install"
ENV_NAME="${FABLE_CONDA_ENV:-sigma-fable}"
JOBS="${FABLE_BUILD_JOBS:-8}"
PATCH="$SCRIPT_DIR/patches/fable-xlmr-app.patch"

"$ROOT_DIR/reproduction/fable/apply_patches.sh"

if git -C "$FABLE_DIR" apply --reverse --check "$PATCH" >/dev/null 2>&1; then
    printf 'Already applied: %s\n' "$(basename "$PATCH")"
elif git -C "$FABLE_DIR" apply --check "$PATCH"; then
    git -C "$FABLE_DIR" apply "$PATCH"
    printf 'Applied: %s\n' "$(basename "$PATCH")"
else
    printf 'Patch does not apply cleanly: %s\n' "$PATCH" >&2
    exit 1
fi

CONDA_PREFIX_PATH="$(conda run -n "$ENV_NAME" bash -c 'printf %s "$CONDA_PREFIX"')"
CMAKE_PREFIX_PATH="$CONDA_PREFIX_PATH;$LIBOTE_PREFIX"

conda run -n "$ENV_NAME" cmake -S "$FABLE_DIR" -B "$FABLE_DIR/build-xlmr" \
    -DLUT_INPUT_SIZE=20 \
    -DLUT_OUTPUT_SIZE=512 \
    -DLUT_MAX_LOG_SIZE=20 \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_PREFIX_PATH="$CMAKE_PREFIX_PATH" \
    -DlibOTe_DIR="$LIBOTE_PREFIX/lib/cmake/libOTe" \
    -DcryptoTools_DIR="$LIBOTE_PREFIX/lib/cmake/cryptoTools" \
    -Dcoproto_DIR="$LIBOTE_PREFIX/lib/cmake/coproto"

conda run -n "$ENV_NAME" cmake --build "$FABLE_DIR/build-xlmr" \
    --target xlmr_embedding --parallel "$JOBS"

printf 'Built %s\n' "$FABLE_DIR/build-xlmr/bin/xlmr_embedding"
