#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
FABLE_DIR="$ROOT_DIR/src/FABLE"
LIBOTE_DIR="$ROOT_DIR/third_party/libOTe"
LIBOTE_PREFIX="$LIBOTE_DIR/install"
ENV_NAME="${FABLE_CONDA_ENV:-sigma-fable}"
JOBS="${FABLE_BUILD_JOBS:-8}"
LIBOTE_REVISION="90dbd858cb47f1381fd9186f8436b14edc0f0dbe"

"$SCRIPT_DIR/apply_patches.sh"

if [[ ! -f "$LIBOTE_PREFIX/lib/cmake/libOTe/libOTeConfig.cmake" ]]; then
    if [[ ! -d "$LIBOTE_DIR/.git" ]]; then
        git clone --recursive https://github.com/osu-crypto/libOTe.git "$LIBOTE_DIR"
    fi
    git -C "$LIBOTE_DIR" checkout "$LIBOTE_REVISION"
    git -C "$LIBOTE_DIR" submodule update --init --recursive
    conda run -n "$ENV_NAME" python "$LIBOTE_DIR/build.py" \
        --boost --sodium --all --par="$JOBS"
    conda run -n "$ENV_NAME" python "$LIBOTE_DIR/build.py" \
        --install="$LIBOTE_PREFIX" --par="$JOBS"
fi

CONDA_PREFIX_PATH="$(conda run -n "$ENV_NAME" bash -c 'printf %s "$CONDA_PREFIX"')"
CMAKE_PREFIX_PATH="$CONDA_PREFIX_PATH;$LIBOTE_PREFIX"

conda run -n "$ENV_NAME" cmake -S "$FABLE_DIR" -B "$FABLE_DIR/build-gelu" \
    -DLUT_INPUT_SIZE=16 \
    -DLUT_OUTPUT_SIZE=37 \
    -DLUT_MAX_LOG_SIZE=16 \
    -DCMAKE_PREFIX_PATH="$CMAKE_PREFIX_PATH" \
    -DlibOTe_DIR="$LIBOTE_PREFIX/lib/cmake/libOTe" \
    -DcryptoTools_DIR="$LIBOTE_PREFIX/lib/cmake/cryptoTools" \
    -Dcoproto_DIR="$LIBOTE_PREFIX/lib/cmake/coproto"

conda run -n "$ENV_NAME" cmake --build "$FABLE_DIR/build-gelu" \
    --target fable --parallel "$JOBS"

printf 'Built %s\n' "$FABLE_DIR/build-gelu/bin/fable"
