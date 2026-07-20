#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
SOURCE="$ROOT_DIR/src/GPU-MPC"
BUILD_TREE="$ROOT_DIR/third_party/EzPC/GPU-MPC"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.1}"
GPU_ARCH="${GPU_ARCH:-80}"
OUTPUT="$BUILD_TREE/experiments/sigma/sigma_xlmr"

if [[ ! -f "$BUILD_TREE/ext/sytorch/build/libsytorch.a" ]]; then
    printf 'Missing SIGMA dependencies under %s; build the official artifact first.\n' "$BUILD_TREE" >&2
    exit 1
fi

"$CUDA_HOME/bin/nvcc" -O3 \
    -diag-suppress=20012,815,611 \
    -gencode "arch=compute_${GPU_ARCH},code=[sm_${GPU_ARCH},compute_${GPU_ARCH}]" \
    -std=c++17 -m64 \
    -Xcompiler="-O3,-w,-std=c++17,-fpermissive,-fpic,-pthread,-fopenmp,-march=native" \
    -I "$BUILD_TREE/ext/cutlass/include" \
    -I "$BUILD_TREE/ext/cutlass/tools/util/include" \
    -I "$SOURCE/ext/sytorch/include" \
    -I "$BUILD_TREE/ext/sytorch/include" \
    -I "$BUILD_TREE/ext/sytorch/ext/llama/include" \
    -I "$BUILD_TREE/ext/sytorch/ext/cryptoTools" \
    -I "$SOURCE" \
    -I "$BUILD_TREE/.deps/include" \
    -L"$BUILD_TREE/ext/cutlass/build/tools/library" \
    -L"$BUILD_TREE/ext/sytorch/build" \
    -L"$BUILD_TREE/ext/sytorch/build/ext/cryptoTools" \
    -L"$BUILD_TREE/ext/sytorch/build/ext/llama" \
    -L"$BUILD_TREE/ext/sytorch/build/ext/bitpack" \
    -L"$BUILD_TREE/ext/sytorch/build/lib" \
    "$SCRIPT_DIR/sigma_xlmr.cu" \
    "$SOURCE/utils/gpu_mem.cu" \
    "$SOURCE/utils/gpu_file_utils.cpp" \
    "$SOURCE/utils/sigma_comms.cpp" \
    -lsytorch -lcryptoTools -lLLAMA -lbitpack -lcuda -lcudart -lcurand \
    -o "$OUTPUT"

printf 'Built %s\n' "$OUTPUT"
