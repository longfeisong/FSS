#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
FABLE_DIR="$ROOT_DIR/src/FABLE"

apply_once() {
    local repo="$1"
    local patch="$2"

    if git -C "$repo" apply --reverse --check "$patch" >/dev/null 2>&1; then
        printf 'Already applied: %s\n' "$(basename "$patch")"
    elif git -C "$repo" apply --check "$patch"; then
        git -C "$repo" apply "$patch"
        printf 'Applied: %s\n' "$(basename "$patch")"
    else
        printf 'Patch does not apply cleanly: %s\n' "$patch" >&2
        exit 1
    fi
}

git -C "$ROOT_DIR" submodule update --init --recursive src/FABLE
apply_once "$FABLE_DIR" "$SCRIPT_DIR/patches/fable-gelu.patch"
apply_once "$FABLE_DIR/extern/BatchPIR" "$SCRIPT_DIR/patches/batchpir-small-table.patch"
