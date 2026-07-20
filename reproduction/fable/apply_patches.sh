#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
FABLE_DIR="$ROOT_DIR/src/FABLE"

apply_once() {
    local repo="$1"
    local patch="$2"
    local whitespace_mode="${3:-strict}"
    local apply_args=()
    if [[ "$whitespace_mode" == "ignore-space-change" ]]; then
        apply_args+=(--ignore-space-change)
    fi

    if git -C "$repo" apply "${apply_args[@]}" --reverse --check "$patch" >/dev/null 2>&1; then
        printf 'Already applied: %s\n' "$(basename "$patch")"
    elif git -C "$repo" apply "${apply_args[@]}" --check "$patch"; then
        git -C "$repo" apply "${apply_args[@]}" "$patch"
        printf 'Applied: %s\n' "$(basename "$patch")"
    else
        printf 'Patch does not apply cleanly: %s\n' "$patch" >&2
        exit 1
    fi
}

git -C "$ROOT_DIR" submodule update --init --recursive src/FABLE
apply_once "$FABLE_DIR" "$SCRIPT_DIR/patches/fable-gelu.patch"
apply_once "$FABLE_DIR/extern/BatchPIR" "$SCRIPT_DIR/patches/batchpir-small-table.patch"
apply_once "$FABLE_DIR/extern/EzPC" "$SCRIPT_DIR/patches/ezpc-build-compat.patch" ignore-space-change
apply_once "$FABLE_DIR/extern/EzPC/SCI/extern/SEAL" "$SCRIPT_DIR/patches/seal-locks-compat.patch"
