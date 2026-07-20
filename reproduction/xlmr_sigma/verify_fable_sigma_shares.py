#!/usr/bin/env python3
"""Verify FABLE GC outputs converted to 50-bit SIGMA arithmetic shares."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


VOCAB_SIZE = 250002
DIMENSIONS = 768
BATCH_SIZE = 512
CHUNK_DIMENSIONS = 32
SIGMA_BITWIDTH = 50


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--table", type=Path, required=True)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--share0", type=Path, required=True)
    parser.add_argument("--share1", type=Path, required=True)
    parser.add_argument("--chunk-start", type=int, default=0)
    parser.add_argument("--chunk-count", type=int, default=1)
    parser.add_argument("--json-output", type=Path)
    return parser.parse_args()


def exact_array(path: Path, dtype: np.dtype, shape: tuple[int, ...]) -> np.ndarray:
    expected_bytes = int(np.prod(shape)) * np.dtype(dtype).itemsize
    if path.stat().st_size != expected_bytes:
        raise ValueError(
            f"{path}: expected {expected_bytes} bytes, got {path.stat().st_size}"
        )
    return np.memmap(path, mode="r", dtype=dtype, shape=shape)


def main() -> None:
    args = parse_args()
    if args.chunk_start < 0 or args.chunk_count <= 0:
        raise ValueError("invalid chunk range")
    dim_start = args.chunk_start * CHUNK_DIMENSIONS
    dim_end = dim_start + args.chunk_count * CHUNK_DIMENSIONS
    if dim_end > DIMENSIONS:
        raise ValueError("chunk range exceeds 768 dimensions")

    table = exact_array(args.table, "<i2", (VOCAB_SIZE, DIMENSIONS))
    queries = exact_array(args.queries, "<u4", (BATCH_SIZE,))
    share0 = exact_array(args.share0, "<u8", (BATCH_SIZE, DIMENSIONS))
    share1 = exact_array(args.share1, "<u8", (BATCH_SIZE, DIMENSIONS))
    if int(queries.max()) >= VOCAB_SIZE:
        raise ValueError("query outside vocabulary")

    ring_mask = np.uint64((1 << SIGMA_BITWIDTH) - 1)
    reconstructed = (share0[:, dim_start:dim_end] + share1[:, dim_start:dim_end]) & ring_mask
    expected_signed = np.asarray(table[queries, dim_start:dim_end], dtype=np.int64)
    expected = expected_signed.view(np.uint64) & ring_mask
    mismatch = reconstructed != expected
    mismatch_count = int(mismatch.sum())
    result = {
        "status": "pass" if mismatch_count == 0 else "fail",
        "queries": BATCH_SIZE,
        "dimensions": [dim_start, dim_end],
        "values_checked": int(expected.size),
        "mismatches": mismatch_count,
        "sigma_bitwidth": SIGMA_BITWIDTH,
        "fixed_min": int(expected_signed.min()),
        "fixed_max": int(expected_signed.max()),
    }
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        with args.json_output.open("w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2)
            handle.write("\n")
    print(json.dumps(result, indent=2))
    if mismatch_count:
        first = np.argwhere(mismatch)[0]
        raise AssertionError(
            f"first mismatch at query={first[0]}, dimension={dim_start + first[1]}"
        )


if __name__ == "__main__":
    main()
