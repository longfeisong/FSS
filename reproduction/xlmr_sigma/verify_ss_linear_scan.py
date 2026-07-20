#!/usr/bin/env python3
"""Debug-only reconstruction check for SS-LinearScan output shares."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from ss_linear_scan_common import encode_signed, ring_mask


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--table", type=Path, required=True)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--share0", type=Path, required=True)
    parser.add_argument("--share1", type=Path, required=True)
    parser.add_argument("--logical-n", type=int, required=True)
    parser.add_argument("--queries-count", type=int, required=True)
    parser.add_argument("--output-dim", type=int, required=True)
    parser.add_argument("--ring-bits", type=int, default=50)
    parser.add_argument("--json-output", type=Path)
    return parser.parse_args()


def exact(path: Path, dtype: str, shape: tuple[int, ...]) -> np.memmap:
    expected = int(np.prod(shape, dtype=np.int64)) * np.dtype(dtype).itemsize
    if path.stat().st_size != expected:
        raise ValueError(f"{path}: expected {expected} bytes, got {path.stat().st_size}")
    return np.memmap(path, mode="r", dtype=dtype, shape=shape)


def main() -> None:
    args = parse_args()
    table = exact(args.table, "<i2", (args.logical_n, args.output_dim))
    queries = exact(args.queries, "<u4", (args.queries_count,))
    share0 = exact(args.share0, "<u8", (args.queries_count, args.output_dim))
    share1 = exact(args.share1, "<u8", (args.queries_count, args.output_dim))
    if int(queries.max()) >= args.logical_n:
        raise ValueError("query is outside the logical table")
    reconstructed = (share0 + share1) & ring_mask(args.ring_bits)
    expected_signed = np.asarray(table[queries], dtype=np.int64)
    expected = encode_signed(expected_signed, args.ring_bits)
    mismatches = reconstructed != expected
    mismatch_count = int(mismatches.sum())
    digest = hashlib.sha256(np.ascontiguousarray(reconstructed).tobytes()).hexdigest()
    result = {
        "status": "pass" if mismatch_count == 0 else "fail",
        "scheme": "ss-linear-scan-debug-reconstruction",
        "values_checked": int(expected.size),
        "mismatches": mismatch_count,
        "shape": [args.queries_count, args.output_dim],
        "dimensions": [0, args.output_dim],
        "ring_bits": args.ring_bits,
        "fixed_scale": 12,
        "reconstructed_sha256": digest,
        "fixed_min": int(expected_signed.min()),
        "fixed_max": int(expected_signed.max()),
        "warning": "debug correctness tool; never run between lookup and SIGMA in production",
    }
    if mismatch_count:
        first = np.argwhere(mismatches)[0]
        result["first_mismatch"] = {
            "query": int(first[0]),
            "dimension": int(first[1]),
            "actual": int(reconstructed[tuple(first)]),
            "expected": int(expected[tuple(first)]),
        }
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        with args.json_output.open("w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2)
            handle.write("\n")
    print(json.dumps(result, indent=2))
    if mismatch_count:
        raise AssertionError(f"SS-LinearScan has {mismatch_count} mismatches")


if __name__ == "__main__":
    main()
