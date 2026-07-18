#!/usr/bin/env python3
"""Verify the FABLE arithmetic-share to SIGMA masked-input conversion."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--share0", type=Path, required=True)
    parser.add_argument("--share1", type=Path, required=True)
    parser.add_argument("--dealer-mask", type=Path, required=True)
    parser.add_argument("--masked0", type=Path, required=True)
    parser.add_argument("--masked1", type=Path, required=True)
    parser.add_argument("--bitwidth", type=int, default=50)
    parser.add_argument("--count", type=int, default=512 * 768)
    parser.add_argument("--json-output", type=Path)
    return parser.parse_args()


def load(path: Path, count: int) -> np.ndarray:
    values = np.fromfile(path, dtype="<u8")
    if values.size != count:
        raise ValueError(f"{path}: expected {count} values, got {values.size}")
    return values


def main() -> None:
    args = parse_args()
    mask = np.uint64((1 << args.bitwidth) - 1)
    share0 = load(args.share0, args.count)
    share1 = load(args.share1, args.count)
    dealer_mask = load(args.dealer_mask, args.count)
    masked0 = load(args.masked0, args.count)
    masked1 = load(args.masked1, args.count)
    same_masked_input = bool(np.array_equal(masked0, masked1))
    embedding = (share0 + share1) & mask
    reconstructed = (masked0 - dealer_mask) & mask
    mismatch_count = int(np.count_nonzero(embedding != reconstructed))
    result = {
        "status": "pass" if same_masked_input and mismatch_count == 0 else "fail",
        "same_public_masked_input": same_masked_input,
        "values_checked": args.count,
        "mismatches": mismatch_count,
        "bitwidth": args.bitwidth,
    }
    if args.json_output:
        args.json_output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    if result["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
