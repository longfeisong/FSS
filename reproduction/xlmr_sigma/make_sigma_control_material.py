#!/usr/bin/env python3
"""Create ignored zero-mask material for a SIGMA correctness oracle.

This is not a comparison baseline.  It runs the same SIGMA fixed-point graph
without masking so its output can be compared bit-for-bit with the secure run.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--masked-input", type=Path, required=True)
    parser.add_argument("--input-mask", type=Path, required=True)
    parser.add_argument("--masked-weights", type=Path, required=True)
    parser.add_argument("--weight-mask", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bitwidth", type=int, default=50)
    return parser.parse_args()


def reconstruct(masked_path: Path, mask_path: Path, ring_mask: np.uint64) -> np.ndarray:
    masked = np.fromfile(masked_path, dtype="<u8")
    mask = np.fromfile(mask_path, dtype="<u8")
    if masked.shape != mask.shape:
        raise ValueError(f"shape mismatch: {masked_path} and {mask_path}")
    return (masked - mask) & ring_mask


def main() -> None:
    args = parse_args()
    ring_mask = np.uint64((1 << args.bitwidth) - 1)
    clear_input = reconstruct(args.masked_input, args.input_mask, ring_mask)
    clear_weights = reconstruct(args.masked_weights, args.weight_mask, ring_mask)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    clear_input.tofile(args.output_dir / "clear-input.u64")
    clear_weights.tofile(args.output_dir / "clear-weights.u64")
    np.zeros_like(clear_input).tofile(args.output_dir / "zero-input-mask.u64")
    np.zeros_like(clear_weights).tofile(args.output_dir / "zero-weight-mask.u64")
    result = {
        "purpose": "correctness oracle only; not a benchmark baseline",
        "input_values": int(clear_input.size),
        "weight_values": int(clear_weights.size),
        "bitwidth": args.bitwidth,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
