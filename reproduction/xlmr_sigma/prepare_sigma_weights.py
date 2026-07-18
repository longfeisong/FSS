#!/usr/bin/env python3
"""Quantize and mask an exported XLM-R encoder for SIGMA."""

from __future__ import annotations

import argparse
import hashlib
import json
import secrets
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--float-weights", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--scale", type=int, default=12)
    parser.add_argument("--bitwidth", type=int, default=50)
    parser.add_argument("--chunk-elements", type=int, default=1 << 20)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    float_weights = np.memmap(args.float_weights, dtype="<f4", mode="r")
    if float_weights.size != manifest["parameters"]:
        raise ValueError("float stream and manifest parameter counts differ")
    if args.bitwidth != 50 or args.scale != 12:
        raise ValueError("the selected SIGMA XLM-R profile is scale=12, bitwidth=50")

    scale_by_element = np.empty(float_weights.size, dtype=np.uint8)
    offset = 0
    for tensor in manifest["order"]:
        size = int(np.prod(tensor["shape"], dtype=np.int64))
        tensor_scale = 2 * args.scale if tensor["name"].endswith(".bias") else args.scale
        scale_by_element[offset : offset + size] = tensor_scale
        offset += size
    if offset != float_weights.size:
        raise AssertionError("manifest tensor sizes do not cover the weight stream")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    dealer_path = args.output_dir / "dealer-weight-mask.u64"
    evaluator_path = args.output_dir / "evaluator-masked-weights.u64"
    ring_mask = np.uint64((1 << args.bitwidth) - 1)
    fixed_min: int | None = None
    fixed_max: int | None = None

    with dealer_path.open("wb") as dealer, evaluator_path.open("wb") as evaluator:
        for start in range(0, float_weights.size, args.chunk_elements):
            stop = min(start + args.chunk_elements, float_weights.size)
            values = np.asarray(float_weights[start:stop], dtype=np.float64)
            scales = scale_by_element[start:stop]
            scale_factors = np.ldexp(
                np.ones(values.shape, dtype=np.float64), scales.astype(np.int32)
            )
            scaled = values * scale_factors
            if not np.all(np.isfinite(scaled)):
                raise OverflowError("non-finite value during fixed-point conversion")
            fixed = np.trunc(scaled).astype(np.int64)
            chunk_min, chunk_max = int(fixed.min()), int(fixed.max())
            fixed_min = chunk_min if fixed_min is None else min(fixed_min, chunk_min)
            fixed_max = chunk_max if fixed_max is None else max(fixed_max, chunk_max)
            encoded = fixed.view(np.uint64) & ring_mask
            raw = secrets.token_bytes(encoded.size * 8)
            weight_mask = np.frombuffer(raw, dtype="<u8") & ring_mask
            masked_weight = (encoded + weight_mask) & ring_mask
            weight_mask.astype("<u8", copy=False).tofile(dealer)
            masked_weight.astype("<u8", copy=False).tofile(evaluator)

    result = {
        "scheme": "SIGMA masked model parameters",
        "layers": manifest["layers"],
        "parameters": manifest["parameters"],
        "bitwidth": args.bitwidth,
        "weight_scale": args.scale,
        "bias_scale": 2 * args.scale,
        "rounding": "truncate toward zero (matches SytorchModule::load)",
        "fixed_min": fixed_min,
        "fixed_max": fixed_max,
        "randomness": "Python secrets.token_bytes (OS CSPRNG)",
        "source_float_sha256": sha256(args.float_weights),
        "dealer_weight_mask": {
            "file": dealer_path.name,
            "bytes": dealer_path.stat().st_size,
            "sha256": sha256(dealer_path),
        },
        "evaluator_masked_weights": {
            "file": evaluator_path.name,
            "bytes": evaluator_path.stat().st_size,
            "sha256": sha256(evaluator_path),
        },
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
