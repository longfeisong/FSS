#!/usr/bin/env python3
"""Secret-share an int16 embedding table for SS-LinearScan.

This command is run by the model owner (P0).  In a distributed deployment P0
keeps P0-table-share.u64 and transfers only P1-table-share.u64 to P1.  The local
artifact layout stores both files together solely to reproduce two-party runs.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

from ss_linear_scan_common import encode_signed, ring_mask, secure_random_ring, sha256


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--table", type=Path, required=True, help="row-major int16 table")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--logical-n", type=int, default=250002)
    parser.add_argument("--output-dim", type=int, default=768)
    parser.add_argument("--block-size", type=int, default=4096)
    parser.add_argument("--ring-bits", type=int, default=50)
    parser.add_argument("--fixed-scale", type=int, default=12)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.logical_n <= 0 or args.output_dim <= 0 or args.block_size <= 0:
        raise ValueError("table dimensions and block size must be positive")
    mask = ring_mask(args.ring_bits)
    expected_bytes = args.logical_n * args.output_dim * 2
    if args.table.stat().st_size != expected_bytes:
        raise ValueError(
            f"{args.table}: expected {expected_bytes} int16-table bytes, "
            f"got {args.table.stat().st_size}"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output0 = args.output_dir / "P0-table-share.u64"
    output1 = args.output_dir / "P1-table-share.u64"
    manifest_path = args.output_dir / "manifest.json"
    existing = [path for path in (output0, output1, manifest_path) if path.exists()]
    if existing and not args.force:
        raise FileExistsError(
            f"refusing to overwrite {existing[0]}; pass --force or choose a new directory"
        )

    physical_n = ((args.logical_n + args.block_size - 1) // args.block_size) * args.block_size
    source = np.memmap(
        args.table, mode="r", dtype="<i2", shape=(args.logical_n, args.output_dim)
    )
    temporary0 = output0.with_suffix(f".u64.{os.getpid()}.tmp")
    temporary1 = output1.with_suffix(f".u64.{os.getpid()}.tmp")
    started = time.perf_counter_ns()
    try:
        with temporary0.open("wb") as handle0, temporary1.open("wb") as handle1:
            for start in range(0, physical_n, args.block_size):
                end = start + args.block_size
                clear = np.zeros((args.block_size, args.output_dim), dtype=np.uint64)
                valid_end = min(end, args.logical_n)
                if start < valid_end:
                    clear[: valid_end - start] = encode_signed(
                        source[start:valid_end], args.ring_bits
                    )
                share0 = secure_random_ring(clear.shape, args.ring_bits)
                share1 = (clear - share0) & mask
                share0.astype("<u8", copy=False).tofile(handle0)
                share1.astype("<u8", copy=False).tofile(handle1)
                handle0.flush()
                handle1.flush()
            os.fsync(handle0.fileno())
            os.fsync(handle1.fileno())
        os.replace(temporary0, output0)
        os.replace(temporary1, output1)
    finally:
        temporary0.unlink(missing_ok=True)
        temporary1.unlink(missing_ok=True)

    elapsed_ms = (time.perf_counter_ns() - started) // 1_000_000
    manifest = {
        "scheme": "ss-linear-scan-static-table-shares",
        "security": {
            "randomness": "operating-system CSPRNG",
            "deployment_note": (
                "P0 retains P0-table-share and transfers only P1-table-share to P1; "
                "the local reproduction directory contains both"
            ),
        },
        "profile": {
            "logical_n": args.logical_n,
            "physical_n": physical_n,
            "output_dim": args.output_dim,
            "block_size": args.block_size,
            "ring_bits": args.ring_bits,
            "fixed_scale": args.fixed_scale,
            "one_hot_scale": 0,
            "lookup_truncation": "none",
        },
        "source": {
            "path": str(args.table),
            "dtype": "<i2",
            "shape": [args.logical_n, args.output_dim],
            "sha256": sha256(args.table),
        },
        "files": {
            "party0": {
                "file": output0.name,
                "bytes": output0.stat().st_size,
                "sha256": sha256(output0),
            },
            "party1": {
                "file": output1.name,
                "bytes": output1.stat().st_size,
                "sha256": sha256(output1),
            },
        },
        "distribution": {
            "p1_transfer_bytes": output1.stat().st_size,
            "transfer_measured": False,
            "note": "copy P1-table-share to P1 over the deployment transport",
        },
        "elapsed_ms": elapsed_ms,
    }
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
