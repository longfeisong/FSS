#!/usr/bin/env python3
"""Generate fresh blockwise Beaver matrix triples for SS-LinearScan."""

from __future__ import annotations

import argparse
import json
import shutil
import time
import uuid
from pathlib import Path

import numpy as np

from ss_linear_scan_common import ring_mask, ring_matmul, secure_random_ring, write_array


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--logical-n", type=int, default=4096)
    parser.add_argument("--queries", type=int, default=8)
    parser.add_argument("--output-dim", type=int, default=32)
    parser.add_argument("--block-size", type=int, default=1024)
    parser.add_argument("--ring-bits", type=int, default=50)
    parser.add_argument("--backend", choices=("numpy", "torch-u50"), default="numpy")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--run-id", default=None)
    parser.add_argument(
        "--inner-chunk",
        type=int,
        default=0,
        help="split the dealer's clear A@B along K; 0 uses one NumPy matmul",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if min(args.logical_n, args.queries, args.output_dim, args.block_size) <= 0:
        raise ValueError("all dimensions must be positive")
    mask = ring_mask(args.ring_bits)
    torch_backend = None
    if args.backend == "torch-u50":
        from ss_linear_scan_torch import TorchU50Backend

        torch_backend = TorchU50Backend(args.gpu, args.ring_bits)
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        if not args.force:
            raise FileExistsError(
                f"{args.output_dir} is not empty; triple material is never overwritten "
                "without --force"
            )
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    party_dirs = [args.output_dir / "party0", args.output_dir / "party1"]
    for directory in party_dirs:
        (directory / "blocks").mkdir(parents=True)

    run_id = args.run_id or f"{time.strftime('%Y%m%dT%H%M%S')}-{uuid.uuid4().hex[:12]}"
    physical_n = ((args.logical_n + args.block_size - 1) // args.block_size) * args.block_size
    blocks: list[dict[str, object]] = []
    total_started = time.perf_counter_ns()
    for block_id, start in enumerate(range(0, physical_n, args.block_size)):
        block_started = time.perf_counter_ns()
        a0 = secure_random_ring((args.queries, args.block_size), args.ring_bits)
        a1 = secure_random_ring((args.queries, args.block_size), args.ring_bits)
        b0 = secure_random_ring((args.block_size, args.output_dim), args.ring_bits)
        b1 = secure_random_ring((args.block_size, args.output_dim), args.ring_bits)
        clear_a = (a0 + a1) & mask
        clear_b = (b0 + b1) & mask
        if torch_backend is None:
            clear_c = ring_matmul(
                clear_a,
                clear_b,
                args.ring_bits,
                inner_chunk=args.inner_chunk,
            )
        else:
            clear_c = torch_backend.matmul(clear_a, clear_b)
        c0 = secure_random_ring(clear_c.shape, args.ring_bits)
        c1 = (clear_c - c0) & mask

        party_files = []
        for party, matrices in enumerate(((a0, b0, c0), (a1, b1, c1))):
            block_dir = party_dirs[party] / "blocks" / f"block_{block_id:06d}"
            party_files.append(
                {
                    name: write_array(block_dir / f"{name}.u64", value)
                    for name, value in zip(("A", "B", "C"), matrices)
                }
            )
        blocks.append(
            {
                "block_id": block_id,
                "row_start": start,
                "row_end": start + args.block_size,
                "party0": party_files[0],
                "party1": party_files[1],
                "generation_ms": (time.perf_counter_ns() - block_started) // 1_000_000,
            }
        )
        print(
            f"generated block {block_id + 1}/{physical_n // args.block_size}",
            flush=True,
        )

    manifest = {
        "scheme": "ss-linear-scan-beaver-matrix-triples",
        "run_id": run_id,
        "freshness": {
            "one_time": True,
            "consumption": "each party atomically reserves every block before reading",
            "retry_policy": "generate a new run after any start, failure, or crash",
            "randomness": "operating-system CSPRNG",
        },
        "profile": {
            "logical_n": args.logical_n,
            "physical_n": physical_n,
            "queries": args.queries,
            "output_dim": args.output_dim,
            "block_size": args.block_size,
            "ring_bits": args.ring_bits,
            "fixed_scale": 12,
            "one_hot_scale": 0,
            "lookup_truncation": "none",
        },
        "blocks": blocks,
        "distribution": {
            "party0_bytes": sum(
                file_info["bytes"]
                for block in blocks
                for file_info in block["party0"].values()
            ),
            "party1_bytes": sum(
                file_info["bytes"]
                for block in blocks
                for file_info in block["party1"].values()
            ),
            "transfer_measured": False,
            "note": "dealer sends each party only its own directory",
        },
        "compute_backend": (
            torch_backend.summary()
            if torch_backend is not None
            else {"name": "numpy-uint64-reference", "inner_chunk": args.inner_chunk}
        ),
        "generation_ms": (time.perf_counter_ns() - total_started) // 1_000_000,
    }
    with (args.output_dir / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")
    print(f"fresh Beaver run: {run_id}")
    print(f"manifest: {args.output_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
