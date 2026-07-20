#!/usr/bin/env python3
"""Generate dealer material for converting FABLE shares to SIGMA inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import secrets
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--queries", type=int, default=512)
    parser.add_argument("--hidden-size", type=int, default=768)
    parser.add_argument("--bitwidth", type=int, default=50)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def random_ring_array(count: int, mask: np.uint64) -> np.ndarray:
    raw = secrets.token_bytes(count * np.dtype("<u8").itemsize)
    return np.frombuffer(raw, dtype="<u8").copy() & mask


def write(path: Path, values: np.ndarray) -> dict[str, object]:
    values.astype("<u8", copy=False).tofile(path)
    return {
        "file": path.name,
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def main() -> None:
    args = parse_args()
    if args.bitwidth <= 0 or args.bitwidth >= 64:
        raise ValueError("bitwidth must be in [1, 63]")
    count = args.queries * args.hidden_size
    mask = np.uint64((1 << args.bitwidth) - 1)
    share0 = random_ring_array(count, mask)
    share1 = random_ring_array(count, mask)
    dealer_mask = (share0 + share1) & mask

    args.output_dir.mkdir(parents=True, exist_ok=True)
    files = {
        "p0_mask_share": write(args.output_dir / "P0-mask-share.u64", share0),
        "p1_mask_share": write(args.output_dir / "P1-mask-share.u64", share1),
        "dealer_mask": write(args.output_dir / "dealer-input-mask.u64", dealer_mask),
    }
    manifest = {
        "scheme": "fable-additive-shares-to-sigma-masked-input",
        "shape": [args.queries, args.hidden_size],
        "bitwidth": args.bitwidth,
        "randomness": "Python secrets.token_bytes (OS CSPRNG)",
        "files": files,
        "access": {
            "P0": "P0-mask-share.u64 only",
            "P1": "P1-mask-share.u64 only",
            "SIGMA dealer": "dealer-input-mask.u64 only",
        },
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
