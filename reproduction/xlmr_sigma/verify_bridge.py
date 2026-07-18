#!/usr/bin/env python3
"""Verify generated additive shares and SIGMA masked inputs from disk."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact_dir", type=Path)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_array(base: Path, spec: dict) -> np.ndarray:
    path = base / spec["file"]
    if sha256(path) != spec["sha256"]:
        raise ValueError(f"checksum mismatch: {path}")
    return np.fromfile(path, dtype=np.dtype(spec["dtype"])).reshape(spec["shape"])


def main() -> None:
    artifact_dir = parse_args().artifact_dir
    with (artifact_dir / "manifest.json").open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    bitwidth = manifest["profile"]["sigma_bitwidth"]
    ring_mask = np.uint64((1 << bitwidth) - 1)
    fixed = read_array(artifact_dir, manifest["files"]["embedding_fixed"])
    ring = fixed.astype(np.int64).view(np.uint64) & ring_mask

    for sequence in manifest["sequences"]:
        index = sequence["index"]
        seq_dir = artifact_dir / f"sequence_{index:03d}"
        files = sequence["files"]
        share0 = read_array(seq_dir, files["share0"])
        share1 = read_array(seq_dir, files["share1"])
        dealer_mask = read_array(seq_dir, files["sigma_dealer_mask"])
        sigma_masked = read_array(seq_dir, files["sigma_masked_input"])
        if not np.array_equal((share0 + share1) & ring_mask, ring[index]):
            raise AssertionError(f"share reconstruction failed for sequence {index}")
        if not np.array_equal(
            (sigma_masked - dealer_mask) & ring_mask, ring[index]
        ):
            raise AssertionError(f"SIGMA mask reconstruction failed for sequence {index}")
    print(f"Verified {len(manifest['sequences'])} sequences from {artifact_dir}")


if __name__ == "__main__":
    main()
