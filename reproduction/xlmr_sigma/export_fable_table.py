#!/usr/bin/env python3
"""Export the full XLM-R word-embedding LUT as row-major signed int16."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from safetensors import safe_open


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--scale", type=int, default=12)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    if args.scale != 12:
        raise ValueError("the SIGMA XLM-R profile requires scale=12")
    source_path = args.model_dir / "model.safetensors"
    tensor_name = "roberta.embeddings.word_embeddings.weight"
    with safe_open(source_path, framework="np", device="cpu") as source:
        table = source.get_tensor(tensor_name)
    if table.shape != (250002, 768):
        raise ValueError(f"unexpected embedding shape: {table.shape}")

    scaled = np.rint(table.astype(np.float64) * (1 << args.scale))
    limits = np.iinfo(np.int16)
    if scaled.min() < limits.min or scaled.max() > limits.max:
        raise OverflowError(
            f"embedding does not fit int16: [{scaled.min()}, {scaled.max()}]"
        )
    fixed = np.ascontiguousarray(scaled.astype("<i2"))
    decoded = fixed.astype(np.float64) / (1 << args.scale)
    max_error = float(np.max(np.abs(decoded - table.astype(np.float64))))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "xlmr_word_embedding_scale12.i16"
    fixed.tofile(output_path)
    manifest = {
        "source": "FacebookAI/xlm-roberta-base",
        "source_tensor": tensor_name,
        "source_model_sha256": sha256(source_path),
        "output_file": output_path.name,
        "output_sha256": sha256(output_path),
        "dtype": "little-endian signed int16",
        "shape": list(fixed.shape),
        "scale": args.scale,
        "fixed_min": int(fixed.min()),
        "fixed_max": int(fixed.max()),
        "max_abs_quantization_error": max_error,
        "bytes": output_path.stat().st_size,
    }
    manifest_path = args.output_dir / "xlmr_word_embedding_scale12.manifest.json"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")

    print(f"Exported FABLE table: {output_path}")
    print(f"Shape: {fixed.shape}; bytes: {output_path.stat().st_size:,}")
    print(f"Fixed range: [{fixed.min()}, {fixed.max()}]")
    print(f"Max quantization error: {max_error:.9g}")
    print(f"SHA256: {manifest['output_sha256']}")


if __name__ == "__main__":
    main()
