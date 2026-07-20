#!/usr/bin/env python3
"""Export XLM-R encoder weights in SIGMA/Sytorch GPUBERT load order."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from safetensors import safe_open


MAX_LAYERS = 12
HIDDEN = 768
INTERMEDIATE = 3072
PARAMETERS_PER_LAYER = 7_087_872


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--layers", type=int, default=MAX_LAYERS)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    if not 1 <= args.layers <= MAX_LAYERS:
        raise ValueError(f"layers must be in [1, {MAX_LAYERS}]")
    source_path = args.model_dir / "model.safetensors"
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / f"xlmr_encoder_{args.layers}layer_sytorch.float32"

    parameter_count = 0
    order = []
    with safe_open(source_path, framework="np", device="cpu") as source:

        def get(name: str, shape: tuple[int, ...]) -> np.ndarray:
            if name not in source.keys():
                raise KeyError(name)
            value = source.get_tensor(name)
            if value.shape != shape:
                raise ValueError(f"{name}: expected {shape}, got {value.shape}")
            return value.astype("<f4", copy=False)

        def write(handle, label: str, value: np.ndarray) -> None:
            nonlocal parameter_count
            contiguous = np.ascontiguousarray(value, dtype="<f4")
            contiguous.tofile(handle)
            parameter_count += contiguous.size
            order.append({"name": label, "shape": list(contiguous.shape)})

        with output_path.open("wb") as output:
            for layer in range(args.layers):
                prefix = f"roberta.encoder.layer.{layer}"
                query_w = get(
                    f"{prefix}.attention.self.query.weight", (HIDDEN, HIDDEN)
                ).T
                key_w = get(
                    f"{prefix}.attention.self.key.weight", (HIDDEN, HIDDEN)
                ).T
                value_w = get(
                    f"{prefix}.attention.self.value.weight", (HIDDEN, HIDDEN)
                ).T
                write(
                    output,
                    f"layer.{layer}.mha.wQKV",
                    np.concatenate((query_w, key_w, value_w), axis=1),
                )

                query_b = get(
                    f"{prefix}.attention.self.query.bias", (HIDDEN,)
                )
                key_b = get(f"{prefix}.attention.self.key.bias", (HIDDEN,))
                value_b = get(
                    f"{prefix}.attention.self.value.bias", (HIDDEN,)
                )
                write(
                    output,
                    f"layer.{layer}.mha.bQKV",
                    np.concatenate((query_b, key_b, value_b)),
                )
                write(
                    output,
                    f"layer.{layer}.mha.wProj",
                    get(
                        f"{prefix}.attention.output.dense.weight",
                        (HIDDEN, HIDDEN),
                    ).T,
                )
                write(
                    output,
                    f"layer.{layer}.mha.bProj",
                    get(f"{prefix}.attention.output.dense.bias", (HIDDEN,)),
                )
                write(
                    output,
                    f"layer.{layer}.ln0.weight",
                    get(f"{prefix}.attention.output.LayerNorm.weight", (HIDDEN,)),
                )
                write(
                    output,
                    f"layer.{layer}.ln0.bias",
                    get(f"{prefix}.attention.output.LayerNorm.bias", (HIDDEN,)),
                )
                write(
                    output,
                    f"layer.{layer}.ffn.up.weight",
                    get(
                        f"{prefix}.intermediate.dense.weight",
                        (INTERMEDIATE, HIDDEN),
                    ).T,
                )
                write(
                    output,
                    f"layer.{layer}.ffn.up.bias",
                    get(
                        f"{prefix}.intermediate.dense.bias", (INTERMEDIATE,)
                    ),
                )
                write(
                    output,
                    f"layer.{layer}.ffn.down.weight",
                    get(
                        f"{prefix}.output.dense.weight",
                        (HIDDEN, INTERMEDIATE),
                    ).T,
                )
                write(
                    output,
                    f"layer.{layer}.ffn.down.bias",
                    get(f"{prefix}.output.dense.bias", (HIDDEN,)),
                )
                write(
                    output,
                    f"layer.{layer}.ln1.weight",
                    get(f"{prefix}.output.LayerNorm.weight", (HIDDEN,)),
                )
                write(
                    output,
                    f"layer.{layer}.ln1.bias",
                    get(f"{prefix}.output.LayerNorm.bias", (HIDDEN,)),
                )

    expected_parameters = PARAMETERS_PER_LAYER * args.layers
    if parameter_count != expected_parameters:
        raise AssertionError(
            f"expected {expected_parameters} parameters, wrote {parameter_count}"
        )
    expected_bytes = expected_parameters * np.dtype("<f4").itemsize
    if output_path.stat().st_size != expected_bytes:
        raise AssertionError(
            f"expected {expected_bytes} bytes, wrote {output_path.stat().st_size}"
        )

    manifest = {
        "format": "SytorchModule::load float32 stream",
        "target": f"GPUBERT<uint64_t>({args.layers}, 12, 768, none, qkvconcat)",
        "source": "FacebookAI/xlm-roberta-base",
        "source_file_sha256": sha256(source_path),
        "output_file": output_path.name,
        "output_sha256": sha256(output_path),
        "dtype": "little-endian float32",
        "parameters": parameter_count,
        "bytes": output_path.stat().st_size,
        "layers": args.layers,
        "order": order,
        "excludes": [
            "word embeddings",
            "position embeddings",
            "token-type embeddings",
            "embedding LayerNorm",
            "masked-LM head",
        ],
    }
    manifest_path = args.output_dir / f"xlmr_encoder_{args.layers}layer_sytorch.manifest.json"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")

    print(f"Exported {parameter_count:,} encoder parameters")
    print(f"Weights: {output_path} ({output_path.stat().st_size:,} bytes)")
    print(f"SHA256: {manifest['output_sha256']}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
