#!/usr/bin/env python3
"""Prepare XLM-R word-embedding shares and SIGMA masked inputs.

This is a functional bridge baseline. Tokenization and lookup happen in
plaintext, so it does not protect token IDs. The additive-share files define
the interface that the future FABLE lookup must produce.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import platform
from pathlib import Path
from typing import Any

import numpy as np
from safetensors import safe_open


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--text", action="append", dest="texts")
    parser.add_argument("--sequence-length", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument(
        "--allow-padding",
        action="store_true",
        help="Permit padded queries; unsuitable for the default timing profile.",
    )
    parser.add_argument("--scale", type=int, default=12)
    parser.add_argument("--bitwidth", type=int, default=50)
    parser.add_argument(
        "--query-seed",
        type=int,
        default=12345,
        help="Seed for the FABLE-author-aligned random token-ID workload.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20260717,
        help="Reproducible test-share seed; not cryptographically secure.",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_config(model_dir: Path) -> dict[str, Any]:
    with (model_dir / "config.json").open(encoding="utf-8") as handle:
        config = json.load(handle)
    expected = {
        "vocab_size": 250002,
        "hidden_size": 768,
        "intermediate_size": 3072,
        "num_hidden_layers": 12,
        "num_attention_heads": 12,
    }
    mismatches = {
        key: (config.get(key), value)
        for key, value in expected.items()
        if config.get(key) != value
    }
    if mismatches:
        raise ValueError(f"unexpected XLM-R configuration: {mismatches}")
    return config


def load_word_embeddings(model_dir: Path) -> np.ndarray:
    weights_path = model_dir / "model.safetensors"
    if not weights_path.is_file():
        raise FileNotFoundError(
            f"{weights_path} not found; run reproduction/xlmr_sigma/download_model.sh"
        )
    tensor_name = "roberta.embeddings.word_embeddings.weight"
    with safe_open(weights_path, framework="np", device="cpu") as weights:
        if tensor_name not in weights.keys():
            raise KeyError(f"{tensor_name} not present in {weights_path}")
        table = weights.get_tensor(tensor_name)
    if table.shape != (250002, 768):
        raise ValueError(f"unexpected embedding shape: {table.shape}")
    return table


def ring_encode(signed_values: np.ndarray, bitwidth: int) -> np.ndarray:
    mask = np.uint64((1 << bitwidth) - 1)
    return signed_values.astype(np.int64).view(np.uint64) & mask


def fable_author_random_ids(
    batch_size: int, sequence_length: int, vocab_size: int, seed: int
) -> np.ndarray:
    """Match embedding.cpp's sequential srand(seed); rand() % vocab_size."""
    libc = ctypes.CDLL(None)
    libc.srand.argtypes = [ctypes.c_uint]
    libc.rand.restype = ctypes.c_int
    libc.srand(ctypes.c_uint(seed))
    values = [
        libc.rand() % vocab_size for _ in range(batch_size * sequence_length)
    ]
    return np.asarray(values, dtype=np.uint32).reshape(batch_size, sequence_length)


def write_array(path: Path, array: np.ndarray) -> dict[str, Any]:
    contiguous = np.ascontiguousarray(array)
    contiguous.tofile(path)
    return {
        "file": path.name,
        "dtype": contiguous.dtype.str,
        "shape": list(contiguous.shape),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def main() -> None:
    args = parse_args()
    if args.sequence_length != 128:
        raise ValueError("the first author-aligned profile requires sequence length 128")
    if args.batch_size != 4:
        raise ValueError("batch size 4 gives FABLE's author-used 512 lookup queries")
    if args.scale != 12 or args.bitwidth != 50:
        raise ValueError("the SIGMA bert-base profile requires scale=12 and bitwidth=50")

    config = load_config(args.model_dir)
    texts = list(args.texts or [])
    if texts:
        from transformers import AutoTokenizer

        if len(texts) > args.batch_size:
            raise ValueError(
                f"at most {args.batch_size} --text arguments are supported"
            )
        while len(texts) < args.batch_size:
            texts.append(texts[len(texts) % len(texts)])
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_dir,
            local_files_only=True,
            use_fast=False,
        )
        encoded = tokenizer(
            texts,
            max_length=args.sequence_length,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="np",
        )
        token_ids = encoded["input_ids"].astype(np.uint32, copy=False)
        attention_mask = encoded["attention_mask"].astype(np.uint8, copy=False)
        input_profile = "xlmr-tokenized-text"
    else:
        token_ids = fable_author_random_ids(
            args.batch_size,
            args.sequence_length,
            config["vocab_size"],
            args.query_seed,
        )
        attention_mask = np.ones_like(token_ids, dtype=np.uint8)
        input_profile = "fable-author-random-token-ids"
    if int(token_ids.max()) >= config["vocab_size"]:
        raise ValueError("tokenizer produced an ID outside the embedding table")
    nonpadding_per_sequence = attention_mask.sum(axis=1)
    if not args.allow_padding and np.any(nonpadding_per_sequence != args.sequence_length):
        raise ValueError(
            "padded inputs are disabled for the 512-query timing profile; "
            "provide longer text or pass --allow-padding for a smoke test"
        )

    word_table = load_word_embeddings(args.model_dir)
    looked_up = word_table[token_ids]
    scaled = np.rint(looked_up.astype(np.float64) * (1 << args.scale))
    i16 = np.iinfo(np.int16)
    if scaled.min() < i16.min or scaled.max() > i16.max:
        raise OverflowError(
            f"scale={args.scale} does not fit int16: [{scaled.min()}, {scaled.max()}]"
        )
    fixed = scaled.astype(np.int16)
    decoded = fixed.astype(np.float64) / (1 << args.scale)
    max_abs_error = float(np.max(np.abs(decoded - looked_up.astype(np.float64))))

    ring = ring_encode(fixed, args.bitwidth)
    ring_mask = np.uint64((1 << args.bitwidth) - 1)
    rng = np.random.default_rng(args.seed)
    share0 = rng.integers(0, 1 << args.bitwidth, size=ring.shape, dtype=np.uint64)
    share1 = (ring - share0) & ring_mask
    dealer_mask = rng.integers(
        0, 1 << args.bitwidth, size=ring.shape, dtype=np.uint64
    )
    sigma_masked = (ring + dealer_mask) & ring_mask

    if not np.array_equal((share0 + share1) & ring_mask, ring):
        raise AssertionError("additive-share reconstruction failed")
    if not np.array_equal((sigma_masked - dealer_mask) & ring_mask, ring):
        raise AssertionError("SIGMA mask reconstruction failed")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "scheme": "xlmr-plaintext-lookup-sigma-bridge",
        "security": {
            "token_lookup_private": False,
            "shares_cryptographically_random": False,
            "reason": "plaintext control path with a deterministic test seed",
        },
        "model": {
            "id": "FacebookAI/xlm-roberta-base",
            "vocab_size": config["vocab_size"],
            "hidden_size": config["hidden_size"],
        },
        "profile": {
            "sequence_length": args.sequence_length,
            "sequence_batch_size": args.batch_size,
            "lookup_batch_size": int(token_ids.size),
            "fixed_point_scale": args.scale,
            "embedding_element_bits": 16,
            "sigma_bitwidth": args.bitwidth,
            "fable_input_bits": 20,
            "fable_domain_size": 1 << 20,
            "nonpadding_tokens_per_sequence": [
                int(value) for value in nonpadding_per_sequence
            ],
            "padding_queries": int(token_ids.size - attention_mask.sum()),
            "unique_token_ids": int(np.unique(token_ids).size),
        },
        "quantization": {
            "rounding": "numpy.rint (ties-to-even)",
            "fixed_min": int(fixed.min()),
            "fixed_max": int(fixed.max()),
            "max_abs_error": max_abs_error,
        },
        "seed": args.seed,
        "query_seed": args.query_seed,
        "input_profile": input_profile,
        "libc": platform.libc_ver(),
        "texts": texts,
        "files": {},
    }

    common_files = {
        "token_ids": write_array(args.output_dir / "token_ids.u32", token_ids),
        "attention_mask": write_array(
            args.output_dir / "attention_mask.u8", attention_mask
        ),
        "embedding_fixed": write_array(
            args.output_dir / "embedding_fixed.i16", fixed
        ),
    }
    manifest["files"].update(common_files)

    sequences = []
    for index in range(args.batch_size):
        seq_dir = args.output_dir / f"sequence_{index:03d}"
        seq_dir.mkdir(exist_ok=True)
        sequence_files = {
            "share0": write_array(seq_dir / "share0.u64", share0[index]),
            "share1": write_array(seq_dir / "share1.u64", share1[index]),
            "sigma_dealer_mask": write_array(
                seq_dir / "sigma_dealer_mask.u64", dealer_mask[index]
            ),
            "sigma_masked_input": write_array(
                seq_dir / "sigma_masked_input.u64", sigma_masked[index]
            ),
        }
        sequences.append({"index": index, "files": sequence_files})
    manifest["sequences"] = sequences

    manifest_path = args.output_dir / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)
        handle.write("\n")

    print(f"Prepared {token_ids.size} XLM-R lookup queries")
    print(f"Input profile: {input_profile}")
    print(
        "Non-padding tokens per sequence: "
        f"{[int(value) for value in nonpadding_per_sequence]}"
    )
    print(f"Unique token IDs: {np.unique(token_ids).size}")
    print(f"Embedding tensor: {tuple(fixed.shape)}, int16 scale={args.scale}")
    print(f"Quantization max abs error: {max_abs_error:.9g}")
    print("Verified share0 + share1 = embedding (mod 2^50)")
    print("Verified sigma_masked_input - dealer_mask = embedding (mod 2^50)")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
