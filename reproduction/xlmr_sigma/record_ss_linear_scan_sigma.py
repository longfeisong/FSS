#!/usr/bin/env python3
"""Create a compact, tracked result for a full SS-LinearScan + SIGMA run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from ss_linear_scan_common import atomic_write_json, load_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sequences", type=int, default=4)
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--reference-fable-run", type=Path)
    return parser.parse_args()


def parse_party_log(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def gpu_samples(path: Path) -> dict[str, dict[str, int]]:
    result: dict[str, dict[str, int]] = {}
    if not path.is_file():
        return result
    for line in path.read_text(encoding="utf-8").splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 5:
            continue
        _, gpu, used, free, utilization = fields
        entry = result.setdefault(
            gpu,
            {"max_memory_used_mib": 0, "min_memory_free_mib": 1 << 60, "max_utilization_percent": 0},
        )
        entry["max_memory_used_mib"] = max(entry["max_memory_used_mib"], int(used))
        entry["min_memory_free_mib"] = min(entry["min_memory_free_mib"], int(free))
        entry["max_utilization_percent"] = max(
            entry["max_utilization_percent"], int(utilization)
        )
    return result


def main() -> None:
    args = parse_args()
    run = load_json(args.run_dir / "metadata.json")
    lookup_verify = load_json(args.run_dir / "verification.json")
    bridge_verify = load_json(args.run_dir / "sigma-input" / "verification.json")
    bridge_parties = [
        parse_party_log(args.run_dir / "sigma-input" / f"P{party}.log")
        for party in (0, 1)
    ]
    lookup_parties = [
        load_json(args.run_dir / f"P{party}-metadata.json") for party in (0, 1)
    ]
    sigma = [
        load_json(args.run_dir / f"sigma-{args.layers}layer-seq{sequence}" / "result.json")
        for sequence in range(args.sequences)
    ]
    hashes = [value["output_sha256"] for value in sigma]
    reference_hashes = None
    if args.reference_fable_run:
        reference_hashes = [
            load_json(
                args.reference_fable_run
                / f"sigma-{args.layers}layer-seq{sequence}"
                / "result.json"
            )["output_sha256"]
            for sequence in range(args.sequences)
        ]

    before = {}
    for line in (args.run_dir / "gpu-before.csv").read_text().splitlines():
        gpu, name, used, free, utilization = [part.strip() for part in line.split(",")]
        before[gpu] = {
            "name": name,
            "memory_used_mib": int(used),
            "memory_free_mib": int(free),
            "utilization_percent": int(utilization),
        }
    during = gpu_samples(args.run_dir / "gpu-during.csv")
    lookup_timing_reliable = all(
        value["utilization_percent"] <= 30 for value in before.values()
    ) and all(value["max_utilization_percent"] <= 30 for value in during.values())
    result = {
        "run_id": args.run_dir.name,
        "status": "pass",
        "model": "FacebookAI/xlm-roberta-base",
        "model_revision": "e73636d4f797dec63c3081bb6ed5c7b0bb3f2089",
        "scope": (
            "full 512x768 SS-LinearScan followed by the common share-to-mask "
            "bridge and XLM-R encoder layer 0 on all four sequences"
        ),
        "lookup": {
            "backend": lookup_parties[0]["compute_backend"]["name"],
            "gpu_devices": [
                lookup_parties[party]["compute_backend"]["device"]
                for party in (0, 1)
            ],
            "table_shape": [250002, 768],
            "logical_n": lookup_parties[0]["profile"]["logical_n"],
            "physical_n": lookup_parties[0]["profile"]["physical_n"],
            "row_block": lookup_parties[0]["profile"]["block_size"],
            "queries": lookup_parties[0]["profile"]["queries"],
            "output_shape": lookup_verify["shape"],
            "values_checked": lookup_verify["values_checked"],
            "mismatches": lookup_verify["mismatches"],
            "reconstructed_sha256": lookup_verify["reconstructed_sha256"],
            "wall_ms": run["wall_ms"],
            "party_wall_ms": run["party_wall_ms"],
            "communication": run["communication"],
            "preprocessing": run["preprocessing"],
            "compute_wall_ms_by_party": {
                f"P{party}": lookup_parties[party]["compute_backend"]["total_wall_ms"]
                for party in (0, 1)
            },
            "gpu_before": before,
            "gpu_during": during,
            "timing_reliable": lookup_timing_reliable,
            "timing_note": (
                "correctness run only: sampled utilization exceeded 30%; the raw "
                "counter includes this protocol and any co-tenant workload"
                if not lookup_timing_reliable
                else "startup and sampled utilization stayed at or below 30%"
            ),
        },
        "share_to_sigma_mask": {
            "values_checked": bridge_verify["values_checked"],
            "mismatches": bridge_verify["mismatches"],
            "same_public_masked_input": bridge_verify["same_public_masked_input"],
            "bytes_sent_per_party": bridge_parties[0]["bytes_sent"],
            "elapsed_us_p0": bridge_parties[0]["elapsed_us"],
            "elapsed_us_p1": bridge_parties[1]["elapsed_us"],
        },
        "sigma": {
            "layers": args.layers,
            "sequence_indices": list(range(args.sequences)),
            "offline_key_bytes_per_party_per_sequence": sigma[0][
                "offline_key_bytes_per_party"
            ],
            "offline_key_bytes_per_party_batch": sum(
                value["offline_key_bytes_per_party"] for value in sigma
            ),
            "online_elapsed_us_p0_batch": sum(
                value["online_elapsed_us_p0"] for value in sigma
            ),
            "online_elapsed_us_p1_batch": sum(
                value["online_elapsed_us_p1"] for value in sigma
            ),
            "online_communication_bytes_per_party_per_sequence": sigma[0][
                "online_communication_bytes_per_party"
            ],
            "online_communication_bytes_per_party_batch": sum(
                value["online_communication_bytes_per_party"] for value in sigma
            ),
            "party_outputs_identical_all_sequences": all(
                value["party_outputs_identical"] for value in sigma
            ),
            "output_sha256_by_sequence": hashes,
            "matches_fable_sigma_by_sequence": (
                [left == right for left, right in zip(hashes, reference_hashes)]
                if reference_hashes is not None
                else None
            ),
            "timing_reliable_all_sequences": all(
                value["timing_reliable"] for value in sigma
            ),
        },
        "semantic_boundary": (
            "word embeddings feed the encoder directly; XLM-R position embeddings "
            "and embedding LayerNorm are not included"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
