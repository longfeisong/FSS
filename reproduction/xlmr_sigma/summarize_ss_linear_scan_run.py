#!/usr/bin/env python3
"""Aggregate party, preprocessing, and communication metrics for one run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ss_linear_scan_common import atomic_write_json, load_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--table-shares", type=Path, required=True)
    parser.add_argument("--triples", type=Path, required=True)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--wall-ms", type=int, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    table_manifest = load_json(args.table_shares / "manifest.json")
    triple_manifest = load_json(args.triples / "manifest.json")
    parties = [
        load_json(args.run_dir / f"P{party}-metadata.json") for party in (0, 1)
    ]
    if any(value.get("status") != "pass" for value in parties):
        raise RuntimeError("a party did not report successful completion")
    phase_names = parties[0]["communication"]["phase_bytes_sent_plus_received"]
    phase_wire_bytes = {
        name: sum(
            party["communication"]["phase_bytes_sent_plus_received"][name]
            for party in parties
        )
        // 2
        for name in phase_names
    }
    triple_bytes = sum(
        int(file_info["bytes"])
        for block in triple_manifest["blocks"]
        for party_name in ("party0", "party1")
        for file_info in block[party_name].values()
    )
    profile = triple_manifest["profile"]
    value = {
        "scheme": "ss-linear-scan",
        "status": "pass",
        "run_dir": str(args.run_dir),
        "table_shares": str(args.table_shares),
        "triples": str(args.triples),
        "queries": str(args.queries),
        "wall_ms": args.wall_ms,
        "party_wall_ms": {
            f"P{party['party']}": party["wall_ms"] for party in parties
        },
        "communication": {
            "two_party_bytes_sent": sum(
                party["communication"]["bytes_sent"] for party in parties
            ),
            "two_party_bytes_received": sum(
                party["communication"]["bytes_received"] for party in parties
            ),
            "phase_wire_bytes": phase_wire_bytes,
            "synchronization_phases": parties[0]["communication"][
                "synchronization_phases"
            ],
        },
        "preprocessing": {
            "static_table_share_bytes_total": sum(
                table_manifest["files"][party]["bytes"]
                for party in ("party0", "party1")
            ),
            "static_table_generation_ms": table_manifest["elapsed_ms"],
            "static_table_p1_transfer_bytes": table_manifest["distribution"][
                "p1_transfer_bytes"
            ],
            "fresh_triple_bytes_total": triple_bytes,
            "triple_generation_ms": triple_manifest["generation_ms"],
            "triple_transfer_measured": triple_manifest["distribution"][
                "transfer_measured"
            ],
            "triple_run_id": triple_manifest["run_id"],
            "one_time_material_consumed": True,
        },
        "bridge_compatible": (
            int(profile["queries"]) == 512 and int(profile["output_dim"]) == 768
        ),
    }
    atomic_write_json(args.run_dir / "metadata.json", value)
    print(json.dumps(value, indent=2))


if __name__ == "__main__":
    main()
