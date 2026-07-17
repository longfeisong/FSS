#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import re
from datetime import datetime
from pathlib import Path


FIELDS = [
    "run_id", "timestamp", "scheme", "variant", "component", "model",
    "layers", "heads", "hidden_size", "ffn_hidden_size", "seq_len",
    "queries_per_block", "chunk_size", "chunks", "lut_index_bits",
    "lut_output_bits", "padded_lut_entries", "cpu_threads", "protocol_ms",
    "wall_ms", "comm_bytes", "correctness", "timing_valid", "notes",
]


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--raw-dir", required=True, type=Path)
    parser.add_argument("--results-dir", required=True, type=Path)
    parser.add_argument("--chunks", required=True, type=int)
    parser.add_argument("--batch-size", required=True, type=int)
    parser.add_argument("--threads", required=True, type=int)
    parser.add_argument("--wall-ms", required=True, type=int)
    return parser.parse_args()


def execution(log: Path) -> tuple[int, int]:
    text = log.read_text(encoding="utf-8", errors="replace")
    matches = re.findall(
        r"^FABLE Execution:\s*\n\s*elapsed (\d+) ms,\s*\n\s*sent (\d+) Bytes",
        text,
        flags=re.MULTILINE,
    )
    if not matches or "[FABLE] Test passed" not in text:
        raise ValueError(f"incomplete or failed FABLE log: {log}")
    elapsed, sent = matches[-1]
    return int(elapsed), int(sent)


def write_markdown(csv_path: Path, markdown_path: Path) -> None:
    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    lines = [
        "# FABLE GELU block results",
        "",
        "| Run | Queries | Chunks | LUT | Protocol ms | Wall ms | Comm GiB | Correctness |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['run_id']} | {row['queries_per_block']} | {row['chunks']} × {row['chunk_size']} "
            f"| {row['lut_index_bits']}→{row['lut_output_bits']} (padded {row['padded_lut_entries']}) "
            f"| {row['protocol_ms']} | {row['wall_ms']} "
            f"| {int(row['comm_bytes']) / 2**30:.3f} | {row['correctness']} |"
        )
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = arguments()
    server_ms = client_ms = comm_bytes = 0
    for chunk in range(1, args.chunks + 1):
        prefix = f"chunk-{chunk:02d}"
        elapsed, sent = execution(args.raw_dir / f"{prefix}-server.log")
        server_ms += elapsed
        comm_bytes += sent
        elapsed, sent = execution(args.raw_dir / f"{prefix}-client.log")
        client_ms += elapsed
        comm_bytes += sent

    results_dir = args.results_dir
    csv_path = results_dir / "runs.csv"
    row = {
        "run_id": args.run_id,
        "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
        "scheme": "FABLE",
        "variant": "gelu-correction-padded16",
        "component": "one-transformer-block-gelu-lut",
        "model": "bert-tiny",
        "layers": "1",
        "heads": "2",
        "hidden_size": "128",
        "ffn_hidden_size": "512",
        "seq_len": "128",
        "queries_per_block": str(args.chunks * args.batch_size),
        "chunk_size": str(args.batch_size),
        "chunks": str(args.chunks),
        "lut_index_bits": "8",
        "lut_output_bits": "37",
        "padded_lut_entries": "65536",
        "cpu_threads": str(args.threads),
        "protocol_ms": str(max(server_ms, client_ms)),
        "wall_ms": str(args.wall_ms),
        "comm_bytes": str(comm_bytes),
        "correctness": "all-chunks-zero-error",
        "timing_valid": "provisional-shared-server",
        "notes": "Official BatchPIR requires log-size 16; valid GELU indices remain 0..255.",
    }
    results_dir.mkdir(parents=True, exist_ok=True)
    exists = csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow(row)
    write_markdown(csv_path, results_dir / "comparison.md")


if __name__ == "__main__":
    main()
