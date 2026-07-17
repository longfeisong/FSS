#!/usr/bin/env python3
"""Archive one two-party run and append it to the comparison ledger."""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import socket
import subprocess
from datetime import datetime
from pathlib import Path


MODEL_CONFIGS = {
    "bert-tiny": (2, 2, 128),
    "bert-base": (12, 12, 768),
    "bert-large": (24, 16, 1024),
    "gpt2": (12, 12, 768),
    "gpt-neo": (24, 16, 2048),
    "gpt-neo-large": (32, 20, 2560),
    "llama7b": (32, 32, 4096),
    "llama13b": (40, 40, 5120),
}

FIELDS = [
    "run_id", "timestamp", "scheme", "variant", "status",
    "model", "layers", "heads", "hidden_size", "seq_len",
    "batch_size", "lut_input_bits", "lut_output_bits", "lut_size",
    "network", "host", "gpu_p0", "gpu_p0_name", "gpu_p0_util_pct",
    "gpu_p0_free_mib", "gpu_p1", "gpu_p1_name", "gpu_p1_util_pct",
    "gpu_p1_free_mib", "cpu_threads", "pool_gib", "repetitions",
    "timing_valid", "correctness", "dealer_p0_ms", "dealer_p1_ms",
    "dealer_max_ms", "key_bytes", "online_p0_ms", "online_p1_ms",
    "online_max_ms", "total_comm_bytes", "comm_time_max_ms",
    "transfer_time_max_ms", "mha_time_max_ms", "matmul_time_max_ms",
    "truncate_time_max_ms", "gelu_time_max_ms", "softmax_time_max_ms",
    "layernorm_time_max_ms", "gelu_comm_bytes", "softmax_comm_bytes",
    "layernorm_comm_bytes", "lookup_total_ms", "lookup_comm_bytes",
    "lut_setup_ms", "lut_query_ms", "lut_answer_ms", "lut_decode_ms",
    "git_commit", "notes",
]


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scheme", default="SIGMA")
    parser.add_argument("--variant", default="baseline-zero")
    parser.add_argument("--model", required=True)
    parser.add_argument("--seq-len", required=True, type=int)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--timestamp", default=datetime.now().astimezone().isoformat(timespec="seconds"))
    parser.add_argument("--network", default="local")
    parser.add_argument("--gpu-p0", default="")
    parser.add_argument("--gpu-p0-name", default="")
    parser.add_argument("--gpu-p0-util", default="")
    parser.add_argument("--gpu-p0-free", default="")
    parser.add_argument("--gpu-p1", default="")
    parser.add_argument("--gpu-p1-name", default="")
    parser.add_argument("--gpu-p1-util", default="")
    parser.add_argument("--gpu-p1-free", default="")
    parser.add_argument("--cpu-threads", default="")
    parser.add_argument("--pool-gib", default="")
    parser.add_argument("--repetitions", default="1")
    parser.add_argument("--timing-valid", choices=("yes", "no"), default="yes")
    parser.add_argument("--correctness", default="zero-smoke-test")
    parser.add_argument("--notes", default="")
    parser.add_argument("--results-dir", default="")
    return parser.parse_args()


def parse_stats(path: Path) -> dict[str, int]:
    if not path.is_file():
        raise FileNotFoundError(f"missing SIGMA statistics file: {path}")
    result: dict[str, int] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        match = re.match(r"^([^=]+)=(\d+)", line.strip())
        if match:
            result[match.group(1).strip()] = int(match.group(2))
    return result


def us_to_ms(value: int | None) -> str:
    return "" if value is None else f"{value / 1000:.3f}"


def max_ms(stats0: dict[str, int], stats1: dict[str, int], key: str) -> str:
    values = [stats[key] for stats in (stats0, stats1) if key in stats]
    return us_to_ms(max(values) if values else None)


def same_value(stats0: dict[str, int], stats1: dict[str, int], key: str) -> str:
    values = [stats[key] for stats in (stats0, stats1) if key in stats]
    if not values:
        return ""
    if len(set(values)) != 1:
        raise ValueError(f"P0/P1 disagree on {key}: {values}")
    return str(values[0])


def git_commit(root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=root,
            text=True, stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def write_markdown(csv_path: Path, markdown_path: Path) -> None:
    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    headers = [
        "Run", "Scheme", "Variant", "Model", "Layers", "Seq/Batch",
        "LUT bits", "GPUs", "Online ms", "Comm MiB", "Key MiB",
        "Correctness", "Timing valid",
    ]
    lines = [
        "# SIGMA / FABLE experiment ledger",
        "",
        "This file is generated from `runs.csv`; do not edit it manually.",
        "",
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
    ]
    for row in rows:
        comm = f"{int(row['total_comm_bytes']) / 2**20:.3f}" if row["total_comm_bytes"] else ""
        key = f"{int(row['key_bytes']) / 2**20:.3f}" if row["key_bytes"] else ""
        seq_batch = row["seq_len"]
        if row["batch_size"]:
            seq_batch += f" / {row['batch_size']}"
        lut_bits = row["lut_input_bits"]
        if row["lut_output_bits"]:
            lut_bits += f"→{row['lut_output_bits']}"
        values = [
            row["run_id"], row["scheme"], row["variant"], row["model"],
            row["layers"], seq_batch, lut_bits,
            f"{row['gpu_p0']},{row['gpu_p1']}", row["online_max_ms"],
            comm, key, row["correctness"], row["timing_valid"],
        ]
        lines.append("| " + " | ".join(value.replace("|", "\\|") for value in values) + " |")
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = arguments()
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parents[2]
    output_dir = script_dir / "output"
    results_dir = (
        Path(args.results_dir).expanduser().resolve()
        if args.results_dir
        else script_dir / "results"
    )
    raw_dir = results_dir / "raw" / args.run_id
    csv_path = results_dir / "runs.csv"
    markdown_path = results_dir / "comparison.md"
    results_dir.mkdir(parents=True, exist_ok=True)

    if raw_dir.exists():
        raise FileExistsError(f"run ID already archived: {args.run_id}")

    model_dir0 = output_dir / "P0" / "models" / f"{args.model}-{args.seq_len}"
    model_dir1 = output_dir / "P1" / "models" / f"{args.model}-{args.seq_len}"
    dealer0 = parse_stats(model_dir0 / "dealer.txt")
    dealer1 = parse_stats(model_dir1 / "dealer.txt")
    eval0 = parse_stats(model_dir0 / "evaluator.txt")
    eval1 = parse_stats(model_dir1 / "evaluator.txt")
    layers, heads, hidden_size = MODEL_CONFIGS.get(args.model, ("", "", ""))

    row = {field: "" for field in FIELDS}
    row.update({
        "run_id": args.run_id,
        "timestamp": args.timestamp,
        "scheme": args.scheme,
        "variant": args.variant,
        "status": "success",
        "model": args.model,
        "layers": str(layers),
        "heads": str(heads),
        "hidden_size": str(hidden_size),
        "seq_len": str(args.seq_len),
        "network": args.network,
        "host": socket.gethostname(),
        "gpu_p0": args.gpu_p0,
        "gpu_p0_name": args.gpu_p0_name,
        "gpu_p0_util_pct": args.gpu_p0_util,
        "gpu_p0_free_mib": args.gpu_p0_free,
        "gpu_p1": args.gpu_p1,
        "gpu_p1_name": args.gpu_p1_name,
        "gpu_p1_util_pct": args.gpu_p1_util,
        "gpu_p1_free_mib": args.gpu_p1_free,
        "cpu_threads": args.cpu_threads,
        "pool_gib": args.pool_gib,
        "repetitions": args.repetitions,
        "timing_valid": args.timing_valid,
        "correctness": args.correctness,
        "dealer_p0_ms": us_to_ms(dealer0.get("Total time")),
        "dealer_p1_ms": us_to_ms(dealer1.get("Total time")),
        "dealer_max_ms": max_ms(dealer0, dealer1, "Total time"),
        "key_bytes": same_value(dealer0, dealer1, "Key size"),
        "online_p0_ms": us_to_ms(eval0.get("Total time")),
        "online_p1_ms": us_to_ms(eval1.get("Total time")),
        "online_max_ms": max_ms(eval0, eval1, "Total time"),
        "total_comm_bytes": same_value(eval0, eval1, "Total Comm"),
        "comm_time_max_ms": max_ms(eval0, eval1, "Comm time"),
        "transfer_time_max_ms": max_ms(eval0, eval1, "Transfer time"),
        "mha_time_max_ms": max_ms(eval0, eval1, "MHA time"),
        "matmul_time_max_ms": max_ms(eval0, eval1, "Matmul time"),
        "truncate_time_max_ms": max_ms(eval0, eval1, "Truncate time"),
        "gelu_time_max_ms": max_ms(eval0, eval1, "Gelu time"),
        "softmax_time_max_ms": max_ms(eval0, eval1, "Softmax time"),
        "layernorm_time_max_ms": max_ms(eval0, eval1, "Layernorm time"),
        "gelu_comm_bytes": same_value(eval0, eval1, "Gelu Comm"),
        "softmax_comm_bytes": same_value(eval0, eval1, "Softmax Comm"),
        "layernorm_comm_bytes": same_value(eval0, eval1, "Layernorm Comm"),
        "git_commit": git_commit(repo_root),
        "notes": args.notes,
    })

    if csv_path.exists():
        with csv_path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames != FIELDS:
                raise ValueError("existing runs.csv has an incompatible header")
            if any(existing["run_id"] == args.run_id for existing in reader):
                raise ValueError(f"duplicate run ID: {args.run_id}")

    raw_dir.mkdir(parents=True)
    for party, model_dir in (("P0", model_dir0), ("P1", model_dir1)):
        for filename in ("dealer.txt", "evaluator.txt", "logs.txt"):
            source = model_dir / filename
            if source.is_file():
                shutil.copy2(source, raw_dir / f"{party}-{filename}")
    (raw_dir / "metadata.json").write_text(
        json.dumps(row, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    write_header = not csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)
    write_markdown(csv_path, markdown_path)
    print(f"Recorded experiment: {args.run_id}")
    print(f"CSV: {csv_path}")
    print(f"Table: {markdown_path}")


if __name__ == "__main__":
    main()
