#!/usr/bin/env python3
"""Guarded, randomized paired benchmark for FABLE and SS-LinearScan.

This controller is intended for a shared A100 server.  It freezes one GPU pair,
waits for a stable low-load window before every measured command, alternates
method order, uses fresh preprocessing material, records one-second host/GPU
telemetry, and preserves contaminated samples instead of silently deleting them.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import re
import signal
import statistics
import subprocess
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
import psutil

from ss_linear_scan_common import atomic_write_json, load_json


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--pairs", type=int, default=7)
    parser.add_argument("--warmup-pairs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260720)
    parser.add_argument(
        "--gpus",
        help=(
            "fixed GPU assignment: P0,P1 for two cards or one index to colocate "
            "both parties; omit to select and freeze the first eligible pair"
        ),
    )
    parser.add_argument("--dealer-gpu", type=int)
    parser.add_argument("--preflight-seconds", type=int, default=60)
    parser.add_argument("--sample-interval", type=float, default=1.0)
    parser.add_argument("--max-wait-minutes", type=float, default=120)
    parser.add_argument("--max-preflight-util", type=int, default=20)
    parser.add_argument("--min-free-mib", type=int, default=20000)
    parser.add_argument("--max-memory-drift-mib", type=int, default=1024)
    parser.add_argument("--foreign-memory-growth-mib", type=int, default=2048)
    parser.add_argument("--method-timeout-minutes", type=float, default=60)
    parser.add_argument("--fable-threads", type=int, default=32)
    parser.add_argument("--fable-label", default="FABLE-current-24-chunks")
    parser.add_argument("--p0-cpuset", default="")
    parser.add_argument("--p1-cpuset", default="")
    parser.add_argument("--table", type=Path, default=SCRIPT_DIR / "artifacts/fable-table/xlmr_word_embedding_scale12.i16")
    parser.add_argument("--queries", type=Path, default=SCRIPT_DIR / "artifacts/plaintext-bridge-smoke/token_ids.u32")
    parser.add_argument("--table-shares", type=Path, default=SCRIPT_DIR / "artifacts/ss-linear-scan/xlmr-table-b4096")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="wait for and print an eligible frozen GPU pair without running methods",
    )
    parser.add_argument(
        "--skip-warmup",
        action="store_true",
        help="equivalent to --warmup-pairs 0",
    )
    return parser.parse_args()


def nvidia_csv(query: str) -> list[list[str]]:
    completed = subprocess.run(
        ["nvidia-smi", f"--query-{query}", "--format=csv,noheader,nounits"],
        check=True,
        text=True,
        capture_output=True,
    )
    return [
        [field.strip() for field in line.split(",")]
        for line in completed.stdout.splitlines()
        if line.strip()
    ]


def gpu_snapshot() -> dict[str, Any]:
    uuid_to_index: dict[str, int] = {}
    gpus: dict[int, dict[str, Any]] = {}
    for index, uuid, name, used, free, utilization in nvidia_csv(
        "gpu=index,uuid,name,memory.used,memory.free,utilization.gpu"
    ):
        gpu = int(index)
        uuid_to_index[uuid] = gpu
        gpus[gpu] = {
            "name": name,
            "memory_used_mib": int(used),
            "memory_free_mib": int(free),
            "utilization_percent": int(utilization),
            "processes": {},
        }
    try:
        rows = nvidia_csv("compute-apps=gpu_uuid,pid,used_memory")
    except subprocess.CalledProcessError:
        rows = []
    for row in rows:
        if len(row) != 3 or row[0] not in uuid_to_index:
            continue
        try:
            pid, memory = int(row[1]), int(row[2])
        except ValueError:
            continue
        gpus[uuid_to_index[row[0]]]["processes"][pid] = memory
    disk = psutil.disk_io_counters()
    memory = psutil.virtual_memory()
    return {
        "timestamp_unix_ns": time.time_ns(),
        "gpus": gpus,
        "host": {
            "loadavg": list(os.getloadavg()),
            "cpu_percent": psutil.cpu_percent(interval=None),
            "memory_available": memory.available,
            "memory_percent": memory.percent,
            "disk_read_bytes": disk.read_bytes if disk else None,
            "disk_write_bytes": disk.write_bytes if disk else None,
        },
    }


def eligible_gpus(
    samples: list[dict[str, Any]], args: argparse.Namespace
) -> list[tuple[int, dict[str, Any]]]:
    if not samples:
        return []
    result = []
    for gpu in sorted(samples[-1]["gpus"]):
        values = [sample["gpus"][gpu] for sample in samples]
        utilizations = [value["utilization_percent"] for value in values]
        used = [value["memory_used_mib"] for value in values]
        free = [value["memory_free_mib"] for value in values]
        pid_sets = [set(value["processes"]) for value in values]
        metrics = {
            "utilization_p95": float(np.percentile(utilizations, 95)),
            "utilization_max": max(utilizations),
            "memory_used_drift_mib": max(used) - min(used),
            "minimum_free_mib": min(free),
            "process_set_stable": all(value == pid_sets[0] for value in pid_sets),
            "processes": values[-1]["processes"],
        }
        if (
            metrics["utilization_p95"] <= args.max_preflight_util
            and metrics["minimum_free_mib"] >= args.min_free_mib
            and metrics["memory_used_drift_mib"] <= args.max_memory_drift_mib
            and metrics["process_set_stable"]
        ):
            result.append((gpu, metrics))
    return sorted(
        result,
        key=lambda item: (
            item[1]["utilization_p95"],
            -item[1]["minimum_free_mib"],
        ),
    )


def wait_for_window(
    args: argparse.Namespace,
    fixed_gpus: tuple[int, int] | None,
    log_path: Path,
) -> tuple[tuple[int, int], dict[str, Any]]:
    required = max(2, math.ceil(args.preflight_seconds / args.sample_interval))
    samples: deque[dict[str, Any]] = deque(maxlen=required)
    deadline = time.monotonic() + args.max_wait_minutes * 60
    log_path.parent.mkdir(parents=True, exist_ok=True)
    last_notice = 0.0
    with log_path.open("a", encoding="utf-8") as log:
        while True:
            sample = gpu_snapshot()
            samples.append(sample)
            log.write(json.dumps(sample, sort_keys=True) + "\n")
            log.flush()
            if len(samples) == required:
                eligible = dict(eligible_gpus(list(samples), args))
                if fixed_gpus is None:
                    if len(eligible) >= 2:
                        chosen = tuple(list(eligible)[:2])
                        return (chosen[0], chosen[1]), {
                            "samples": len(samples),
                            "gpus": {str(gpu): eligible[gpu] for gpu in chosen},
                            "last_snapshot": sample,
                        }
                elif all(gpu in eligible for gpu in fixed_gpus):
                    return fixed_gpus, {
                        "samples": len(samples),
                        "gpus": {str(gpu): eligible[gpu] for gpu in fixed_gpus},
                        "last_snapshot": sample,
                    }
            now = time.monotonic()
            if now >= deadline:
                raise TimeoutError("no stable GPU window before --max-wait-minutes")
            if now - last_notice >= 30:
                candidates = [gpu for gpu, _ in eligible_gpus(list(samples), args)]
                print(
                    f"waiting for stable GPUs: {len(samples)}/{required} samples; "
                    f"currently eligible={candidates}",
                    flush=True,
                )
                last_notice = now
            time.sleep(args.sample_interval)


def process_group_pids(pgid: int) -> set[int]:
    result = set()
    for process in psutil.process_iter(["pid"]):
        try:
            if os.getpgid(process.pid) == pgid:
                result.add(process.pid)
        except (OSError, psutil.Error):
            continue
    return result


def run_monitored(
    command: list[str],
    environment: dict[str, str],
    run_dir: Path,
    gpus: tuple[int, int],
    baseline_snapshot: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "command.log"
    telemetry_path = run_dir / "telemetry.jsonl"
    started_ns = time.time_ns()
    reasons: set[str] = set()
    baseline_processes = {
        gpu: dict(baseline_snapshot["gpus"][gpu]["processes"]) for gpu in gpus
    }
    with log_path.open("w", encoding="utf-8") as output:
        process = subprocess.Popen(
            command,
            cwd=ROOT_DIR,
            env=environment,
            stdout=output,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            text=True,
        )
        pgid = process.pid
        deadline = time.monotonic() + args.method_timeout_minutes * 60
        with telemetry_path.open("w", encoding="utf-8") as telemetry:
            while process.poll() is None:
                sample = gpu_snapshot()
                own_pids = process_group_pids(pgid)
                for gpu in gpus:
                    current = sample["gpus"][gpu]["processes"]
                    for pid, used in current.items():
                        if pid not in baseline_processes[gpu] and pid not in own_pids:
                            reasons.add(f"gpu{gpu}:new_foreign_pid:{pid}")
                        previous = baseline_processes[gpu].get(pid)
                        if (
                            previous is not None
                            and used - previous > args.foreign_memory_growth_mib
                        ):
                            reasons.add(f"gpu{gpu}:foreign_memory_growth:{pid}")
                sample["experiment"] = {
                    "pgid": pgid,
                    "own_pids": sorted(own_pids),
                    "contamination_reasons": sorted(reasons),
                }
                telemetry.write(json.dumps(sample, sort_keys=True) + "\n")
                telemetry.flush()
                if time.monotonic() >= deadline:
                    reasons.add("method_timeout")
                    os.killpg(pgid, signal.SIGTERM)
                    try:
                        process.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        os.killpg(pgid, signal.SIGKILL)
                    break
                time.sleep(args.sample_interval)
        return_code = process.wait()
    return {
        "command": command,
        "return_code": return_code,
        "wall_ms_controller": (time.time_ns() - started_ns) // 1_000_000,
        "contaminated": bool(reasons),
        "contamination_reasons": sorted(reasons),
        "command_log": str(log_path),
        "telemetry": str(telemetry_path),
    }


def fable_metrics(run_dir: Path) -> dict[str, Any]:
    metadata = load_json(run_dir / "metadata.json")
    verification = load_json(run_dir / "verification.json")
    sent = 0
    for party in (0, 1):
        text = (run_dir / f"P{party}.log").read_text(encoding="utf-8")
        sent += sum(int(value) for value in re.findall(r"sent ([0-9]+) Bytes", text))
    return {
        "artifact_dir": str(run_dir),
        "wall_ms": metadata["wall_ms"],
        "sent_bytes": sent,
        "mismatches": verification["mismatches"],
        "values_checked": verification["values_checked"],
    }


def ss_metrics(run_dir: Path) -> dict[str, Any]:
    metadata = load_json(run_dir / "metadata.json")
    verification = load_json(run_dir / "verification.json")
    return {
        "artifact_dir": str(run_dir),
        "wall_ms": metadata["wall_ms"],
        "sent_bytes": metadata["communication"]["two_party_bytes_sent"],
        "mismatches": verification["mismatches"],
        "values_checked": verification["values_checked"],
    }


def append_csv(path: Path, record: dict[str, Any]) -> None:
    fields = [
        "pair_id",
        "warmup",
        "order",
        "method",
        "method_label",
        "status",
        "contaminated",
        "contamination_reasons",
        "gpu_p0",
        "gpu_p1",
        "wall_ms",
        "sent_bytes",
        "mismatches",
        "values_checked",
        "artifact_dir",
        "triple_dir",
    ]
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        row = dict(record)
        row["contamination_reasons"] = ";".join(record["contamination_reasons"])
        writer.writerow(row)


def summarize(records: list[dict[str, Any]], seed: int) -> dict[str, Any]:
    measured = [record for record in records if not record["warmup"]]
    pairs: dict[int, dict[str, dict[str, Any]]] = {}
    for record in measured:
        pairs.setdefault(record["pair_id"], {})[record["method"]] = record
    clean = []
    clean_method_times: dict[str, list[float]] = {"fable": [], "ss-linear-scan": []}
    for pair_id, values in sorted(pairs.items()):
        if set(values) != {"fable", "ss-linear-scan"}:
            continue
        if any(
            value["status"] != "pass"
            or value["contaminated"]
            or value["mismatches"] != 0
            for value in values.values()
        ):
            continue
        speedup = values["fable"]["wall_ms"] / values["ss-linear-scan"]["wall_ms"]
        clean.append({"pair_id": pair_id, "speedup_fable_over_ss": speedup})
        for method in clean_method_times:
            clean_method_times[method].append(float(values[method]["wall_ms"]))
    ratios = [value["speedup_fable_over_ss"] for value in clean]
    summary: dict[str, Any] = {
        "measured_records": len(measured),
        "pair_ids_seen": len(pairs),
        "complete_pairs": sum(
            set(values) == {"fable", "ss-linear-scan"} for values in pairs.values()
        ),
        "clean_pairs": len(clean),
        "contaminated_records": sum(value["contaminated"] for value in measured),
        "clean_pair_values": clean,
    }
    summary["clean_method_wall_ms"] = {}
    for method, values in clean_method_times.items():
        if not values:
            continue
        summary["clean_method_wall_ms"][method] = {
            "count": len(values),
            "mean": statistics.fmean(values),
            "median": statistics.median(values),
            "min": min(values),
            "max": max(values),
            "stdev": statistics.stdev(values) if len(values) > 1 else None,
            "iqr": float(np.percentile(values, 75) - np.percentile(values, 25)),
        }
    if ratios:
        rng = np.random.default_rng(seed)
        bootstrap = [
            float(np.median(rng.choice(ratios, size=len(ratios), replace=True)))
            for _ in range(10000)
        ]
        summary["paired_speedup"] = {
            "mean": statistics.fmean(ratios),
            "median": statistics.median(ratios),
            "min": min(ratios),
            "max": max(ratios),
            "stdev": statistics.stdev(ratios) if len(ratios) > 1 else None,
            "bootstrap_95pct_ci": [
                float(np.percentile(bootstrap, 2.5)),
                float(np.percentile(bootstrap, 97.5)),
            ],
        }
    return summary


def balanced_schedule(warmups: int, pairs: int, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    result = []
    for warmup, count in ((True, warmups), (False, pairs)):
        orders = [
            ["fable", "ss-linear-scan"] if index % 2 == 0 else ["ss-linear-scan", "fable"]
            for index in range(count)
        ]
        rng.shuffle(orders)
        for index, order in enumerate(orders):
            result.append({"pair_id": index, "warmup": warmup, "order": order})
    return result


def parse_gpu_assignment(value: str | None) -> tuple[int, int] | None:
    if value is None:
        return None
    indices = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if len(indices) == 1:
        indices = (indices[0], indices[0])
    elif len(indices) != 2:
        raise ValueError("--gpus must contain one index or a P0,P1 pair")
    if any(index < 0 for index in indices):
        raise ValueError("--gpus indices must be non-negative")
    return indices


def validate(args: argparse.Namespace) -> None:
    required = [
        args.table,
        args.queries,
        args.table_shares / "manifest.json",
        SCRIPT_DIR / "run_fable_xlmr.sh",
        SCRIPT_DIR / "run_ss_linear_scan.sh",
        SCRIPT_DIR / "generate_ss_beaver_triples.py",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing required artifacts: {missing}")
    if args.pairs <= 0 or args.warmup_pairs < 0:
        raise ValueError("pair counts are invalid")
    parse_gpu_assignment(args.gpus)


def main() -> None:
    args = parse_args()
    if args.skip_warmup:
        args.warmup_pairs = 0
    validate(args)
    schedule = balanced_schedule(args.warmup_pairs, args.pairs, args.seed)
    fixed_gpus = parse_gpu_assignment(args.gpus)
    if args.dry_run:
        snapshot = gpu_snapshot()
        print(json.dumps({"schedule": schedule, "current_gpu_state": snapshot}, indent=2))
        return

    run_id = time.strftime("%Y%m%dT%H%M%S") + "-paired-lookup"
    output_dir = args.output_dir or SCRIPT_DIR / "artifacts/paired-benchmark" / run_id
    output_dir.mkdir(parents=True, exist_ok=False)
    if args.preflight_only:
        selected, selection = wait_for_window(
            args, fixed_gpus, output_dir / "gpu-selection.jsonl"
        )
        result = {"p0": selected[0], "p1": selected[1], "selection": selection}
        atomic_write_json(output_dir / "selected-gpus.json", result)
        print(json.dumps(result, indent=2))
        return
    atomic_write_json(output_dir / "config.json", {**vars(args), "output_dir": str(output_dir), "table": str(args.table), "queries": str(args.queries), "table_shares": str(args.table_shares), "schedule": schedule})
    records: list[dict[str, Any]] = []

    fixed_gpus, selection = wait_for_window(
        args, fixed_gpus, output_dir / "gpu-selection.jsonl"
    )
    atomic_write_json(
        output_dir / "selected-gpus.json",
        {"p0": fixed_gpus[0], "p1": fixed_gpus[1], "selection": selection},
    )
    dealer_gpu = args.dealer_gpu if args.dealer_gpu is not None else fixed_gpus[0]
    print(f"frozen GPU pair: P0={fixed_gpus[0]}, P1={fixed_gpus[1]}", flush=True)

    for schedule_index, pair in enumerate(schedule):
        pair_label = f"{'warmup' if pair['warmup'] else 'pair'}-{pair['pair_id']:02d}"
        for order_index, method in enumerate(pair["order"]):
            method_dir = output_dir / pair_label / f"{order_index}-{method}"
            triple_dir: Path | None = None
            preprocessing: dict[str, Any] | None = None
            if method == "ss-linear-scan":
                triple_dir = output_dir / "triples" / f"{pair_label}-{order_index}"
                preflight_gpus, _ = wait_for_window(
                    args,
                    fixed_gpus,
                    method_dir / "preprocessing-preflight.jsonl",
                )
                assert preflight_gpus == fixed_gpus
                pre_started = time.perf_counter_ns()
                pre_log = method_dir / "triple-generation.log"
                method_dir.mkdir(parents=True, exist_ok=True)
                with pre_log.open("w", encoding="utf-8") as log:
                    completed = subprocess.run(
                        [
                            sys.executable,
                            str(SCRIPT_DIR / "generate_ss_beaver_triples.py"),
                            "--backend",
                            "torch-u50",
                            "--gpu",
                            str(dealer_gpu),
                            "--output-dir",
                            str(triple_dir),
                            "--logical-n",
                            "250002",
                            "--queries",
                            "512",
                            "--output-dim",
                            "768",
                            "--block-size",
                            "4096",
                        ],
                        cwd=ROOT_DIR,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        text=True,
                    )
                preprocessing = {
                    "return_code": completed.returncode,
                    "wall_ms": (time.perf_counter_ns() - pre_started) // 1_000_000,
                    "triple_dir": str(triple_dir),
                }
                if completed.returncode:
                    raise RuntimeError(f"triple generation failed: {pre_log}")

            _, gate = wait_for_window(
                args,
                fixed_gpus,
                method_dir / "online-preflight.jsonl",
            )
            environment = os.environ.copy()
            environment.update(
                {
                    "XLMR_FABLE_TABLE": str(args.table),
                    "XLMR_TOKEN_IDS": str(args.queries),
                    "FABLE_THREADS": str(args.fable_threads),
                    "FABLE_P0_CPUSET": args.p0_cpuset,
                    "FABLE_P1_CPUSET": args.p1_cpuset,
                    "SS_LINEAR_SCAN_BACKEND": "torch-u50",
                    "SS_LINEAR_SCAN_P0_GPU": str(fixed_gpus[0]),
                    "SS_LINEAR_SCAN_P1_GPU": str(fixed_gpus[1]),
                    "SS_LINEAR_SCAN_P0_CPUSET": args.p0_cpuset,
                    "SS_LINEAR_SCAN_P1_CPUSET": args.p1_cpuset,
                    "FABLE_XLMR_PORT": str(18800 + schedule_index * 4 + order_index),
                    "SS_LINEAR_SCAN_PORT": str(28800 + schedule_index * 4 + order_index),
                }
            )
            if method == "fable":
                command = [str(SCRIPT_DIR / "run_fable_xlmr.sh"), "0", "24"]
                artifact_dir = None
            else:
                assert triple_dir is not None
                artifact_dir = method_dir / "artifacts"
                command = [
                    str(SCRIPT_DIR / "run_ss_linear_scan.sh"),
                    str(args.table_shares),
                    str(triple_dir),
                    str(artifact_dir),
                ]
            monitored = run_monitored(
                command,
                environment,
                method_dir,
                fixed_gpus,
                gate["last_snapshot"],
                args,
            )
            status = "fail" if monitored["return_code"] else "pass"
            metrics: dict[str, Any] = {}
            if status == "pass" and method == "fable":
                text = Path(monitored["command_log"]).read_text(encoding="utf-8")
                match = re.search(r"^Artifacts: (.+)$", text, re.MULTILINE)
                if not match:
                    status = "fail"
                else:
                    artifact_dir = Path(match.group(1).strip())
                    metrics = fable_metrics(artifact_dir)
            elif status == "pass":
                assert artifact_dir is not None
                metrics = ss_metrics(artifact_dir)
            record = {
                "pair_id": pair["pair_id"],
                "warmup": pair["warmup"],
                "order": order_index,
                "method": method,
                "method_label": (
                    args.fable_label if method == "fable" else "SS-LinearScan-A100"
                ),
                "status": status,
                "gpu_p0": fixed_gpus[0],
                "gpu_p1": fixed_gpus[1],
                "preflight": gate,
                "preprocessing": preprocessing,
                "triple_dir": str(triple_dir) if triple_dir else "",
                **monitored,
                **metrics,
            }
            records.append(record)
            with (output_dir / "records.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
            append_csv(output_dir / "runs.csv", record)
            atomic_write_json(output_dir / "summary.json", summarize(records, args.seed))
            print(
                f"{pair_label} {method}: {status}; "
                f"wall_ms={record.get('wall_ms')}; contaminated={record['contaminated']}",
                flush=True,
            )
            if status != "pass":
                raise RuntimeError(f"method failed; see {monitored['command_log']}")

    summary = summarize(records, args.seed)
    atomic_write_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2))
    print(f"paired benchmark artifacts: {output_dir}")


if __name__ == "__main__":
    main()
