#!/usr/bin/env python3
"""Run one online party of the blockwise SS-LinearScan lookup.

P0 receives only a static random share of the model table.  P1 receives only a
static random table share and holds the private token IDs.  For every block P1
secret-shares a one-hot slice, the parties open Beaver D/F differences, and
each accumulates an additive output share.  No lookup result is reconstructed.
"""

from __future__ import annotations

import argparse
import json
import socket
import time
from pathlib import Path

import numpy as np

from ss_linear_scan_common import (
    FramedSocket,
    InheritedFdStream,
    atomic_write_json,
    connect_with_retry,
    connect_unix_with_retry,
    exact_memmap,
    load_json,
    mark_triple_complete,
    reserve_triple,
    ring_mask,
    ring_matmul,
    secure_random_ring,
    sha256,
    validate_profile,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--party", type=int, choices=(0, 1), required=True)
    parser.add_argument("--table-shares", type=Path, required=True)
    parser.add_argument("--triples", type=Path, required=True)
    parser.add_argument("--queries", type=Path, help="required only by P1; uint32 IDs")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18840)
    parser.add_argument(
        "--unix-socket",
        type=Path,
        help="local AF_UNIX transport; leaves TCP host/port unused",
    )
    parser.add_argument(
        "--socket-fd",
        type=int,
        help="already-connected socket descriptor, used by hermetic tests",
    )
    parser.add_argument("--inner-chunk", type=int, default=0)
    parser.add_argument("--backend", choices=("numpy", "torch-u50"), default="numpy")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--connect-timeout", type=float, default=30)
    return parser.parse_args()


def profile_tuple(profile: dict[str, object]) -> tuple[int, int, int, int, int, int]:
    return validate_profile(profile)


def main() -> None:
    args = parse_args()
    if args.party == 1 and args.queries is None:
        raise ValueError("P1 requires --queries")
    table_manifest = load_json(args.table_shares / "manifest.json")
    triple_manifest = load_json(args.triples / "manifest.json")
    table_profile = dict(table_manifest["profile"])
    triple_profile = dict(triple_manifest["profile"])
    logical_n, physical_n, queries_count, output_dim, block_size, bits = profile_tuple(
        triple_profile
    )
    table_comparable = (
        int(table_profile["logical_n"]),
        int(table_profile["physical_n"]),
        int(table_profile["output_dim"]),
        int(table_profile["block_size"]),
        int(table_profile["ring_bits"]),
    )
    if table_comparable != (logical_n, physical_n, output_dim, block_size, bits):
        raise ValueError(
            f"table/triple profile mismatch: {table_comparable} vs "
            f"{(logical_n, physical_n, output_dim, block_size, bits)}"
        )
    if int(table_profile.get("fixed_scale", 12)) != 12:
        raise ValueError("the XLM-R baseline requires table scale=12")
    if int(table_profile.get("one_hot_scale", 0)) != 0:
        raise ValueError("one-hot must use scale 0")
    if table_profile.get("lookup_truncation", "none") != "none":
        raise ValueError("lookup matmul must not truncate")

    table_share_path = args.table_shares / f"P{args.party}-table-share.u64"
    table_share = exact_memmap(table_share_path, (physical_n, output_dim))
    party_triples = args.triples / f"party{args.party}"
    run_id = str(triple_manifest["run_id"])
    block_count = physical_n // block_size
    if len(triple_manifest.get("blocks", [])) != block_count:
        raise ValueError("triple manifest has the wrong block count")

    query_ids: np.ndarray | None = None
    query_sha256: str | None = None
    if args.party == 1:
        expected_query_bytes = queries_count * 4
        if args.queries.stat().st_size != expected_query_bytes:
            raise ValueError(
                f"{args.queries}: expected {expected_query_bytes} bytes, "
                f"got {args.queries.stat().st_size}"
            )
        query_ids = np.fromfile(args.queries, dtype="<u4")
        if int(query_ids.max()) >= logical_n:
            raise ValueError("private query is outside logical table bounds")
        query_sha256 = sha256(args.queries)

    handshake = {
        "protocol": "ss-linear-scan-v1",
        "party": args.party,
        "run_id": run_id,
        "profile": triple_profile,
    }
    if args.socket_fd is not None:
        connection = InheritedFdStream(args.socket_fd)
    elif args.party == 0:
        family = socket.AF_UNIX if args.unix_socket else socket.AF_INET
        listener = socket.socket(family, socket.SOCK_STREAM)
        if args.unix_socket:
            if args.unix_socket.exists():
                raise FileExistsError(f"Unix socket path already exists: {args.unix_socket}")
            args.unix_socket.parent.mkdir(parents=True, exist_ok=True)
            listener.bind(str(args.unix_socket))
        else:
            listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            listener.bind((args.host, args.port))
        listener.listen(1)
        connection, _ = listener.accept()
        listener.close()
    else:
        if args.unix_socket:
            connection = connect_unix_with_retry(args.unix_socket, args.connect_timeout)
        else:
            connection = connect_with_retry(args.host, args.port, args.connect_timeout)

    mask = ring_mask(bits)
    torch_backend = None
    if args.backend == "torch-u50":
        from ss_linear_scan_torch import TorchU50Backend

        torch_backend = TorchU50Backend(args.gpu, bits)

    def secure_gemm(left: np.ndarray, right: np.ndarray) -> np.ndarray:
        if torch_backend is not None:
            return torch_backend.matmul(left, right)
        return ring_matmul(left, right, bits, inner_chunk=args.inner_chunk)

    output_share = np.zeros((queries_count, output_dim), dtype=np.uint64)
    block_metrics: list[dict[str, object]] = []
    phase_bytes = {"handshake": 0, "input_sharing": 0, "beaver_open": 0, "completion": 0}
    started = time.perf_counter_ns()
    with connection:
        channel = FramedSocket(connection)
        before = channel.bytes_sent + channel.bytes_received
        if args.party == 0:
            peer_handshake = channel.recv_json()
            channel.send_json(handshake)
        else:
            channel.send_json(handshake)
            peer_handshake = channel.recv_json()
        after = channel.bytes_sent + channel.bytes_received
        phase_bytes["handshake"] += after - before
        expected_peer = 1 - args.party
        if peer_handshake.get("protocol") != handshake["protocol"]:
            raise ValueError("peer selected a different protocol")
        if int(peer_handshake.get("party", -1)) != expected_peer:
            raise ValueError("peer party identifier is invalid")
        if peer_handshake.get("run_id") != run_id:
            raise ValueError("peer selected different triple material")
        if peer_handshake.get("profile") != triple_profile:
            raise ValueError("peer selected a different lookup profile")

        for block_id, start in enumerate(range(0, physical_n, block_size)):
            block_started = time.perf_counter_ns()
            marker = reserve_triple(party_triples, block_id, run_id)
            triple_dir = party_triples / "blocks" / f"block_{block_id:06d}"
            a_share = np.asarray(
                exact_memmap(triple_dir / "A.u64", (queries_count, block_size))
            )
            b_share = np.asarray(
                exact_memmap(triple_dir / "B.u64", (block_size, output_dim))
            )
            c_share = np.asarray(
                exact_memmap(triple_dir / "C.u64", (queries_count, output_dim))
            )

            before = channel.bytes_sent + channel.bytes_received
            if args.party == 1:
                assert query_ids is not None
                clear_x = np.zeros((queries_count, block_size), dtype=np.uint64)
                in_block = (query_ids >= start) & (query_ids < start + block_size)
                rows = np.nonzero(in_block)[0]
                clear_x[rows, query_ids[rows].astype(np.int64) - start] = 1
                x0 = secure_random_ring(clear_x.shape, bits)
                x_share = (clear_x - x0) & mask
                channel.send_array(x0)
                del clear_x, x0
            else:
                x_share = channel.recv_array((queries_count, block_size))
            after = channel.bytes_sent + channel.bytes_received
            input_bytes = after - before
            phase_bytes["input_sharing"] += input_bytes

            e_share = np.asarray(table_share[start : start + block_size])
            d_local = (x_share - a_share) & mask
            f_local = (e_share - b_share) & mask
            before = channel.bytes_sent + channel.bytes_received
            if args.party == 0:
                d_peer = channel.recv_array(d_local.shape)
                f_peer = channel.recv_array(f_local.shape)
                channel.send_array(d_local)
                channel.send_array(f_local)
            else:
                channel.send_array(d_local)
                channel.send_array(f_local)
                d_peer = channel.recv_array(d_local.shape)
                f_peer = channel.recv_array(f_local.shape)
            after = channel.bytes_sent + channel.bytes_received
            open_bytes = after - before
            phase_bytes["beaver_open"] += open_bytes
            d_clear = (d_local + d_peer) & mask
            f_clear = (f_local + f_peer) & mask

            compute_started = time.perf_counter_ns()
            z_share = c_share.copy()
            d_right = (b_share + f_clear) & mask if args.party == 0 else b_share
            z_share = (
                z_share
                + secure_gemm(d_clear, d_right)
            ) & mask
            z_share = (
                z_share
                + secure_gemm(a_share, f_clear)
            ) & mask
            output_share = (output_share + z_share) & mask
            compute_ms = (time.perf_counter_ns() - compute_started) // 1_000_000
            mark_triple_complete(marker, run_id, block_id)
            block_metrics.append(
                {
                    "block_id": block_id,
                    "row_start": start,
                    "row_end": start + block_size,
                    "input_share_bytes": input_bytes,
                    "beaver_open_bytes": open_bytes,
                    "compute_ms": compute_ms,
                    "wall_ms": (time.perf_counter_ns() - block_started) // 1_000_000,
                    "triple_status": "consumed",
                }
            )

        before = channel.bytes_sent + channel.bytes_received
        completion = {"status": "complete", "party": args.party, "blocks": block_count}
        if args.party == 0:
            peer_completion = channel.recv_json()
            channel.send_json(completion)
        else:
            channel.send_json(completion)
            peer_completion = channel.recv_json()
        after = channel.bytes_sent + channel.bytes_received
        phase_bytes["completion"] += after - before
        if peer_completion.get("status") != "complete":
            raise RuntimeError("peer did not complete the lookup")
        total_sent = channel.bytes_sent
        total_received = channel.bytes_received

    args.output.parent.mkdir(parents=True, exist_ok=True)
    output_share.astype("<u8", copy=False).tofile(args.output)
    metadata = {
        "scheme": "ss-linear-scan",
        "status": "pass",
        "party": args.party,
        "run_id": run_id,
        "profile": triple_profile,
        "privacy": {
            "lookup_reconstructed": False,
            "one_hot_scale": 0,
            "fixed_point_truncation": False,
            "triple_reuse": False,
        },
        "compute_backend": (
            torch_backend.summary()
            if torch_backend is not None
            else {"name": "numpy-uint64-reference", "inner_chunk": args.inner_chunk}
        ),
        "queries_sha256": query_sha256,
        "table_share_sha256": table_manifest["files"][f"party{args.party}"]["sha256"],
        "output": {
            "file": str(args.output),
            "shape": [queries_count, output_dim],
            "dtype": "<u8",
            "bytes": args.output.stat().st_size,
            "sha256": sha256(args.output),
        },
        "communication": {
            "bytes_sent": total_sent,
            "bytes_received": total_received,
            "phase_bytes_sent_plus_received": phase_bytes,
            "synchronization_phases": {
                "handshake": 1,
                "input_sharing": block_count,
                "beaver_open": block_count,
                "completion": 1,
                "total": 2 * block_count + 2,
            },
        },
        "blocks": block_metrics,
        "wall_ms": (time.perf_counter_ns() - started) // 1_000_000,
    }
    atomic_write_json(args.metadata, metadata)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
