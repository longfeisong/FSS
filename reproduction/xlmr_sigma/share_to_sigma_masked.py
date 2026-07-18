#!/usr/bin/env python3
"""Convert one party's FABLE arithmetic share to a public SIGMA masked value.

P0 receives then sends; P1 sends then receives.  Neither message exposes the
embedding because x_i is one-time padded by the dealer-provided r_i.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import socket
import struct
import time
from pathlib import Path

import numpy as np


HEADER = struct.Struct("!Q")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--party", type=int, choices=(0, 1), required=True)
    parser.add_argument("--share", type=Path, required=True)
    parser.add_argument("--mask-share", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18820)
    parser.add_argument("--bitwidth", type=int, default=50)
    parser.add_argument("--count", type=int, default=512 * 768)
    return parser.parse_args()


def load_exact(path: Path, count: int) -> np.ndarray:
    values = np.fromfile(path, dtype="<u8")
    if values.size != count:
        raise ValueError(f"{path}: expected {count} uint64 values, got {values.size}")
    return values


def recv_exact(sock: socket.socket, size: int) -> bytes:
    result = bytearray(size)
    view = memoryview(result)
    offset = 0
    while offset < size:
        received = sock.recv_into(view[offset:])
        if received == 0:
            raise ConnectionError("peer disconnected during masked-share exchange")
        offset += received
    return bytes(result)


def send_array(sock: socket.socket, values: np.ndarray) -> None:
    payload = values.astype("<u8", copy=False).tobytes(order="C")
    sock.sendall(HEADER.pack(len(payload)))
    sock.sendall(payload)


def recv_array(sock: socket.socket, count: int) -> np.ndarray:
    (size,) = HEADER.unpack(recv_exact(sock, HEADER.size))
    expected = count * np.dtype("<u8").itemsize
    if size != expected:
        raise ValueError(f"peer payload has {size} bytes; expected {expected}")
    return np.frombuffer(recv_exact(sock, size), dtype="<u8").copy()


def connect_with_retry(host: str, port: int) -> socket.socket:
    deadline = time.monotonic() + 30
    while True:
        try:
            return socket.create_connection((host, port), timeout=30)
        except ConnectionRefusedError:
            if time.monotonic() >= deadline:
                raise
            time.sleep(0.1)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    args = parse_args()
    ring_mask = np.uint64((1 << args.bitwidth) - 1)
    share = load_exact(args.share, args.count)
    mask_share = load_exact(args.mask_share, args.count)
    local_masked_share = (share + mask_share) & ring_mask

    started = time.perf_counter_ns()
    if args.party == 0:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
            listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            listener.bind((args.host, args.port))
            listener.listen(1)
            conn, _ = listener.accept()
            with conn:
                peer_masked_share = recv_array(conn, args.count)
                send_array(conn, local_masked_share)
    else:
        with connect_with_retry(args.host, args.port) as conn:
            send_array(conn, local_masked_share)
            peer_masked_share = recv_array(conn, args.count)

    sigma_masked = (local_masked_share + peer_masked_share) & ring_mask
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sigma_masked.astype("<u8", copy=False).tofile(args.output)
    elapsed_us = (time.perf_counter_ns() - started) // 1000
    result = {
        "status": "pass",
        "party": args.party,
        "values": args.count,
        "bitwidth": args.bitwidth,
        "bytes_sent": args.count * 8 + HEADER.size,
        "bytes_received": args.count * 8 + HEADER.size,
        "elapsed_us": elapsed_us,
        "output": str(args.output),
        "output_sha256": sha256(args.output),
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
