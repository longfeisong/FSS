#!/usr/bin/env python3
"""Shared utilities for the dealer-assisted SS-LinearScan baseline.

All arithmetic is performed in Z_(2^ring_bits).  NumPy's uint64 matrix
multiplication wraps modulo 2^64; reducing the result modulo 2^ring_bits is
therefore exact for ring_bits <= 64.
"""

from __future__ import annotations

import hashlib
import json
import os
import socket
import struct
import time
from pathlib import Path
from typing import Any

import numpy as np


FRAME_HEADER = struct.Struct("!Q")


def ring_mask(ring_bits: int) -> np.uint64:
    if not 1 <= ring_bits <= 63:
        raise ValueError("ring_bits must be in [1, 63]")
    return np.uint64((1 << ring_bits) - 1)


def secure_random_ring(shape: tuple[int, ...], ring_bits: int) -> np.ndarray:
    """Return OS-random uint64 ring elements.

    Rejection sampling is unnecessary because the modulus is a power of two.
    Masking uniform 64-bit strings gives uniform elements in Z_(2^k).
    """
    count = int(np.prod(shape, dtype=np.int64))
    values = np.frombuffer(os.urandom(count * 8), dtype="<u8").copy()
    values &= ring_mask(ring_bits)
    return values.reshape(shape)


def encode_signed(values: np.ndarray, ring_bits: int) -> np.ndarray:
    signed = np.asarray(values, dtype=np.int64)
    return signed.view(np.uint64) & ring_mask(ring_bits)


def ring_matmul(
    left: np.ndarray,
    right: np.ndarray,
    ring_bits: int,
    *,
    inner_chunk: int = 0,
) -> np.ndarray:
    """Matrix multiply exactly modulo 2^ring_bits without fixed-point truncation."""
    left = np.asarray(left, dtype=np.uint64)
    right = np.asarray(right, dtype=np.uint64)
    if left.ndim != 2 or right.ndim != 2 or left.shape[1] != right.shape[0]:
        raise ValueError(f"incompatible matmul shapes: {left.shape} and {right.shape}")
    mask = ring_mask(ring_bits)
    common = left.shape[1]
    if inner_chunk <= 0 or inner_chunk >= common:
        return (left @ right) & mask
    output = np.zeros((left.shape[0], right.shape[1]), dtype=np.uint64)
    for start in range(0, common, inner_chunk):
        end = min(start + inner_chunk, common)
        output = (output + left[:, start:end] @ right[start:end, :]) & mask
    return output


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_array(path: Path, values: np.ndarray) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    contiguous = np.ascontiguousarray(values, dtype="<u8")
    contiguous.tofile(path)
    return {
        "file": path.name,
        "dtype": "<u8",
        "shape": list(contiguous.shape),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def exact_memmap(path: Path, shape: tuple[int, ...]) -> np.memmap:
    expected = int(np.prod(shape, dtype=np.int64)) * 8
    try:
        actual = path.stat().st_size
    except FileNotFoundError as error:
        raise FileNotFoundError(f"missing ring artifact: {path}") from error
    if actual != expected:
        raise ValueError(f"{path}: expected {expected} bytes, got {actual}")
    return np.memmap(path, mode="r", dtype="<u8", shape=shape)


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def reserve_triple(party_dir: Path, block_id: int, run_id: str) -> Path:
    """Irreversibly reserve a triple before reading it.

    O_EXCL makes concurrent or crash-retry reuse fail closed.  A failed run must
    use freshly generated preprocessing material.
    """
    marker = party_dir / "consumed" / f"block_{block_id:06d}.json"
    marker.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "status": "reserved",
        "run_id": run_id,
        "block_id": block_id,
        "pid": os.getpid(),
        "reserved_unix_ns": time.time_ns(),
    }
    try:
        descriptor = os.open(marker, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as error:
        raise RuntimeError(
            f"triple block {block_id} for party directory {party_dir} was already "
            "reserved; generate a fresh triple run"
        ) from error
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    return marker


def mark_triple_complete(marker: Path, run_id: str, block_id: int) -> None:
    atomic_write_json(
        marker,
        {
            "status": "consumed",
            "run_id": run_id,
            "block_id": block_id,
            "pid": os.getpid(),
            "completed_unix_ns": time.time_ns(),
        },
    )


def recv_exact(sock: socket.socket, size: int) -> bytes:
    result = bytearray(size)
    view = memoryview(result)
    offset = 0
    while offset < size:
        received = sock.recv_into(view[offset:])
        if received == 0:
            raise ConnectionError("peer disconnected")
        offset += received
    return bytes(result)


class FramedSocket:
    """Length-framed socket with explicit byte counters."""

    def __init__(self, sock: socket.socket):
        self.sock = sock
        self.bytes_sent = 0
        self.bytes_received = 0

    def send_bytes(self, payload: bytes | memoryview) -> None:
        header = FRAME_HEADER.pack(len(payload))
        self.sock.sendall(header)
        self.sock.sendall(payload)
        self.bytes_sent += len(header) + len(payload)

    def recv_bytes(self, expected_size: int | None = None) -> bytes:
        header = recv_exact(self.sock, FRAME_HEADER.size)
        (size,) = FRAME_HEADER.unpack(header)
        if expected_size is not None and size != expected_size:
            raise ValueError(f"peer frame has {size} bytes; expected {expected_size}")
        payload = recv_exact(self.sock, size)
        self.bytes_received += len(header) + size
        return payload

    def send_json(self, value: dict[str, Any]) -> None:
        self.send_bytes(json.dumps(value, sort_keys=True).encode("utf-8"))

    def recv_json(self) -> dict[str, Any]:
        value = json.loads(self.recv_bytes().decode("utf-8"))
        if not isinstance(value, dict):
            raise ValueError("peer handshake is not an object")
        return value

    def send_array(self, values: np.ndarray) -> None:
        contiguous = np.ascontiguousarray(values, dtype="<u8")
        self.send_bytes(memoryview(contiguous).cast("B"))

    def recv_array(self, shape: tuple[int, ...]) -> np.ndarray:
        size = int(np.prod(shape, dtype=np.int64)) * 8
        return np.frombuffer(self.recv_bytes(size), dtype="<u8").copy().reshape(shape)


class InheritedFdStream:
    """Socket-like wrapper that performs I/O without creating a socket object."""

    def __init__(self, descriptor: int):
        self.descriptor = descriptor

    def sendall(self, payload: bytes | memoryview) -> None:
        view = memoryview(payload).cast("B")
        offset = 0
        while offset < len(view):
            written = os.write(self.descriptor, view[offset:])
            if written == 0:
                raise ConnectionError("peer disconnected while sending")
            offset += written

    def recv_into(self, buffer: memoryview) -> int:
        return os.readv(self.descriptor, [buffer])

    def close(self) -> None:
        if self.descriptor >= 0:
            os.close(self.descriptor)
            self.descriptor = -1

    def __enter__(self) -> "InheritedFdStream":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def connect_with_retry(host: str, port: int, timeout_seconds: float = 30) -> socket.socket:
    deadline = time.monotonic() + timeout_seconds
    while True:
        try:
            sock = socket.create_connection((host, port), timeout=timeout_seconds)
            sock.settimeout(None)
            return sock
        except ConnectionRefusedError:
            if time.monotonic() >= deadline:
                raise
            time.sleep(0.1)


def connect_unix_with_retry(path: Path, timeout_seconds: float = 30) -> socket.socket:
    deadline = time.monotonic() + timeout_seconds
    while True:
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(timeout_seconds)
        try:
            sock.connect(str(path))
            sock.settimeout(None)
            return sock
        except (FileNotFoundError, ConnectionRefusedError):
            sock.close()
            if time.monotonic() >= deadline:
                raise
            time.sleep(0.1)


def validate_profile(profile: dict[str, Any]) -> tuple[int, int, int, int, int, int]:
    required = (
        "logical_n",
        "physical_n",
        "queries",
        "output_dim",
        "block_size",
        "ring_bits",
    )
    missing = [key for key in required if key not in profile]
    if missing:
        raise ValueError(f"profile is missing {missing}")
    logical_n, physical_n, queries, output_dim, block_size, bits = (
        int(profile[key]) for key in required
    )
    if logical_n <= 0 or queries <= 0 or output_dim <= 0 or block_size <= 0:
        raise ValueError("profile dimensions must be positive")
    if physical_n < logical_n or physical_n % block_size:
        raise ValueError("physical_n must pad logical_n to a whole block")
    ring_mask(bits)
    return logical_n, physical_n, queries, output_dim, block_size, bits
