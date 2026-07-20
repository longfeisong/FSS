#!/usr/bin/env python3
"""NVIDIA Ampere tensor-core backend for exact matrix multiplication in Z_(2^50).

Each 50-bit ring element is decomposed into eight non-negative base-128 limbs.
For limb pairs whose combined shift is below 50 bits, ``torch._int_mm`` performs
signed int8 x int8 -> int32 GEMM.  Digits are in [0, 127], and a 4096-row block
keeps every int32 dot product below 2^31.  Shifted partial products are summed
and masked to 50 bits, yielding the same result as uint64 GEMM modulo 2^50.
"""

from __future__ import annotations

import math
import time
from typing import Any

import numpy as np


class TorchU50Backend:
    name = "torch-int8-tensorcore-u50"
    limb_bits = 7

    def __init__(self, gpu: int, ring_bits: int = 50):
        if ring_bits != 50:
            raise ValueError("the tensor-core backend currently implements ring_bits=50")
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("PyTorch cannot access CUDA")
        if gpu < 0 or gpu >= torch.cuda.device_count():
            raise ValueError(f"GPU {gpu} is outside [0, {torch.cuda.device_count()})")
        self.torch = torch
        self.device = torch.device(f"cuda:{gpu}")
        torch.cuda.set_device(self.device)
        self.gpu = gpu
        self.ring_bits = ring_bits
        self.limbs = math.ceil(ring_bits / self.limb_bits)
        self.calls = 0
        self.total_wall_ms = 0.0
        self.last_call: dict[str, Any] = {}

    @staticmethod
    def _padded(value: int, multiple: int, minimum: int = 0) -> int:
        return max(minimum, ((value + multiple - 1) // multiple) * multiple)

    def _digits(self, values: np.ndarray, shape: tuple[int, int]) -> list[Any]:
        torch = self.torch
        if values.shape == shape:
            padded = np.asarray(values, dtype=np.uint64)
        else:
            padded = np.zeros(shape, dtype=np.uint64)
            padded[: values.shape[0], : values.shape[1]] = values
        result = []
        for limb in range(self.limbs):
            digit = ((padded >> (limb * self.limb_bits)) & 0x7F).astype(
                np.int8, copy=False
            )
            result.append(
                torch.from_numpy(np.ascontiguousarray(digit)).to(
                    self.device, non_blocking=False
                )
            )
        return result

    def matmul(self, left: np.ndarray, right: np.ndarray) -> np.ndarray:
        torch = self.torch
        left = np.asarray(left, dtype=np.uint64)
        right = np.asarray(right, dtype=np.uint64)
        if left.ndim != 2 or right.ndim != 2 or left.shape[1] != right.shape[0]:
            raise ValueError(f"incompatible matmul shapes: {left.shape}, {right.shape}")
        if left.shape[1] * 127 * 127 > np.iinfo(np.int32).max:
            raise OverflowError("base-128 int32 accumulation would overflow")

        rows, common, columns = left.shape[0], left.shape[1], right.shape[1]
        padded_rows = self._padded(rows, 8, 32)
        padded_common = self._padded(common, 8)
        padded_columns = self._padded(columns, 8, 8)
        started = time.perf_counter_ns()
        left_digits = self._digits(left, (padded_rows, padded_common))
        right_digits = self._digits(right, (padded_common, padded_columns))
        torch.cuda.synchronize(self.device)
        transfer_ms = (time.perf_counter_ns() - started) / 1_000_000

        gemm_started = time.perf_counter_ns()
        output = torch.zeros(
            (padded_rows, padded_columns), dtype=torch.int64, device=self.device
        )
        gemm_calls = 0
        max_sum = (self.ring_bits - 1) // self.limb_bits
        for limb_sum in range(max_sum + 1):
            shift = limb_sum * self.limb_bits
            for left_limb in range(limb_sum + 1):
                right_limb = limb_sum - left_limb
                if left_limb >= self.limbs or right_limb >= self.limbs:
                    continue
                partial = torch._int_mm(
                    left_digits[left_limb], right_digits[right_limb]
                )
                output.add_(partial.to(torch.int64).bitwise_left_shift(shift))
                gemm_calls += 1
        output.bitwise_and_((1 << self.ring_bits) - 1)
        torch.cuda.synchronize(self.device)
        gemm_ms = (time.perf_counter_ns() - gemm_started) / 1_000_000
        result = (
            output[:rows, :columns]
            .cpu()
            .numpy()
            .astype(np.uint64, copy=False)
            .copy()
        )
        torch.cuda.synchronize(self.device)
        wall_ms = (time.perf_counter_ns() - started) / 1_000_000
        self.calls += 1
        self.total_wall_ms += wall_ms
        self.last_call = {
            "input_shape_left": list(left.shape),
            "input_shape_right": list(right.shape),
            "padded_shape": [padded_rows, padded_common, padded_columns],
            "limb_bits": self.limb_bits,
            "limbs": self.limbs,
            "int8_gemm_calls": gemm_calls,
            "host_to_device_ms": transfer_ms,
            "gemm_ms": gemm_ms,
            "wall_ms": wall_ms,
        }
        del left_digits, right_digits, output
        return result

    def summary(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "gpu": self.gpu,
            "device": self.torch.cuda.get_device_name(self.device),
            "torch_version": self.torch.__version__,
            "torch_cuda_version": self.torch.version.cuda,
            "calls": self.calls,
            "total_wall_ms": self.total_wall_ms,
            "last_call": self.last_call,
        }
