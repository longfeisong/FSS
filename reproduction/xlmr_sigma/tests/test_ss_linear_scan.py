#!/usr/bin/env python3

from __future__ import annotations

import json
import socket
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPT_DIR))

from ss_linear_scan_common import reserve_triple, ring_mask, ring_matmul  # noqa: E402
from run_paired_lookup_benchmark import (  # noqa: E402
    balanced_schedule,
    parse_gpu_assignment,
    summarize,
)


class RingArithmeticTests(unittest.TestCase):
    def test_uint64_matmul_matches_python_integer_ring(self) -> None:
        bits = 50
        mask = int(ring_mask(bits))
        left = np.array(
            [[mask, mask - 3, 17], [1 << 49, (1 << 49) + 9, mask]],
            dtype=np.uint64,
        )
        right = np.array(
            [[mask - 2, 7], [11, mask], [(1 << 49) + 1, 13]],
            dtype=np.uint64,
        )
        actual = ring_matmul(left, right, bits, inner_chunk=1)
        expected = np.empty(actual.shape, dtype=np.uint64)
        for row in range(left.shape[0]):
            for column in range(right.shape[1]):
                expected[row, column] = sum(
                    int(left[row, inner]) * int(right[inner, column])
                    for inner in range(left.shape[1])
                ) & mask
        np.testing.assert_array_equal(actual, expected)

    def test_triple_reservation_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            party_dir = Path(temporary)
            reserve_triple(party_dir, 0, "fresh-run")
            with self.assertRaisesRegex(RuntimeError, "already reserved"):
                reserve_triple(party_dir, 0, "fresh-run")

    def test_a100_tensorcore_backend_matches_ring_reference(self) -> None:
        try:
            import torch
        except ImportError:
            self.skipTest("PyTorch is not installed")
        if not torch.cuda.is_available():
            self.skipTest("CUDA is not available")
        from ss_linear_scan_torch import TorchU50Backend

        rng = np.random.default_rng(20260720)
        left = rng.integers(0, 1 << 50, size=(8, 17), dtype=np.uint64)
        right = rng.integers(0, 1 << 50, size=(17, 5), dtype=np.uint64)
        actual = TorchU50Backend(0).matmul(left, right)
        expected = ring_matmul(left, right, 50)
        np.testing.assert_array_equal(actual, expected)


class EndToEndTests(unittest.TestCase):
    def run_command(
        self, arguments: list[str], *, environment: dict[str, str] | None = None
    ) -> subprocess.CompletedProcess[str]:
        completed = subprocess.run(
            arguments,
            check=False,
            text=True,
            capture_output=True,
            env=environment,
            timeout=60,
        )
        if completed.returncode:
            self.fail(
                f"command failed ({completed.returncode}): {arguments}\n"
                f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
            )
        return completed

    def test_padded_block_lookup_with_negative_values(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            logical_n, query_count, output_dim, block_size = 17, 4, 5, 8
            table = (
                np.arange(logical_n * output_dim, dtype=np.int16).reshape(
                    logical_n, output_dim
                )
                - 41
            )
            queries = np.array([16, 0, 7, 16], dtype="<u4")
            table_path = root / "table.i16"
            query_path = root / "queries.u32"
            table.astype("<i2").tofile(table_path)
            queries.tofile(query_path)
            table_shares = root / "table-shares"
            triples = root / "triples"
            run_dir = root / "run"

            self.run_command(
                [
                    sys.executable,
                    str(SCRIPT_DIR / "prepare_ss_table_shares.py"),
                    "--table",
                    str(table_path),
                    "--output-dir",
                    str(table_shares),
                    "--logical-n",
                    str(logical_n),
                    "--output-dim",
                    str(output_dim),
                    "--block-size",
                    str(block_size),
                ]
            )
            self.run_command(
                [
                    sys.executable,
                    str(SCRIPT_DIR / "generate_ss_beaver_triples.py"),
                    "--output-dir",
                    str(triples),
                    "--logical-n",
                    str(logical_n),
                    "--queries",
                    str(query_count),
                    "--output-dim",
                    str(output_dim),
                    "--block-size",
                    str(block_size),
                ]
            )
            run_dir.mkdir()
            endpoint0, endpoint1 = socket.socketpair()
            base = [sys.executable, str(SCRIPT_DIR / "ss_linear_scan_party.py")]
            process0 = subprocess.Popen(
                base
                + [
                    "--party",
                    "0",
                    "--socket-fd",
                    str(endpoint0.fileno()),
                    "--table-shares",
                    str(table_shares),
                    "--triples",
                    str(triples),
                    "--output",
                    str(run_dir / "P0-share.u64"),
                    "--metadata",
                    str(run_dir / "P0-metadata.json"),
                ],
                pass_fds=(endpoint0.fileno(),),
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            process1 = subprocess.Popen(
                base
                + [
                    "--party",
                    "1",
                    "--socket-fd",
                    str(endpoint1.fileno()),
                    "--table-shares",
                    str(table_shares),
                    "--triples",
                    str(triples),
                    "--queries",
                    str(query_path),
                    "--output",
                    str(run_dir / "P1-share.u64"),
                    "--metadata",
                    str(run_dir / "P1-metadata.json"),
                ],
                pass_fds=(endpoint1.fileno(),),
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            endpoint0.close()
            endpoint1.close()
            stdout0, stderr0 = process0.communicate(timeout=30)
            stdout1, stderr1 = process1.communicate(timeout=30)
            self.assertEqual(process0.returncode, 0, f"{stdout0}\n{stderr0}")
            self.assertEqual(process1.returncode, 0, f"{stdout1}\n{stderr1}")
            self.run_command(
                [
                    sys.executable,
                    str(SCRIPT_DIR / "verify_ss_linear_scan.py"),
                    "--table",
                    str(table_path),
                    "--queries",
                    str(query_path),
                    "--share0",
                    str(run_dir / "P0-share.u64"),
                    "--share1",
                    str(run_dir / "P1-share.u64"),
                    "--logical-n",
                    str(logical_n),
                    "--queries-count",
                    str(query_count),
                    "--output-dim",
                    str(output_dim),
                    "--json-output",
                    str(run_dir / "verification.json"),
                ]
            )
            self.run_command(
                [
                    sys.executable,
                    str(SCRIPT_DIR / "summarize_ss_linear_scan_run.py"),
                    "--run-dir",
                    str(run_dir),
                    "--table-shares",
                    str(table_shares),
                    "--triples",
                    str(triples),
                    "--queries",
                    str(query_path),
                    "--wall-ms",
                    "1",
                ]
            )

            verification = json.loads((run_dir / "verification.json").read_text())
            self.assertEqual(verification["status"], "pass")
            self.assertEqual(verification["mismatches"], 0)
            summary = json.loads((run_dir / "metadata.json").read_text())
            self.assertEqual(summary["status"], "pass")
            self.assertFalse(summary["bridge_compatible"])
            self.assertGreater(summary["communication"]["two_party_bytes_sent"], 0)
            self.assertTrue(
                summary["preprocessing"]["one_time_material_consumed"]
            )
            share0 = np.fromfile(run_dir / "P0-share.u64", dtype="<u8").reshape(
                query_count, output_dim
            )
            share1 = np.fromfile(run_dir / "P1-share.u64", dtype="<u8").reshape(
                query_count, output_dim
            )
            reconstructed = (share0 + share1) & ring_mask(50)
            expected = table[queries].astype(np.int64).view(np.uint64) & ring_mask(50)
            np.testing.assert_array_equal(reconstructed, expected)
            self.assertFalse(np.array_equal(share0, expected))
            self.assertFalse(np.array_equal(share1, expected))
            for party in (0, 1):
                for block_id in range(3):
                    marker = (
                        triples
                        / f"party{party}"
                        / "consumed"
                        / f"block_{block_id:06d}.json"
                    )
                    self.assertEqual(json.loads(marker.read_text())["status"], "consumed")


class PairedBenchmarkTests(unittest.TestCase):
    def test_gpu_assignment_accepts_single_card_colocation(self) -> None:
        self.assertEqual(parse_gpu_assignment("1"), (1, 1))
        self.assertEqual(parse_gpu_assignment("1,4"), (1, 4))
        self.assertEqual(parse_gpu_assignment("1,1"), (1, 1))
        self.assertIsNone(parse_gpu_assignment(None))
        with self.assertRaisesRegex(ValueError, "one index or a P0,P1 pair"):
            parse_gpu_assignment("0,1,2")
        with self.assertRaisesRegex(ValueError, "non-negative"):
            parse_gpu_assignment("-1")

    def test_schedule_is_balanced_and_reproducible(self) -> None:
        first = balanced_schedule(1, 7, 99)
        second = balanced_schedule(1, 7, 99)
        self.assertEqual(first, second)
        measured = [value for value in first if not value["warmup"]]
        starts = [value["order"][0] for value in measured]
        self.assertLessEqual(abs(starts.count("fable") - starts.count("ss-linear-scan")), 1)

    def test_summary_uses_only_clean_complete_pairs(self) -> None:
        records = []
        for pair_id, fable_ms, ss_ms, contaminated in (
            (0, 200, 100, False),
            (1, 330, 110, False),
            (2, 1000, 100, True),
        ):
            for method, wall_ms in (("fable", fable_ms), ("ss-linear-scan", ss_ms)):
                records.append(
                    {
                        "pair_id": pair_id,
                        "warmup": False,
                        "method": method,
                        "status": "pass",
                        "contaminated": contaminated and method == "fable",
                        "mismatches": 0,
                        "wall_ms": wall_ms,
                    }
                )
        result = summarize(records, 7)
        self.assertEqual(result["clean_pairs"], 2)
        self.assertEqual(result["paired_speedup"]["median"], 2.5)


if __name__ == "__main__":
    unittest.main()
