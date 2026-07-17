# SIGMA GELU LUT → FABLE experiment

This experiment replaces the **256-entry GELU correction lookup** used by
SIGMA with FABLE's confidential batched lookup protocol. It targets one
BERT-Tiny Transformer Block:

- hidden size: 128
- attention heads: 2
- FFN hidden size: 512
- sequence length: 128
- GELU lookups: `128 × 512 = 65,536`
- SIGMA fixed-point scale: 12; ring bitwidth: 37

## What is replaced

SIGMA computes GELU as `ReLU(x) - correction[x]`. The added FABLE LUT uses the
same integer table generation as SIGMA's `reluSubGelu`: an 8-bit index with six
fractional bits and a 37-bit container for scale-12 output.

This branch currently validates and benchmarks the confidential LUT component.
It does **not** yet route live GPU tensors from the complete SIGMA executable
through FABLE. The live bridge still needs the SIGMA masked-value/dealer-mask
conversion at the GPU/GC boundary; the standalone benchmark uses FABLE's
original BOB-owned secret-query interface.

## FABLE implementation limits

The official BatchPIR parameter table supports minimum log database size 16,
so the 256-entry SIGMA table is padded to 65,536 entries. Queries are restricted
to valid indices 0–255. BatchPIR also has 8,192 cuckoo buckets; a 65,536-query
single batch fails because deduplication replaces duplicates with unique dummy
queries. The runner therefore uses 16 independent batches of 4,096.

Two compatibility fixes are included:

- use only active PIR task groups when the subbucket has fewer than 128 columns;
- include `fmt/format.h` for current `fmt` packages.

## Build and run

```bash
conda activate sigma-fable
./reproduction/fable/build_gelu.sh
./reproduction/fable/run_gelu_block.sh 16 4096 16
```

The runner automatically cleans stale instances of this exact FABLE binary,
runs both local parties, verifies every chunk, saves raw logs, and appends the
summary to `results/runs.csv`.

## First A100-server result

The FABLE portion is CPU/network based; the A100 remains relevant for the
surrounding SIGMA Block, not for this isolated lookup.

| Measurement | SIGMA baseline | FABLE replacement |
|---|---:|---:|
| Scope | GELU, 2 blocks | GELU LUT, 1 block |
| Queries per block | 65,536 | 65,536 |
| Online/protocol time | 207.078 ms total; ≈103.539 ms/block | 75,437 ms/block |
| GELU/LUT communication | 3,473,408 B total; ≈1,736,704 B/block | 19,372,844,388 B/block |
| Correctness | zero-input SIGMA smoke test | all 16 chunks, zero lookup error |

The two correctness tests are not equivalent: the SIGMA baseline used zero
weights and inputs, while the FABLE benchmark checked random valid GELU indices.
Timing is provisional because both measurements were made on a shared server,
and the original SIGMA run had high utilization on one selected GPU.
