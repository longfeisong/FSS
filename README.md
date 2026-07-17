# FSS secure Transformer experiments

This repository contains reproducible experiments for secure Transformer
inference with function secret sharing and confidential lookup tables.

The baseline branch establishes SIGMA on a shared NVIDIA A100 server. The
`fable-integration` branch additionally evaluates replacement of SIGMA's GELU
correction lookup with the official FABLE implementation.

## Current status

- SIGMA BERT-Tiny (`2` blocks, sequence length `128`) builds and runs as two
  local parties on two dynamically selected A100 GPUs.
- The current artifact uses zero input and zero weights. It validates the
  end-to-end protocol path, but is not yet a numerical-correctness experiment.
- Runs are archived as raw P0/P1 statistics and appended to CSV/Markdown tables.
- FABLE's confidential LUT now reproduces SIGMA's 8-bit GELU correction table.
- One BERT-Tiny Block is evaluated as 16 FABLE batches × 4,096 queries; every
  batch passes with zero lookup error.
- This is currently a component replacement benchmark. The live GPU tensor ↔
  FABLE GC share bridge is the next integration step.

See [`reproduction/sigma/README.md`](reproduction/sigma/README.md) for setup,
execution, and SIGMA results. See
[`reproduction/fable/README.md`](reproduction/fable/README.md) for the FABLE
patches, build/run commands, implementation limits, and comparison result.

## Repository layout

```text
src/GPU-MPC/             # Complete tracked GPU-MPC source snapshot
src/FABLE/               # Official FABLE revision pinned as a submodule
reproduction/sigma/
├── patches/        # Minimal patches applied to the official SIGMA artifact
├── scripts/        # Local two-party launcher and result recorder
├── output/         # Verbatim output tree produced by the SIGMA artifact
└── results/        # Append-only ledger and archived raw statistics
reproduction/fable/
├── patches/        # SIGMA GELU LUT and BatchPIR compatibility patches
├── results/        # FABLE block-level CSV/Markdown ledger
├── build_gelu.sh   # Reproducible dependency/configuration build
└── run_gelu_block.sh # Auto-cleaning two-party chunk runner
```

The GPU-MPC source is imported from EzPC commit `f24bf3e`; FABLE is pinned to
`be6e73220e25af6d699c532b38e9526701a68280`. CUTLASS, SEAL and other large
external dependencies remain pinned Git submodules. Papers, binaries, build
trees, conda environments, and Git caches are excluded.
