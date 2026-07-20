# FSS secure Transformer experiments

This repository contains reproducible experiments for secure Transformer
inference with function secret sharing and confidential lookup tables.

The `fable-integration` branch implements a private XLM-R embedding lookup with
the official FABLE artifact and feeds its secret output into real SIGMA encoder
blocks without revealing the embedding.

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
- The XLM-R/FABLE route fixes the interface at 512 private lookups x 768
  dimensions, int16 scale 12, and converts the GC output into 50-bit arithmetic
  shares.
- A dealer-masked handoff converts those shares into SIGMA inputs without
  reconstructing the embedding; real XLM-R weights are also masked rather than
  loaded as public clear parameters.
- The complete 24-chunk, 512×768 private embedding passes with zero mismatches
  and feeds a real XLM-R block; the masked output is bit-identical to a zero-mask
  SIGMA correctness oracle.
- The secure SS-LinearScan comparison baseline now implements blockwise
  one-hot/table secret sharing, fresh Beaver matrix triples, fail-closed triple
  consumption, exact A100 Tensor Core arithmetic in `Z_(2^50)`, and the same
  share-to-SIGMA output contract. The full 512×768 lookup and all four XLM-R
  Layer 0 sequences pass, with output hashes identical to FABLE+SIGMA.

See [`reproduction/sigma/README.md`](reproduction/sigma/README.md) for setup,
execution, and SIGMA results. See
[`reproduction/fable/README.md`](reproduction/fable/README.md) for the FABLE
patches, build/run commands, implementation limits, and comparison result.
See [`reproduction/xlmr_sigma/README.md`](reproduction/xlmr_sigma/README.md)
for the new XLM-R Embedding -> SIGMA route and its security boundary.

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
reproduction/xlmr_sigma/
├── config.json        # Pinned hybrid experiment profile
├── download_model.sh  # Pinned Hugging Face model download
├── fable/xlmr_embedding.cpp # private vector lookup + GC-to-ring shares
├── run_fable_xlmr.sh  # complete 24-chunk confidential embedding
├── run_sigma_input_bridge.sh # secure shares-to-SIGMA conversion
├── sigma_xlmr.cu      # masked real-weight SIGMA encoder driver
├── run_sigma_xlmr.sh  # validated two-A100 execution
└── results/           # Compact correctness ledger
```

The GPU-MPC source is imported from EzPC commit `f24bf3e`; FABLE is pinned to
`be6e73220e25af6d699c532b38e9526701a68280`. CUTLASS, SEAL and other large
external dependencies remain pinned Git submodules. Papers, binaries, build
trees, conda environments, and Git caches are excluded.
