# FSS secure Transformer experiments

This repository contains reproducible experiments for secure Transformer
inference with function secret sharing and confidential lookup tables.

The current branch establishes the SIGMA baseline on a shared NVIDIA A100
server. It includes the exact source revision, compatibility patches, a safe
two-party local launcher, automatic result collection, and the first successful
BERT-Tiny smoke-test result.

## Current status

- SIGMA BERT-Tiny (`2` blocks, sequence length `128`) builds and runs as two
  local parties on two dynamically selected A100 GPUs.
- The current artifact uses zero input and zero weights. It validates the
  end-to-end protocol path, but is not yet a numerical-correctness experiment.
- Runs are archived as raw P0/P1 statistics and appended to CSV/Markdown tables.
- FABLE integration is intentionally deferred to a separate branch.

See [`reproduction/sigma/README.md`](reproduction/sigma/README.md) for setup,
execution, result interpretation, and the planned comparison methodology.

## Repository layout

```text
reproduction/sigma/
├── patches/        # Minimal patches applied to the official SIGMA artifact
├── scripts/        # Local two-party launcher and result recorder
└── results/        # Append-only ledger and archived raw statistics
```

Third-party repositories, papers, binaries, build trees, and conda environments
are deliberately excluded from version control.
