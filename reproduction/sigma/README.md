# SIGMA baseline reproduction

## Scope

This baseline uses the official Microsoft Research EzPC implementation of
[SIGMA](https://github.com/mpc-msri/EzPC/tree/master/GPU-MPC/experiments/sigma)
at commit `f24bf3e`.

The tested configuration is:

| Field | Value |
|---|---|
| GPU | NVIDIA A100-SXM4-80GB (`sm_80`) |
| CUDA | 12.1 |
| Model selector | `bert-tiny` |
| Transformer blocks | 2 |
| Heads / hidden size | 2 / 128 |
| Sequence length | 128 |
| Fixed-point scale / bit width | 12 / 37 |
| Parties | P0 and P1 on the same host, separate GPUs |
| Inputs and weights | all zero (smoke test only) |

## Source setup

```bash
git clone --recursive https://github.com/mpc-msri/EzPC.git third_party/EzPC
git -C third_party/EzPC checkout f24bf3e
git -C third_party/EzPC submodule update --init --recursive

git -C third_party/EzPC apply \
  ../../reproduction/sigma/patches/gpu-memory-pool.patch
git -C third_party/EzPC/GPU-MPC/ext/sytorch/ext/sci/extern/SEAL apply \
  ../../../../../../../../../reproduction/sigma/patches/seal-locks-mutex.patch
```

The first patch makes SIGMA's 40 GiB CUDA pool warm-up configurable with
`SIGMA_GPU_POOL_GB`; setting it to zero skips only the warm-up allocation. The
second patch supplies the `<mutex>` include required to compile the pinned old
SEAL source with the current compiler.

Build using the official instructions with these A100-specific variables:

```bash
export CUDA_HOME=/usr/local/cuda-12.1
export CUDA_VERSION=12.1
export GPU_ARCH=80
export CUDACXX=/usr/local/cuda-12.1/bin/nvcc

cd third_party/EzPC/GPU-MPC
make sigma
```

Copy the local experiment helpers next to the generated `sigma` executable:

```bash
cd /home/slf/LLM/FSS
cp reproduction/sigma/scripts/run_local_auto.sh \
  third_party/EzPC/GPU-MPC/experiments/sigma/
cp reproduction/sigma/scripts/record_result.py \
  third_party/EzPC/GPU-MPC/experiments/sigma/
```

## Run and record

From `third_party/EzPC/GPU-MPC/experiments/sigma`:

```bash
SIGMA_ALLOWED_GPUS=2,7 \
SIGMA_GPU_POOL_GB=0 \
SIGMA_MIN_FREE_MIB=6000 \
SIGMA_RESULTS_DIR=/home/slf/LLM/FSS/reproduction/sigma/results \
./run_local_auto.sh bert-tiny 128 4
```

The launcher:

- selects at most two GPUs only from `SIGMA_ALLOWED_GPUS`;
- records utilization at launch and flags contaminated timing runs;
- starts P0, waits for port `42002`, then starts P1;
- cleans only stale SIGMA processes from this exact workspace;
- archives raw statistics and appends a row only after both parties succeed.

`SIGMA_AUTO_CLEAN=0` disables pre-run cleanup. `SIGMA_RECORD_RESULTS=0`
disables recording. Use `Ctrl+C` for normal interruption.

## Results

- `output/P0` and `output/P1` preserve the artifact's original output directory
  structure for the recorded baseline run.
- `results/runs.csv` is the detailed, append-only machine-readable ledger.
- `results/comparison.md` is generated from the CSV for quick inspection.
- `results/raw/<run-id>/` preserves P0/P1 source statistics and metadata before
  the next artifact run overwrites its output directory.

The first run is a functional baseline only. Its timing is marked invalid
because GPU 2 had 94% utilization at launch. Communication and key sizes remain
useful protocol counters.

For formal comparisons, repeat each configuration at least five times under
controlled utilization and report mean and standard deviation. SIGMA and the
future FABLE variant must use the same tensor shapes, inputs, GPU policy,
network setting, and correctness criterion.

## Planned FABLE branch

FABLE's official interface consumes SCI garbled-circuit `Integer` queries,
whereas SIGMA holds ring arithmetic shares on GPU. It is therefore not a
drop-in backend replacement. The integration branch will proceed in stages:

1. reproduce the official standalone FABLE benchmark;
2. implement and validate secure arithmetic/Boolean share conversion;
3. replace the LUT component of GELU first, keeping other SIGMA operators;
4. compare nonzero-output correctness, online latency, communication,
   preprocessing time, and key/material size;
5. only then run the full Transformer comparison.
