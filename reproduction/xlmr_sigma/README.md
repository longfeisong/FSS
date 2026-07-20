# FABLE Embedding to SIGMA XLM-R

## Purpose

This directory implements the secure hybrid path:

```text
private XLM-R token IDs (P1) + private XLM-R embedding table (P0)
  -> FABLE confidential vector lookup
  -> signed int16 values (scale 12), retained inside the GC
  -> additive shares in Z_(2^50), without revealing the lookup result
  -> secure share-to-mask conversion (x0,x1 -> public x+r; dealer retains r)
  -> SIGMA XLM-R Transformer blocks with masked model parameters
```

The implemented FABLE stage protects both the token IDs and the embedding
table in the semi-honest two-party model. The SIGMA handoff does not reconstruct
the embedding: each evaluator sends only `x_i + r_i`, where `r_i` is its
dealer-provided one-time mask share.

## Author-aligned profile

No paper author evaluated this exact hybrid model. The profile deliberately
combines parameters that are present in the two official artifacts:

| Component | Setting | Source |
|---|---:|---|
| Encoder shape | 12 layers, 12 heads, hidden 768, FFN 3072 | SIGMA `bert-base` |
| Sequence length | 128 | SIGMA artifact experiments |
| SIGMA arithmetic | scale 12, bitwidth 50 | SIGMA `bert-base` |
| Lookup input domain | 20 bits (`2^20`) | FABLE embedding application |
| Embedding element | signed 16-bit | FABLE embedding application |
| Lookup batch | 512 queries | FABLE: 32 samples x 16 words |
| Actual model | vocab 250,002, hidden 768 | XLM-R-base config |

Four XLM-R sequences of length 128 give the same 512-query batch used by the
FABLE authors. Valid token IDs are in `[0, 250002)`; the FABLE table uses
the author-tested 20-bit input domain and pads unused rows. This avoids adding
an unvalidated BatchPIR parameter set for an 18-bit table in the first result.

XLM-R-base was not itself evaluated by SIGMA. It is selected because its
12/12/768/3072 encoder shape matches SIGMA's BERT-base profile while providing
a native, large 250,002-row word-embedding table.

## FABLE output and the SIGMA handoff

SIGMA's FSS protocols use a masked-value representation. If `x` is the fixed
point embedding and the dealer mask is `r`, both online parties operate on the
public masked value `x + r (mod 2^50)`, while the preprocessing keys encode
`r`. The existing SIGMA executable does not consume `x0` on P0 and `x1` on P1
as its input tensor.

The secure conversion produces both required representations:

```text
share0 + share1                         = x  (mod 2^50)
sigma_masked_input - sigma_dealer_mask = x  (mod 2^50)
```

FABLE's GC sign-extends each private int16 result to 50 bits. P0 supplies a
uniform private share to the GC; only the complementary share is revealed to
P1. The subsequent socket protocol one-time-pads each share with a dealer mask
share before exchange. Neither party learns `x` or the full `r`.

## Setup

Create the small CPU-only preprocessing environment:

```bash
conda env create -f reproduction/xlmr_sigma/environment.yml
conda activate xlmr-sigma-bridge
```

Download the pinned public model snapshot. The 1.1 GiB weights are stored under
`.cache/` and are not committed:

```bash
./reproduction/xlmr_sigma/download_model.sh
```

The pinned snapshot is:

```text
FacebookAI/xlm-roberta-base@e73636d4f797dec63c3081bb6ed5c7b0bb3f2089
model.safetensors sha256:
6fd4797bc397c3b8b55d6bb5740366b57e6a3ce91c04c77f22aafc0c128e6feb
```

## Prepare the model and 512-query workload

```bash
conda run -n xlmr-sigma-bridge \
  python reproduction/xlmr_sigma/prepare_bridge.py \
  --model-dir .cache/models/xlm-roberta-base \
  --output-dir reproduction/xlmr_sigma/artifacts/plaintext-bridge-smoke

conda run -n xlmr-sigma-bridge \
  python reproduction/xlmr_sigma/verify_bridge.py \
  reproduction/xlmr_sigma/artifacts/plaintext-bridge-smoke
```

With no `--text`, the script follows FABLE's author workload:
`srand(12345); rand() % vocab_size`, reshaped into four 128-token sequences.
This makes the 512-query lookup batch almost entirely unique and avoids a pad-
or repetition-heavy timing result. The libc version and query seed are recorded
in the manifest.

Pass `--text` up to four times for a model-semantic functional test. Padding is
rejected by default because repeated pad lookups distort a FABLE batch; pass
`--allow-padding` only for a short-input smoke test.

Generated artifacts are ignored by Git. `manifest.json` records shapes,
dtypes, checksums, quantization error and the exact test texts. Each
`sequence_NNN/` directory contains one `[128, 768]` SIGMA input, because the
four sequences may share one FABLE lookup batch but must remain four separate
attention computations.

Export the 12 XLM-R encoder layers in the exact float stream consumed by
`SytorchModule::load` and SIGMA's `GPUBERT` graph:

```bash
conda run -n xlmr-sigma-bridge \
  python reproduction/xlmr_sigma/export_sytorch_encoder.py \
  --model-dir .cache/models/xlm-roberta-base \
  --output-dir reproduction/xlmr_sigma/artifacts/sytorch-encoder
```

The exporter transposes Hugging Face linear matrices to Sytorch's
`[input, output]` layout and concatenates Q, K and V in SIGMA's `qkvconcat`
order. It writes 85,054,464 float32 parameters (340,217,856 bytes) and a
manifest with every tensor in load order. Embedding tables and the masked-LM
head are intentionally excluded. The verified output SHA256 is
`54b719a23e4b04f4dad395ca9218829007bb31376981cea3cd5fd962eae38743`.

## Current verified result

The included scripts were run against the pinned model with four sequences:

| Check | Result |
|---|---:|
| Lookup queries | 512 |
| Unique token IDs / padding queries | `512 / 0` |
| Embedding tensor | `[4, 128, 768]` |
| Fixed-point range | `[-2528, 2612]` |
| Maximum quantization error | `0.0001220703125` |
| int16 overflow | none |
| additive-share reconstruction | pass |
| SIGMA-mask reconstruction | pass |

The deterministic seed exists only to reproduce this interface test. It is not
a cryptographic randomness source.

The plaintext bridge artifacts supply only the deterministic query workload and
an independent correctness oracle. They are not the implemented secure path or
the comparison baseline.

## Build and run the complete FABLE embedding

Export the 250,002 x 768 table as signed int16 at scale 12, then build the
FABLE application at the author-used 20-bit input and 512-bit output profile:

```bash
conda run -n xlmr-sigma-bridge python \
  reproduction/xlmr_sigma/export_fable_table.py \
  --model-dir .cache/models/xlm-roberta-base \
  --output-dir reproduction/xlmr_sigma/artifacts/fable-table

./reproduction/xlmr_sigma/build_fable_xlmr.sh
```

Run all 768 output dimensions for the 512 private token IDs:

```bash
./reproduction/xlmr_sigma/run_fable_xlmr.sh 0 24
```

FABLE's native embedding application returns 32 int16 dimensions (512 bits)
per lookup. The current correctness-first implementation therefore processes
24 sequential output chunks. Each chunk uses the same 512 private queries;
the final share files have shape `[512,768]`. This is complete and secure, but
repeating setup/query work is an optimization target and must be disclosed in
performance tables.

Convert the two FABLE output shares to SIGMA's public masked input:

```bash
./reproduction/xlmr_sigma/run_sigma_input_bridge.sh \
  reproduction/xlmr_sigma/artifacts/fable-xlmr/<full-run-id>
```

## SS-LinearScan secure baseline

The primary secure comparison baseline is implemented as a separate lookup
backend with the same output contract as FABLE:

```text
P1 private token IDs -> blockwise secret-shared one-hot X
P0 private int16 table -> persistent arithmetic shares E0/E1
fresh dealer triples -> secure X @ E without truncation
P0-share.u64 + P1-share.u64 -> existing SIGMA input bridge
```

The implementation uses one-hot scale 0, table/output scale 12, and arithmetic
in `Z_(2^50)`. It pads only the public row dimension to a whole block and never
truncates the lookup product. Every party atomically reserves a triple block
before reading it. A started or crashed run therefore cannot be resumed with
the same material.

First run the M1 correctness profile from the implementation plan. Here
`small-table.i16` is a row-major `4096 x 32` test table and
`small-queries.u32` contains eight IDs:

```bash
python reproduction/xlmr_sigma/prepare_ss_table_shares.py \
  --table small-table.i16 \
  --output-dir reproduction/xlmr_sigma/artifacts/ss-linear-scan/m1-table \
  --logical-n 4096 --output-dim 32 --block-size 1024

python reproduction/xlmr_sigma/generate_ss_beaver_triples.py \
  --backend numpy \
  --output-dir reproduction/xlmr_sigma/artifacts/ss-linear-scan/m1-triples \
  --logical-n 4096 --queries 8 --output-dim 32 --block-size 1024

XLMR_TOKEN_IDS=small-queries.u32 \
XLMR_FABLE_TABLE=small-table.i16 \
SS_LINEAR_SCAN_BACKEND=numpy \
  ./reproduction/xlmr_sigma/run_ss_linear_scan.sh \
  reproduction/xlmr_sigma/artifacts/ss-linear-scan/m1-table \
  reproduction/xlmr_sigma/artifacts/ss-linear-scan/m1-triples
```

Generate full XLM-R static table shares once, then fresh triples for every
lookup run:

```bash
python reproduction/xlmr_sigma/prepare_ss_table_shares.py \
  --table reproduction/xlmr_sigma/artifacts/fable-table/xlmr_word_embedding_scale12.i16 \
  --output-dir reproduction/xlmr_sigma/artifacts/ss-linear-scan/xlmr-table

python reproduction/xlmr_sigma/generate_ss_beaver_triples.py \
  --backend torch-u50 --gpu 0 \
  --output-dir reproduction/xlmr_sigma/artifacts/ss-linear-scan/triples/<fresh-run-id> \
  --logical-n 250002 --queries 512 --output-dim 768 --block-size 4096

SS_LINEAR_SCAN_BACKEND=torch-u50 \
SS_LINEAR_SCAN_P0_GPU=0 SS_LINEAR_SCAN_P1_GPU=4 \
  ./reproduction/xlmr_sigma/run_ss_linear_scan.sh \
  reproduction/xlmr_sigma/artifacts/ss-linear-scan/xlmr-table \
  reproduction/xlmr_sigma/artifacts/ss-linear-scan/triples/<fresh-run-id>
```

For the full `512 x 768` output, the runner prints the exact existing bridge
command. The debug verifier reconstructs shares only after the run to compare
against the clear gather oracle; it is not part of the lookup-to-SIGMA path.

The Ampere GPU backend decomposes every 50-bit ring element into eight non-negative
base-128 limbs and evaluates the required products with exact INT8-to-int32
Tensor Core GEMMs. Shifted partial products are recombined modulo `2^50`; this
is not an approximate floating-point acceleration. Both the small profile and
the full workload are checked against the NumPy uint64 ring implementation. It
has been validated at full scale on A100; RTX 3090 support uses the same Ampere
INT8 path and must pass the included CUDA smoke test on the target host before
formal measurements.

Run the complete baseline, bridge, and four XLM-R Layer 0 sequences with one
command. It creates fresh triple material on every invocation:

```bash
SS_LINEAR_SCAN_DEALER_GPU=0 \
SS_LINEAR_SCAN_P0_GPU=0 SS_LINEAR_SCAN_P1_GPU=4 \
  ./reproduction/xlmr_sigma/run_ss_linear_scan_sigma.sh 1
```

The verified full run `20260720T093004-xlmr-full-gpu` produced 393,216 lookup
values with zero mismatches in 41.594 s and sent 6,241,127,532 bytes total. The
share-to-mask bridge also had zero mismatches. All four SIGMA output hashes are
identical to the corresponding FABLE+SIGMA outputs. Fresh triples took 43.159 s
and 5,591,007,232 bytes; reusable table shares took 13.253 s and 3,120,562,176
bytes. These timings remain correctness-run numbers because sampled shared-GPU
utilization exceeded the experiment threshold.

The local reproduction tree contains both parties' artifacts; on separate
hosts P0 retains only its table/triple shares and P1 receives only its own.

Run the hermetic protocol tests with:

```bash
python -m unittest -v reproduction/xlmr_sigma/tests/test_ss_linear_scan.py
```

### Paired measurements on a shared A100 server

`run_paired_lookup_benchmark.py` performs a guarded randomized comparison when
exclusive GPUs are unavailable. It freezes one GPU pair, requires a stable
low-load window before every command, alternates method order, creates fresh
SS-LinearScan triples, records one-second GPU/PID/CPU/I/O telemetry, retains
contaminated samples, and computes paired speedup with a bootstrap interval.

Inspect the schedule and current server state without waiting or running work:

```bash
python reproduction/xlmr_sigma/run_paired_lookup_benchmark.py \
  --pairs 7 --warmup-pairs 1 --seed 20260720 --dry-run
```

Test only the 60-second admission gate. This does not generate or consume any
preprocessing material:

```bash
python reproduction/xlmr_sigma/run_paired_lookup_benchmark.py \
  --pairs 7 --warmup-pairs 1 --preflight-only \
  --preflight-seconds 60 --max-wait-minutes 120
```

After the preflight reports a suitable pair, launch the formal lookup-only
comparison. Replace the GPU and CPU sets with the selected pair and its NUMA-
local cores (`nvidia-smi topo -m`):

```bash
nohup python reproduction/xlmr_sigma/run_paired_lookup_benchmark.py \
  --gpus 4,5 --dealer-gpu 4 \
  --p0-cpuset 40-59,120-139 --p1-cpuset 60-79,140-159 \
  --pairs 7 --warmup-pairs 1 --seed 20260720 \
  --preflight-seconds 60 --max-wait-minutes 120 \
  > reproduction/xlmr_sigma/results/paired-benchmark.log 2>&1 &
```

The output directory contains `records.jsonl`, `runs.csv`, `summary.json`, each
command log, every admission window, and per-second telemetry. A sample is
excluded from the primary paired statistic only when a predeclared contamination
condition is recorded; raw samples are never deleted. This currently compares
the 24-chunk correctness-first `FABLE-current` implementation. Run the same
controller again after FABLE query/setup reuse is implemented and label that
result `FABLE-optimized`.

For a host with only one available RTX 3090, pass one GPU index. The controller
then colocates P0 and P1 on that card while retaining the same admission gate,
fresh-triple policy, randomized method order, and telemetry:

```bash
python reproduction/xlmr_sigma/run_paired_lookup_benchmark.py \
  --gpus 1 --dealer-gpu 1 \
  --pairs 7 --warmup-pairs 1 --seed 20260720 \
  --preflight-seconds 60 --max-wait-minutes 120
```

This is a **single-card colocated** profile, not the two-card A100 profile. Run
both FABLE and SS-LinearScan through this controller on the RTX 3090 host; do
not compute speedup by mixing an A100 measurement from another server. The
full measured schedule retains about 5.2 GiB of fresh triples per SS run, so
one warmup plus seven measured pairs requires roughly 42 GiB before other
artifacts. The target environment needs `psutil` plus a CUDA-enabled PyTorch
build compatible with its NVIDIA driver. `environment.yml` supplies the
non-CUDA Python dependencies; install PyTorch after checking `nvidia-smi`.

## Run real XLM-R weights in SIGMA

The SIGMA driver supports the first 1 through 12 XLM-R encoder blocks. Model
parameters are quantized in their correct Sytorch order, then separated into a
dealer-only parameter mask and evaluator-visible masked parameters:

```bash
conda run -n xlmr-sigma-bridge python \
  reproduction/xlmr_sigma/export_sytorch_encoder.py \
  --model-dir .cache/models/xlm-roberta-base \
  --output-dir reproduction/xlmr_sigma/artifacts/sytorch-encoder \
  --layers 1

conda run -n xlmr-sigma-bridge python \
  reproduction/xlmr_sigma/prepare_sigma_weights.py \
  --float-weights reproduction/xlmr_sigma/artifacts/sytorch-encoder/xlmr_encoder_1layer_sytorch.float32 \
  --manifest reproduction/xlmr_sigma/artifacts/sytorch-encoder/xlmr_encoder_1layer_sytorch.manifest.json \
  --output-dir reproduction/xlmr_sigma/artifacts/sigma-weights-1layer

./reproduction/xlmr_sigma/build_sigma_xlmr.sh
SIGMA_P0_GPU=7 SIGMA_P1_GPU=2 \
  ./reproduction/xlmr_sigma/run_sigma_xlmr.sh \
  reproduction/xlmr_sigma/artifacts/fable-xlmr/<full-run-id> 1 0
```

The runner refuses any partial lookup result and records GPU load, offline key
size, online latency, communication, identical-party-output check, and output
hash. A run observed above 30% GPU utilization is marked timing-unreliable.

## Implementation status

1. **Implemented and verified:** FABLE private lookup, GC-to-arithmetic sharing,
   arithmetic-share-to-SIGMA masking, masked XLM-R model parameters, and a real
   selectable-depth SIGMA encoder driver.
2. **Complete composition verified:** all 24 FABLE chunks (4×128×768 values)
   through one real XLM-R encoder block per sequence, using independent SIGMA
   preprocessing. Lookup reconstruction has zero mismatches, both online
   parties agree for all four sequences, and sequence 0 is bit-identical to a
   zero-mask SIGMA correctness oracle.
3. **Performance optimization:** reuse FABLE query/setup work across the 24
   response chunks before treating lookup time or communication as optimized.
4. **Model-semantic extension:** securely add XLM-R position embeddings and
   embedding LayerNorm before claiming end-to-end XLM-R inference. The current
   graph deliberately starts at word embeddings and runs encoder blocks.
5. **Secure baseline M1 implemented:** blockwise dealer-assisted SS-LinearScan,
   static private-table shares, private one-hot input sharing, one-time triple
   enforcement, the common 50-bit output contract, and debug-oracle validation.
6. **Secure baseline M2/M3 implemented and verified:** exact A100 INT8 Tensor
   Core ring GEMM, full `250002×768` table, 512 private queries, 62 streamed
   blocks, the common share-to-mask bridge, and XLM-R Layer 0 on all sequences.
   Formal repeated timing remains pending a sufficiently idle pair of GPUs.
