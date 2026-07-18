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

The runner refuses a partial FABLE result and records GPU load, offline key
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
