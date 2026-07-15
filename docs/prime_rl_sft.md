# Prime-RL SFT on proof-pipeline turns

This path trains OLMo3Sink directly on pre-rendered proof-pipeline turns. The
default production mix contains:

- `ycchen/dsflash-proof-distill-v2-test/data/per_turn.parquet`
- `nvidia/Nemotron-Math-Proofs-v2/data/train.jsonl`

It is offline SFT: there is no policy rollout server, teacher server, verifier
environment, or OPD KL loss.

## Data conversion

`src/prime_sft.py` streams both sources and creates a fingerprinted cache with
`train.parquet`, `validation.parquet`, and `manifest.json`. Conversion holds
only small Arrow batches in RAM; it does not materialize the 17.1 GB Nemotron
JSONL as Python objects.

- Natural source proportions are preserved for `prove`, `verify`, `select`, and
  `refine`; `plain` is excluded.
- `messages_json` supplies the existing system/user prompt unchanged.
- The assistant target keeps `reasoning_content` and `content` separate. The
  DeepSeek-v4 tokenizer template renders them as the thinking block and final
  answer.
- Nemotron's native `messages` are retained without prompt or answer rewriting.
  Its subsets map as `proof -> prove`, `verification -> verify`, and
  `meta-verification -> meta`.
- Validation holds out all rows belonging to 33 deterministic `problem_id`
  values, preventing turns from one problem appearing in both splits.
- Nemotron rows whose normalized problem text exactly matches a held-out
  `per_turn` validation problem are excluded from training by default.
- Rows whose rendered length exceeds the configured context are skipped at
  tokenization time, not truncated. Prime-RL reports these through its overflow
  metrics; the source-audit script provides a deterministic preflight estimate.
- On a multi-node launch, trainer rank 0 performs conversion once and the other
  nodes wait for the shared cache marker.

The pinned Nemotron revision is
`7665d7f1d006fd89aa852a9dab8060c60b63f814`. Its dataset card reports 82,737
rows and approximately 5.0 billion tokens: 24,696 proof, 28,865 verification,
and 29,176 meta-verification rows. The selected `per_turn` source contributes
74,025 rows before its validation split. Runtime tokenization remains the final
authority for context-overflow skipping.

The full node-side audit found valid two-message `user -> assistant` structure
for all 82,737 rows, no nonempty tool payloads, and 5,751 distinct normalized
problem texts. Every assistant message has a separate `reasoning_content` trace;
the final `content` contains the proof, verification, or meta-verification
answer. With the production tokenizer, a deterministic 256-row sample had a
median rendered length of 45,070 tokens and 11 rows (4.30%) above the 131,072
token limit. The overflow was concentrated in very long proof reasoning traces.

Run the streaming source audit on a node after download:

```bash
python scripts/analyze_nemotron_math_proofs_v2.py \
  /tmp/data/Nemotron-Math-Proofs-v2/data/train.jsonl \
  --output /tmp/data/Nemotron-Math-Proofs-v2/analysis.json \
  --tokenizer /tmp/models/opd-32b-deploy/opd-32b-deploy \
  --token-sample-size 256 \
  --seq-len 131072
```

## Production launch

The standalone NII command uses all eight nodes and all eight H200 GPUs per
node:

```bash
export OLMO_RUN_DIR_NAME="prime_sft_per_turn_$(date -u +%Y%m%d_%H%M%S)"
bash operator_commands/prime_rl_sft_8node_per_turn_ctx131072.sh
```

Defaults:

| Setting | Value |
|---|---|
| Context | 131,072 tokens |
| Global batch | 128 packed sequences, 16,777,216 padded tokens/step |
| Microbatch | 1 sequence/GPU |
| Gradient accumulation | 2 microsteps |
| Parallelism | 8 nodes, one 8-GPU FSDP island/node, HSDP replicate=8, CP=1 |
| Precision | FP8 linear training, BF16 optimization/reduction |
| Loss | Liger fused linear cross entropy, assistant tokens only |
| Optimizer | Transformer Engine fused AdamW; BF16 states offloaded to CPU |
| LR | 4e-7, cosine, 10 warmup steps, 5e-8 minimum |
| Steps | 900 |
| Validation | every 50 steps, never before step 1 |
| Checkpoints | every 50 steps, keep the newest 20 weight-only checkpoints |
| W&B | online, shared run across every trainer process |

Activation and optimizer offload remain enabled because the 131k sequence and
optimizer states do not fit together on GPU. Keep microbatch 1 and use
accumulation for larger global batches: each microbatch completes backward and
releases its activation set before the next one begins. Microbatch 2 held two
activation sets at once, doubled peak host RAM, and caused the NII container to
be OOM-killed before its first step.

The command requires the student checkpoint at
`/tmp/models/opd-32b-deploy/opd-32b-deploy`. It downloads each public source
once when missing. Nemotron is pinned to the revision above and stored at
`/tmp/data/Nemotron-Math-Proofs-v2/data/train.jsonl`. Runtime repositories,
normalized data, output, and dependency overlays live below the run-specific
shared root in `/tmp/prime-sft-runs/`.

Disable or narrow the second source without changing code:

```bash
export PRIME_SFT_NEMOTRON_ENABLED=false
# Or retain only selected subsets:
export PRIME_SFT_NEMOTRON_SUBSETS=proof,verification
```

Hopper defaults to Magi FA3 sink attention. The original native FA3 path remains
available for parity checks:

```bash
export PRIME_TRAINER_ATTN=olmo3_sink_fa3_native
```

B200/B300 are forced to Magi FA2 because the current Magi FA4 sink wrapper does
not expose OLMo3's sliding-window argument.

## Smaller checks

A one-node launch preserves the per-GPU microbatch and infers a global batch of
16 with two accumulation microsteps:

```bash
export PRIME_SFT_TRAIN_NODES=0
export PRIME_SFT_MAX_STEPS=2
export PRIME_SFT_SEQ_LEN=8192
export PRIME_SFT_VALIDATION_PROBLEMS=1
bash operator_commands/prime_rl_sft_8node_per_turn_ctx131072.sh
```

Resolve the shell command without downloading data or loading a model:

```bash
GLOBAL_RANK=0 \
PRIME_SFT_TRAIN_NODES=0 \
PRIME_COMMAND_PREVIEW=1 \
PRIME_GPU_NAMES_OVERRIDE='NVIDIA H200' \
PRIME_GPU_COMPUTE_CAPS_OVERRIDE=9.0 \
bash operator_commands/prime_rl_sft_8node_per_turn_ctx131072.sh
```

The wrapper always uses `--fetch-update`. Do not use `--no-fetch-update` for
cluster launches; the SFT local-file, external-launcher, and OLMo3Sink changes
come from the current Prime-RL source checkout.
