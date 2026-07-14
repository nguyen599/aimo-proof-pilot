# Prime-RL SFT on `per_turn.parquet`

This path trains OLMo3Sink directly on the pre-rendered proof-pipeline turns in
`ycchen/dsflash-proof-distill-v2-test/data/per_turn.parquet`. It is offline SFT:
there is no policy rollout server, teacher server, verifier environment, or OPD
KL loss.

## Data conversion

`src/prime_sft.py` streams the source parquet and creates a fingerprinted cache
with `train.parquet`, `validation.parquet`, and `manifest.json`.

- Natural source proportions are preserved for `prove`, `verify`, `select`, and
  `refine`; `plain` is excluded.
- `messages_json` supplies the existing system/user prompt unchanged.
- The assistant target keeps `reasoning_content` and `content` separate. The
  DeepSeek-v4 tokenizer template renders them as the thinking block and final
  answer.
- Validation holds out all rows belonging to 33 deterministic `problem_id`
  values, preventing turns from one problem appearing in both splits.
- Rows whose rendered length exceeds the configured context are skipped at
  tokenization time, not truncated. The manifest records a source-token-count
  estimate of these rows.
- On a multi-node launch, trainer rank 0 performs conversion once and the other
  nodes wait for the shared cache marker.

For the current source snapshot this produces 72,989 training rows and 1,036
validation rows. The held-out split contains 33 problems, and the train split
contains the other 2,146. Source token counts estimate 366 over-context train
rows and three over-context validation rows; runtime tokenization is the final
authority for skipping them.

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
| Global batch | 64 packed sequences, 8,388,608 padded tokens/step |
| Microbatch | 1 sequence/GPU |
| Parallelism | 8 nodes, one 8-GPU FSDP island/node, HSDP replicate=8, CP=1 |
| Precision | FP8 linear training, BF16 optimization/reduction |
| Loss | Liger fused linear cross entropy, assistant tokens only |
| Optimizer | Transformer Engine fused AdamW |
| LR | 2e-7, cosine, 10 warmup steps, 3e-8 minimum |
| Steps | 1,000 |
| Validation | every 50 steps, never before step 1 |
| Checkpoints | every 100 steps, keep the newest two, including optimizer state |
| W&B | online, shared run across every trainer process |

The command requires the student checkpoint at
`/tmp/models/opd-32b-deploy/opd-32b-deploy`. It downloads the public parquet to
`/tmp/data/opd-v2-test/data/per_turn.parquet` once when it is missing. Runtime
repositories, normalized data, output, and dependency overlays live below the
run-specific shared root in `/tmp/prime-sft-runs/`.

Hopper defaults to Magi FA3 sink attention. The original native FA3 path remains
available for parity checks:

```bash
export PRIME_TRAINER_ATTN=olmo3_sink_fa3_native
```

B200/B300 are forced to Magi FA2 because the current Magi FA4 sink wrapper does
not expose OLMo3's sliding-window argument.

## Smaller checks

A one-node launch preserves the per-GPU batch and infers a global batch of 8:

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
