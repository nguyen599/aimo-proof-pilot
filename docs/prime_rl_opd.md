# Prime-RL and OPD Extension

This document covers the post-Proof-Pilot extension of the repository for IMO
2026 research. It is separate from the original Kaggle submission, which used
standard OLMo-3.1-32B-Think without sink attention. See the repository
[README](../README.md) for the original Phase 1-3 history and final submission.

The extension adds:

- OLMo3Sink training and vLLM rollout support;
- native Prime-RL SFT and online policy distillation;
- a self-contained proof, verifier, meta-verifier, and refinement environment;
- frozen-teacher hidden-state and token-logprob distillation;
- long-context distributed trainer, policy, and teacher layouts.

## Main Components

| Path | Purpose |
|---|---|
| `src/train_engine_rl.py` | Prime-RL launcher, component config generation, and asset materialization. |
| `src/proof_opd_env.py` | Proof environment for single-turn, multi-turn, and hybrid datasets. |
| `src/prime_sft.py` | Streaming normalization and problem-level train/validation split for per-turn SFT. |
| `src/olmo3_sink/` | OLMo3Sink model, attention implementations, vLLM adapter, and conversion helpers. |
| `operator_commands/prime_rl_opd_*.sh` | Single-node and multi-node OPD launchers. |
| `operator_commands/prime_rl_sft_*.sh` | Native Prime-RL SFT launchers. |
| `PRIME_RL_BATCHING_NOTES.md` | Rollout, packing, and optimizer-step batching semantics. |

## Native SFT on Per-Turn Data

The native Prime-RL SFT path trains on pre-rendered `prove`, `verify`, `select`,
and `refine` rows in `per_turn.parquet`. The 8-node launcher supports 131k
context, HSDP, FP8 OLMo3Sink training, Liger fused cross entropy, cosine learning
rate scheduling, validation, and periodic checkpoints.

```bash
export OLMO_RUN_DIR_NAME="prime_sft_per_turn_$(date -u +%Y%m%d_%H%M%S)"
bash operator_commands/prime_rl_sft_8node_per_turn_ctx131072.sh
```

See [prime_rl_sft.md](prime_rl_sft.md) for dataset semantics, topology,
attention selection, one-node overrides, and command-preview validation.

## Four-GPU OPD Launcher

The historical four-H200 launcher is:

```bash
bash operator_commands/prime_rl_opd_4xh200_muon_imo_ctx16384_2train_1policy_1teacher.sh
```

Its current defaults use:

- GPU 0 for policy vLLM rollout;
- GPUs 1-2 for the trainer;
- GPU 3 for the frozen OPD teacher;
- Muon optimization and FP8 trainer modules;
- FP8 policy and teacher inference;
- 20,480-token trainer and rollout limits;
- 40,960-token vLLM model length;
- 16,384 maximum batched vLLM tokens;
- 16 policy and 8 teacher concurrent sequences.

The command filename retains the older `ctx16384` label even though its defaults
have since changed. Inspect the printed configuration before every run.

Example overrides:

```bash
export PRIME_OPD_MODEL_PATH=/vol/olmo_train_assets/models/opd-32b-deploy/opd-32b-deploy
export PRIME_OPD_TEACHER_MODEL_PATH="$PRIME_OPD_MODEL_PATH"
export PRIME_OPD_DATASET_PATH=/workspace/aimo-proof-pilot/data/imo_data_1959_2024.csv
export PRIME_OPD_VERIFIABLE_DATASET_PATH=/workspace/aimo-proof-pilot/data/astral-bench.csv
export PRIME_OPD_EVAL_VERIFIABLE_DATASET_PATH=/workspace/aimo-proof-pilot/data/hmmt_feb_2026.csv
export PRIME_OPD_VERIFIABLE_FRACTION=0.20
export PRIME_PROOF_MAX_EXAMPLES=1481
export MAX_TRAIN_STEPS=30
export WANDB_MODE=online
export WANDB_PROJECT=olmo3-prime-rl

bash operator_commands/prime_rl_opd_4xh200_muon_imo_ctx16384_2train_1policy_1teacher.sh
```

## OPD Dataset Modes

`src/proof_opd_env.py` supports three execution modes.

### Single-turn

Use `--prime_proof_dataset_mode single` for pre-rendered rows with a
`messages_json` field and stage `prove`, `verify`, `select`, or `refine`. Each
row makes exactly one model call and stops. Existing system and user messages
are passed through unchanged.

```bash
export PRIME_PROOF_DATASET_MODE=single
export PRIME_OPD_DATASET_PATH=/path/to/per_turn.parquet
export PRIME_GROUP_SIZE=1
export PRIME_PROOF_CANDIDATE_GATE=false

bash operator_commands/prime_rl_opd_8node_full_vocab_dpsk_ctx81920_nodes345.sh
```

The launcher can materialize default assets from Hugging Face:

- student: `chankhavu/yccchen-olmo3-deploy`;
- teacher: `deepseek-ai/DeepSeek-V4-Flash`;
- data: `ycchen/dsflash-proof-distill-v2-test`, file `data/per_turn.parquet`.

Complete local paths take precedence. In separated deployments, policy nodes
need only the student, teacher nodes need only the teacher, and trainer nodes
need the dataset plus model artifacts required by the selected loss.

### Multi-turn

A multi-turn proof row enters the generated proof pipeline:

1. Generate one proof.
2. Require parseable final output and, by default, a closed `</think>` block.
3. Route only a configured fraction of proofs into later stages; stop the rest as proof-only traces.
4. Run independent verifier prompts over the extracted proof.
5. Run meta-verification only for valid verifier outputs whose proof score is below `1`.
6. Compute `format_score * average(verifier_score * meta_score)`, clamped to `[0, 1]`.
7. Optionally refine low-scoring proofs with the most useful verifier evidence.
8. Optionally run a selector when the generated-selector path is enabled.

Verifier score `1` skips meta-verification and uses a neutral meta multiplier.
Refinement evidence is ordered by lowest verifier score and then highest
effective meta score, prioritizing reviews that identify concrete problems.

### Hybrid

Hybrid mode deterministically combines pre-rendered single-turn rows with
multi-turn proof problems:

```bash
export PRIME_PROOF_DATASET_MODE=hybrid
export PRIME_OPD_PER_TURN_DATASET_PATH=/path/to/per_turn.parquet
export PRIME_OPD_MULTI_TURN_DATASET_PATH=/path/to/imo_data_1959_2024.csv
export PRIME_OPD_MULTI_TURN_FRACTION=0.20
export PRIME_GROUP_SIZE=1
export PRIME_PROOF_CANDIDATE_GATE=false
export PRIME_PROOF_MULTI_TURN_CONTINUE_FRACTION=0.25
export PRIME_PROOF_NUM_VERIFIERS=4
export PRIME_PROOF_REFINE_ROUNDS=1
export PRIME_PROOF_ENABLE_SELECTOR=false

bash operator_commands/prime_rl_opd_8node_full_vocab_dpsk_ctx81920_nodes345.sh
```

The continuation fraction is an expected ratio over independent tasks, not a
guarantee for each block of samples. The older grouped candidate-ranking path
remains available through `--prime_proof_candidate_gate true`.

## Invalid Output Handling

- A proof that reaches its token limit or lacks required closing structure does not enter verification.
- Invalid or length-stopped verifier output skips meta-verification and receives a zero proof score.
- Invalid train proof output ends the trace with zero proof reward.
- Boxed-answer correctness is measured by the separate eval environment, not by train-time OPD reward.

## Metrics and Trace Logging

Important W&B metrics include:

- `proof_opd_reward`;
- `proof_opd_format_score`;
- `proof_opd_proof_score`;
- `proof_opd_meta_score`;
- `proof_opd_selector_valid`;
- per-stage policy generated-token metrics such as `proof_opd_verify_policy_generated_tokens`.

Generic prompt, reasoning, generated, and total token counts are also emitted.
Sample-level stage records and raw output excerpts are stored in
`proof_opd_trace`. Eval runs report boxed-answer accuracy through the eval reward
and eval sample table.

## OLMo3Sink vLLM Benchmark

Use the offline benchmark to measure rollout speed and inspect finish reasons:

```bash
python scripts/bench_vllm_olmo3sink_speed.py \
  --model /path/to/opd-32b-v33-s150 \
  --batch-size 16 \
  --prompt-tokens 1024 \
  --total-output-tokens 131072 \
  --tensor-parallel-size 1 \
  --data-parallel-size 8 \
  --gpu-memory-utilization 0.95 \
  --kv-cache-dtype fp8 \
  --block-size 256 \
  --quantization fp8 \
  --vllm-disabled-kernels FlashInferFP8ScaledMMLinearKernel \
  --out-json /tmp/olmo3sink_vllm_bench.json
```

The default shape is eight independent one-GPU engines with two requests per
engine. The script reports model load time, prompt and output token counts,
finish reasons, aggregate and per-rank throughput, and GPU snapshots.

## Containers

Build the Docker image:

```bash
CUDA_VERSION=12.8.1 bash scripts/build_docker.sh
# or
CUDA_VERSION=13.0.2 bash scripts/build_docker.sh
```

Build and upload a Singularity/Apptainer image:

```bash
bash scripts/build_sif_and_upload.sh
```

For OPD jobs, mount this repository at `/workspace/aimo-proof-pilot` and a
writable model/output volume at `/vol/olmo_train_assets`. Keep `HF_TOKEN`,
`WANDB_API_KEY`, and optional runtime Git credentials in environment variables,
not in launch scripts or images.
