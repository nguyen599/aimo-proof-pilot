# AIMO Proof Pilot

Training and deployment code for the OLMo-3.1-32B-Think system submitted to
[AIMO Proof Pilot](https://www.kaggle.com/competitions/ai-mathematical-olympiad-proof-pilot/).
The original submission model is standard OLMo3, without sink attention. This
repository later became the base for OLMo3Sink, Prime-RL, and online policy
distillation experiments targeting the IMO 2026 setting.

Large model weights, checkpoints, caches, `.sif` files, W&B runs, and private
credentials are intentionally not committed.

## Project History

The repository has two related but distinct tracks:

| Track | Model | Goal |
|---|---|---|
| Original AIMO Proof Pilot | `allenai/Olmo-3.1-32B-Think` | Train and quantize a proof-oriented model for the one-GPU Kaggle submission. |
| IMO 2026 extension | OLMo3Sink student with Prime-RL/OPD | Continue proof training with verifier traces, a frozen teacher, long contexts, and distributed rollout/training. |

The original implementation was migrated from an earlier repository, so this
checkout does not contain the complete Phase 1-3 commit history. The retained
stage reports, published artifacts, and cluster launchers are the historical
sources for the summary below.

The original Proof Pilot work is the primary project described below. Detailed
documentation for the later track is in
[docs/prime_rl_opd.md](docs/prime_rl_opd.md), with native per-turn SFT details in
[docs/prime_rl_sft.md](docs/prime_rl_sft.md).

## Original Proof Pilot Pipeline

### Phase 1: Broad Mathematical SFT

Phase 1 adapted OLMo-3.1-32B-Think to long mathematical reasoning at the
model's original 65,536-token context length. The final dataset combined:

| Source | Filtering and role | Rows |
|---|---|---:|
| [AstralMath-v1](https://huggingface.co/datasets/nguyen599/AstralMath-v1) | Keep traces with recorded success rate `1`; tool-integrated and verifiable reasoning. | 143,668 |
| [Nemotron-SFT-Math-v4](https://huggingface.co/datasets/nvidia/Nemotron-SFT-Math-v4) | Keep non-tool math traces and favor long responses; long-form reasoning adaptation. | 146,530 |
| [FineProofs-SFT](https://huggingface.co/datasets/lm-provers/FineProofs-SFT) | Keep proof-style supervised examples. | 2,387 |

The resulting
[phase 1 dataset](https://huggingface.co/datasets/nguyen599/aimo-proof-pilot-sft-phase-1)
contains 292,585 samples and about 6B raw tokens.

The published phase 1 run used 24 H200 GPUs, BF16, 65,536-token sequences,
microbatch size 2, gradient accumulation 16, AdamW, and a `5e-6` learning rate.
It trained for about 2,000 optimizer steps at roughly 2.1M tokens per step. The
final checkpoint reached about **93% maj@8 on AIME 2026 without tools**.

Artifacts:

- [Phase 1 checkpoint](https://huggingface.co/nguyen599/allenai-Olmo-3.1-32B-Think-aimo-proof-pilot-sft-v1-ckpt-2000)
- [Phase 1 W&B run](https://api.wandb.ai/links/nguyen599-dev/3icm5a8b)
- Launcher: `operator_commands/phase1_32b_tp8_pp3_seq65536.sh`

An earlier branch extended RoPE scaling for a roughly 92k context and trained
for about 1,000 steps. It reached about 70% on AIME 2026, remained slow at
inference, and frequently exhausted the context, so the final pipeline returned
to the original 65k context.

### Phase 2: Proof and Self-Verification SFT

Phase 2 was the main proof-specialization stage. It started from the phase 1
checkpoint and used approximately 28k examples derived from
[Nemotron-Math-Proofs-v2](https://huggingface.co/datasets/nvidia/Nemotron-Math-Proofs-v2).
The prompts followed a
[DeepSeekMath-V2-style](https://arxiv.org/abs/2511.22570v1) format: generate a
parseable proof, assess its correctness and completeness, and expose weak or
incomplete reasoning through self-evaluation.

The published run used 24 H200 GPUs, BF16, 65,536-token sequences, microbatch
size 3, AdamW, and a lower `1e-6` learning rate to preserve phase 1 capability
while learning proof and verifier behavior. It trained for about 1,000
additional steps and approximately 3.1B tokens.

Artifacts and result:

- [Phase 2 dataset](https://huggingface.co/datasets/nguyen599/aimo-proof-pilot-sft-phase-2)
- [Final phase 2 checkpoint](https://huggingface.co/nguyen599/allenai-Olmo-3.1-32B-Think-aimo-proof-pilot-sft-v2-ckpt-1000)
- [Full phase 2 checkpoint collection](https://huggingface.co/datasets/nguyen599/olmo3-ckpt-phase2)
- [Phase 2 W&B run](https://api.wandb.ai/links/nguyen599-dev/8tadmcfc)
- Launcher: `operator_commands/phase2_32b_tp8_pp3_seq65536.sh`
- Advanced IMO ProofBench v2: **32.5%** with one verification round and no refinement

### Phase 3: RL Refinement Experiment

Phase 3 explored CISPO-based reinforcement learning with proof rewards on IMO
and IMO-shortlist problems. The goal was to improve complete, verifiable proofs
instead of rewarding long but unfinished arguments.

This phase remained experimental. Only 15 RL steps completed before the final
submission deadline, and its checkpoint reached about **30%** on the Advanced
part of IMO ProofBench v2. It did not replace the stronger phase 2 checkpoint
and was not used in the final Kaggle notebook.

Relevant experimental launchers include:

- `operator_commands/rlcsd_verl_async_cispo_32b_1train_2rollout_imo4x8_100steps_fp8.sh`
- `operator_commands/rl_32b_step1100_oneprompt8_disablecar.sh`

The later Prime-RL/OPD work is a continuation of this direction, not part of the
original submitted three-phase result.

## Final Kaggle System

The final model was the phase 2 SFT checkpoint quantized to **NVFP4-W4A16** for
one RTX 6000 Pro GPU. The final inference pipeline operated under the Kaggle
one-hour-per-problem limit:

1. Generate 14 independent candidates.
2. Use proof generation plus self-evaluation for 8 candidates.
3. Use a proof-only prompt for 6 candidates so shorter attempts can survive the context limit.
4. Discard candidates with self-evaluation score zero and truncated proof-only generations.
5. Run four independent verifier calls for every surviving candidate.
6. Select the proof with the highest average verifier score.

Meta-verification and refinement were tested during development but omitted
from the final notebook because of the one-GPU time budget.

Final artifacts:

- [Quantized Kaggle model](https://www.kaggle.com/models/nguyennguyen599/allenai-olmo-3.1-32b-think-aimo-proof-pilot)
- [Final Kaggle notebook](https://www.kaggle.com/code/nguyennguyen599/aimo-proof-pilot)
- Original submission/inference checkout: [nguyen599/submissions-instructions](https://github.com/nguyen599/submissions-instructions)

## Running the Original SFT Stages

The launchers expect one container per cluster node. The host supplies
`GLOBAL_RANK`, `WORLD_SIZE`, `MASTER_ADDR`, and `MASTER_PORT`; `src/train.py`
starts the internal distributed worker processes.

```bash
# Phase 1 broad mathematical SFT
bash operator_commands/phase1_32b_tp8_pp3_seq65536.sh

# Phase 2 proof and self-verification SFT
bash operator_commands/phase2_32b_tp8_pp3_seq65536.sh
```

Both launchers use the direct OLMo-core backend with tensor and pipeline
parallelism. Their paths reflect the original NII cluster layout and must be
overridden or edited for another cluster. The report values above describe the
published runs; operational launcher defaults changed during later experiments,
so inspect the command before an exact reproduction.

The conversion helpers retain the final Hugging Face and NVFP4 export workflow:

- `operator_commands/convert_step1100_to_hf_node0.sh`
- `operator_commands/convert_phase2_phase3_nvfp4_8gpu_upload.sh`
- `operator_commands/convert_phase3_step15_nvfp4_upload_modelopt.sh`

## Repository Layout

| Path | Purpose |
|---|---|
| `src/train.py` | Stable training wrapper used inside Docker, Singularity, and cluster jobs. |
| `src/train_engine.py` | Original OLMo-core/Open-Instruct SFT backend and data conversion. |
| `operator_commands/phase1_*.sh` | Broad mathematical SFT launcher. |
| `operator_commands/phase2_*.sh` | Proof and self-verification SFT launcher. |
| `operator_commands/rl*.sh`, `operator_commands/rlcsd_*.sh` | Phase 3 and later RL experiments. |
| `src/train_engine_rl.py` | Later Prime-RL launcher and config writer. |
| `src/proof_opd_env.py` | Later proof, verifier, meta-verifier, refinement, and eval environment. |
| `src/olmo3_sink/` | Later OLMo3Sink model, attention, vLLM adapter, and conversion helpers. |
| `docs/prime_rl_opd.md` | Prime-RL/OPD extension architecture and launch instructions. |
| `docs/prime_rl_sft.md` | Native Prime-RL SFT on pre-rendered per-turn data. |
| `Dockerfile`, `*.def`, `scripts/` | Container builds, dependency installation, and cluster utilities. |

## Quick Checks

Run from the repository root:

```bash
python -m py_compile src/train.py src/train_engine.py src/train_engine_rl.py src/proof_opd_env.py
bash -n operator_commands/phase1_32b_tp8_pp3_seq65536.sh
bash -n operator_commands/phase2_32b_tp8_pp3_seq65536.sh
```

## Containers

Build the current Docker image with:

```bash
CUDA_VERSION=12.8.1 bash scripts/build_docker.sh
# or
CUDA_VERSION=13.0.2 bash scripts/build_docker.sh
```

Build the Singularity/Apptainer image with:

```bash
bash scripts/build_sif_and_upload.sh
```

The current images include dependencies for both the original OLMo-core SFT
path and the later Prime-RL/OPD path.

## Data, Secrets, and Artifacts

`src/train.py` accepts JSON, JSONL, and Parquet chat datasets. Original SFT data
uses a `messages` column with OpenAI-style chat messages. The later proof RL
environment also accepts CSV and task-specific `question` or `problem` fields.

Use environment variables for credentials:

- `HF_TOKEN`
- `WANDB_API_KEY`
- `GITHUB_TOKEN` for runtime repository updates
- `OPENROUTER_API_KEY` only for optional API-judge paths

Do not commit model weights, checkpoints, generated caches, `.sif` files, W&B
directories, private tokens, or presigned URLs.
