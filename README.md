# AIMO Proof Pilot

Public training and container assets for proof-oriented OLMo3 / OLMo3Sink experiments. The main maintained path in this snapshot is Prime-RL OPD training with a DeepSeekMath-V2-style proof environment.

Large model weights, checkpoints, caches, `.sif` files, W&B runs, and private credentials are intentionally not committed.

## Repository Layout

| Path | Purpose |
|---|---|
| `src/train.py` | Training wrapper used inside Docker/Singularity. It can fetch runtime updates and dispatch to SFT, Prime-RL, VERL, or operator mode. |
| `src/train_engine_rl.py` | Prime-RL launcher and config writer for OLMo3Sink / OPD training. |
| `src/proof_opd_env.py` | Current OPD environment: proof generation, verifier, meta-verifier, optional refinement, and verifiable-answer metrics. |
| `src/olmo3_sink/` | OLMo3Sink model, vLLM adapter, FA3 sink attention, and conversion helpers. |
| `operator_commands/` | Reproducible launch scripts for Modal or cluster/container runs. |
| `imo_data_1959_2024.csv` | Proof-style IMO data with `question` and `solution` columns. |
| `astralbench.csv` | Verifiable answer data with `problem` and `answer` columns, mixed into OPD training for boxed-answer accuracy tracking. |
| `Dockerfile` | CUDA 13 / Torch 2.11 image definition for Prime-RL and VERL experiments. |
| `*.def`, `scripts/build_sif_and_upload.sh` | Singularity/Apptainer build files and helpers. |

## Quick Checks

Run from the repository root:

```bash
python -m py_compile src/train.py src/train_engine.py src/train_engine_rl.py src/proof_opd_env.py
bash -n operator_commands/prime_rl_opd_4xh200_muon_imo_ctx16384_2train_1policy_1teacher.sh
```

The command filename is historical; the current default in that script is a 20,480-token trainer context.

## Current Best OPD Pipeline: 4xH200, 20k Context

Use:

```bash
bash operator_commands/prime_rl_opd_4xh200_muon_imo_ctx16384_2train_1policy_1teacher.sh
```

Default topology:

- GPU 0: policy vLLM rollout server.
- GPUs 1-2: trainer, `CP=2`, Ulysses context parallelism.
- GPU 3: frozen OPD teacher vLLM server.
- Optimizer: `muon`.
- Trainer FP8: enabled.
- Policy and teacher vLLM quantization: FP8.
- Trainer context length: `20480`.
- Rollout max completion tokens: `20480`.
- vLLM max model length: `40960`.
- vLLM `max_num_batched_tokens`: `16384`.
- Policy max concurrent sequences: `16`.
- Teacher max concurrent sequences: `8`.

Default data mix:

- `imo_data_1959_2024.csv` supplies proof-only tasks.
- `astralbench.csv` supplies verifiable tasks mixed into training.
- `aime_2026.csv` supplies the separate verifiable eval set.
- `PRIME_OPD_VERIFIABLE_DATASET_PATH` selects the training-mix verifiable CSV, defaulting to `/tmp/aimo-proof-pilot-runtime/astralbench.csv` in the packaged launch script.
- `PRIME_OPD_EVAL_VERIFIABLE_DATASET_PATH` selects the eval CSV, defaulting to `/tmp/aimo-proof-pilot-runtime/aime_2026.csv`.
- `PRIME_OPD_VERIFIABLE_FRACTION=0.20` mixes 20% verifiable rows into the train environment.
- `PRIME_OPD_VERIFIABLE_MIX_SEED=34521` makes the mixed proof/verifiable ordering reproducible.
- `PRIME_OPD_EVAL_INTERVAL=10` runs eval every 10 steps. Eval metrics appear under `eval/proof_math_verifiable/...` in W&B.
- `PRIME_PROOF_MAX_EXAMPLES=20` keeps the default launch cheap. Increase it for real runs, for example `PRIME_PROOF_MAX_EXAMPLES=1481`.

Example container-style launch:

```bash
export PRIME_OPD_MODEL_PATH=/vol/olmo_train_assets/models/opd-32b-deploy/opd-32b-deploy
export PRIME_OPD_TEACHER_MODEL_PATH="$PRIME_OPD_MODEL_PATH"
export PRIME_OPD_DATASET_PATH=/workspace/aimo-proof-pilot/imo_data_1959_2024.csv
export PRIME_OPD_VERIFIABLE_DATASET_PATH=/workspace/aimo-proof-pilot/astralbench.csv
export PRIME_OPD_EVAL_VERIFIABLE_DATASET_PATH=/workspace/aimo-proof-pilot/aime_2026.csv
export PRIME_OPD_EVAL_INTERVAL=10
export PRIME_OPD_VERIFIABLE_FRACTION=0.20
export PRIME_OPD_VERIFIABLE_MIX_SEED=34521
export PRIME_PROOF_MAX_EXAMPLES=1481
export MAX_TRAIN_STEPS=30
export WANDB_MODE=online
export WANDB_PROJECT=olmo3-prime-rl

bash operator_commands/prime_rl_opd_4xh200_muon_imo_ctx16384_2train_1policy_1teacher.sh
```

The script expects `/app/train.py` inside the image and writes outputs/logs under `/vol/olmo_train_assets/`. Mount this repository at `/workspace/aimo-proof-pilot` and mount a writable volume at `/vol/olmo_train_assets`.

At startup, the script prints the proof dataset path, verifiable dataset path, verifiable fraction, mix seed, max example count, context length, and rollout completion-token cap. Check these lines first when validating that a run is using the intended mixer settings.

## OPD Environment

`src/proof_opd_env.py` implements the Prime-RL environment used by the command above. It is intentionally self-contained: it reads CSV/JSON/Parquet rows, builds prompts, tracks per-sample stage state, computes reward, and exposes W&B metrics through the Prime-RL rubric interface.

Dataset workflow:

1. Load the proof dataset from `--prime_proof_dataset_path`.
2. Load the optional verifiable dataset from `--prime_proof_verifiable_dataset_path`.
3. Normalize proof rows into `task_type=proof` examples using `question`, `problem`, or the first user message.
4. Normalize verifiable rows into `task_type=verifiable` examples using `problem`/`question` plus `answer`.
5. Mix rows deterministically with `--prime_proof_verifiable_fraction` and `--prime_proof_mix_seed`. If the verifiable file is small, rows are repeated to preserve the requested fraction.
6. Store task metadata in each row's `info` field, including `task_type`, `task_id`, `source_index`, and `gold_answer` for verifiable rows.

Runtime workflow:

1. Proof generation prompt.
2. Parse the assistant output and require a closed `</think>` unless `--prime_proof_require_closed_think false` is used.
3. Extract only the visible `## Solution` section; if the generation is truncated or malformed, stop the trace and assign zero proof reward.
4. Run a verifier prompt over the extracted proof.
5. If verifier output is valid, run a meta-verifier prompt over the verifier analysis.
6. Compute `reward = format_score * verifier_score * meta_score`, clamped to `[0, 1]`.
7. Optionally run a refinement round when the selected reward is below `--prime_proof_refine_early_stop_reward`.

For verifiable tasks, the proof-generation prompt additionally asks the model to include one final answer in `\boxed{...}` inside the `## Solution` section. The boxed answer is used only for metrics; the OPD proof/verifier/meta reward path stays unchanged.

Verifier and meta-verifier calls are local model calls in this OPD setup, not OpenRouter/API calls. `--prime_proof_judge_backend none` is expected for the current command.

Invalid-output behavior:

- If proof generation reaches the token limit or omits a closed thinking block, verifier/meta stages are skipped.
- If verifier output is invalid or reaches the token limit, meta verification is skipped and the proof score falls back to zero.
- For verifiable rows, boxed-answer accuracy is still reported when a boxed answer can be extracted from a valid solution section; otherwise it reports `0`.

Important W&B metrics:

- `proof_opd_reward`: final proof reward used by the environment.
- `proof_opd_format_score`: format compliance score.
- `proof_opd_proof_score`: verifier score.
- `proof_opd_meta_score`: meta-verifier score.
- `proof_opd_task_is_verifiable`: `1` for AstralBench-style rows, `0` for proof-only rows.
- `proof_opd_verifiable_accuracy`: `1` if the boxed answer matches, `0` if wrong or missing, `-1` for proof-only rows.
- `proof_opd_boxed_present`: whether a boxed answer was found in the solution section.

## Data Formats

Proof data should provide one of:

- `question`
- `problem`
- `messages` as a fallback source for the first user problem text

Optional `solution` is kept for reference. Verifiable data should provide:

- `problem` or `question`
- `answer`

`astralbench.csv` follows this format.

## Docker

Build:

```bash
DOCKER_BUILDKIT=1 docker build -f Dockerfile -t aimo-proof-pilot:cu130 .
```

Run with four H200 GPUs:

```bash
docker run --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
  -v "$PWD":/workspace/aimo-proof-pilot \
  -v /path/to/olmo_train_assets:/vol/olmo_train_assets \
  -e HF_TOKEN \
  -e WANDB_API_KEY \
  -e WANDB_MODE=online \
  aimo-proof-pilot:cu130 \
  bash /workspace/aimo-proof-pilot/operator_commands/prime_rl_opd_4xh200_muon_imo_ctx16384_2train_1policy_1teacher.sh
```

## Singularity / Apptainer

Build and upload helpers are in `scripts/`:

```bash
bash scripts/build_sif_and_upload.sh
```

Manual run shape:

```bash
singularity run --nv container.sif \
  --backend prime_rl \
  --model_path /path/to/model \
  --dataset_path /path/to/imo_data_1959_2024.csv \
  --prime_env_id proof-opd-env
```

For the provided OPD shell script, bind this repo to `/workspace/aimo-proof-pilot` and bind a writable model/output volume to `/vol/olmo_train_assets`.

## Secrets and Artifacts

Use environment variables for credentials:

- `HF_TOKEN`
- `WANDB_API_KEY`
- `OPENROUTER_API_KEY` if using API-judge paths

Do not commit model weights, checkpoints, generated caches, `.sif` files, W&B directories, private tokens, or presigned URLs.
