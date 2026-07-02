# AIMO Proof Pilot Submission

This repository contains the runnable submission code for proof-oriented OLMo3
training and inference. It supports:

- supervised fine-tuning with OLMo-core and Open-Instruct data conversion,
- Prime-RL experiments for OLMo3Sink / OPD training,
- Singularity container builds for cluster execution,

Model weights, large datasets, cache directories, `.sif` images, and private
credentials are intentionally not stored in this repository.

## Repository Layout

| Path | Purpose |
|---|---|
| `src/train.py` | Training wrapper. Fetches runtime updates and dispatches to SFT, Prime-RL, VERL, or operator mode. |
| `src/train_engine.py` | OLMo-core / Open-Instruct SFT backend and sweep utilities. |
| `src/train_engine_rl.py` | Prime-RL backend for OLMo3Sink / OPD experiments. |
| `src/deepseek_math_v2_env.py` | DeepSeekMath-V2-style proof reward environment. |
| `scripts/` | Build, upload, data-preparation, and operator-client helpers. |
| `operator_commands/` | Example cluster/Modal commands for reproducible experiments. |
| `*.def` | Singularity definition files used to build runnable images. |
| `test.csv` | Tiny smoke-test input. |
| `imo_data_1959_2024.csv` | Small public proof dataset with `question` and `solution` columns. |

## Quick Checks

Run from the repository root:

```bash
python -m py_compile src/run.py src/train.py src/train_engine.py src/train_engine_rl.py
python src/run.py --help
python src/train.py --help
```

For shell scripts:

```bash
bash -n scripts/build_sif_and_upload.sh
bash -n operator_commands/prime_rl_opd_4xh200_muon_imo_ctx16384_2train_1policy_1teacher.sh
```

## Inference

`src/run.py` expects an input CSV with at least:

- `id`
- `problem`

Example:

```bash
python src/run.py \
  --model_path /path/to/model \
  --input_csv test.csv \
  --output_csv outputs/predictions.csv \
  --logdir logs/inference \
  --num_ctx 65536 \
  --max_new_tokens 32768 \
  --temperature 0.7 \
  --top_p 0.95
```

The output CSV includes at least `id,prediction`. Extra diagnostic columns may be
added by the pipeline.

## SFT Training

The main SFT backend is `olmo_core_sft`. It accepts local JSON/JSONL/Parquet
datasets. For chat SFT, use a `messages` column containing OpenAI-style chat
messages. Rows may optionally include a `tool_schema` column for tool-aware chat
templates.

Example:

```bash
python src/train.py \
  --backend olmo_core_sft \
  --model_path /path/to/olmo3-32b \
  --dataset_path /path/to/train.parquet \
  --output_path outputs/sft \
  --logdir logs/sft \
  --model_arch olmo3_32b \
  --num_gpus 8 \
  --tensor_parallel_degree 8 \
  --pipeline_parallel_degree 1 \
  --max_seq_length 65536 \
  --per_device_batch_size 2 \
  --gradient_accumulation_steps 4 \
  --optimizer skip_step_adamw
```

On multi-node clusters, run one Singularity container per node and let
`src/train.py` launch internal `torchrun`. The host should provide
`GLOBAL_RANK`, `WORLD_SIZE`, `MASTER_ADDR`, and `MASTER_PORT`.

## Prime-RL / OLMo3Sink

Prime-RL experiments are launched through `src/train.py --backend prime_rl`.
The current OLMo3Sink OPD layout uses:

- GPU 0: policy vLLM rollout server,
- GPU 1-2: trainer with context parallelism `CP=2`,
- GPU 3: frozen OPD teacher vLLM server.

The current 16k-context IMO smoke command is:

```bash
bash operator_commands/prime_rl_opd_4xh200_muon_imo_ctx16384_2train_1policy_1teacher.sh
```

Important defaults:

- dataset: `imo_data_1959_2024.csv`,
- columns: `question` and `solution`,
- context length: `16384`,
- rollout max completion tokens: `12288`,
- trainer FP8: enabled,
- vLLM policy/teacher quantization: FP8,
- optimizer: Muon.

Override runtime settings with environment variables:

```bash
PRIME_OPD_CTX_LEN=16384 \
PRIME_OPD_COMPLETION_TOKENS=8192 \
MAX_TRAIN_STEPS=1 \
bash operator_commands/prime_rl_opd_4xh200_muon_imo_ctx16384_2train_1policy_1teacher.sh
```

## Docker Build

`Dockerfile` is self-contained: it clones this public repository during the
image build, copies `src/` into `/app`, and installs the CUDA training stack.
This means the file can be shared without also sharing a local checkout.

```bash
DOCKER_BUILDKIT=1 docker build -f Dockerfile -t aimo-proof-pilot:cu130 .
```

If you only have the Dockerfile and no build context, build from stdin:

```bash
DOCKER_BUILDKIT=1 docker build -t aimo-proof-pilot:cu130 - < Dockerfile
```

For private wheel downloads, pass an HF token as a BuildKit secret:

```bash
DOCKER_BUILDKIT=1 docker build \
  --secret id=hf_token,src=/path/to/hf_token.txt \
  -f Dockerfile -t aimo-proof-pilot:cu130 .
```

## Singularity Build

Build scripts are in `scripts/`. A typical end-to-end build/upload flow is:

```bash
bash scripts/build_sif_and_upload.sh
```

For manual local builds:

```bash
singularity build sft-phase1_train_YYYYMMDD.sif sft-phase1_train_YYYYMMDD.def
```

Run a container with:

```bash
singularity run --nv container.sif --backend olmo_core_sft --help
```

## Data Formats

Supported training data formats:

- `.json` / `.jsonl` with a `messages` column,
- `.parquet` with `messages`, `problem`, `question`, or task-specific columns,
- CSV for Prime-RL proof environments.

For proof RL, `question` or `problem` is used as the prompt. `solution` is used
when available for evaluation or reward construction.

## Secrets and Artifacts

Use environment variables for credentials:

- `HF_TOKEN`
- `WANDB_API_KEY`
- `OPENROUTER_API_KEY`

Do not commit:

- model weights,
- generated checkpoints,
- `.sif` images,
- cache folders,
- W&B runs,
- private tokens or presigned URLs.

## Development Notes

Keep entry-point arguments stable: external runners call only `src/run.py` or
`src/train.py`. Prefer adding new backend-specific logic to dedicated modules
instead of growing the wrapper. Before pushing changes, run syntax checks and the
smallest available smoke test.
