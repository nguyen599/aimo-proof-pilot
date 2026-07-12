# Prime-RL OPD on AI2 Beaker B200

This folder packages the current OLMo3Sink 32B full-vocabulary OPD pipeline for
eight AI2 Titan nodes (64 B200 GPUs total). It is a role-separated deployment:
This is the same role layout used by the current eight-node NII launcher.

| Beaker replica | Role | GPUs |
|---|---|---:|
| 0 | Prime-RL trainer and orchestrator | 8 |
| 1-6 | Student policy rollout, TP=1 and DP=8 per node | 48 total |
| 7 | DeepSeek-V4-Flash teacher hidden-state scorer, TP=8 | 8 |

The nodes exchange control data over host networking and large artifacts over a
shared WEKA mount. This topology does not run one 64-GPU trainer: the trainer,
policy, and teacher each remain node-local, while Prime-RL coordinates them.

## Files

- `Dockerfile`: thin Beaker layer over the full training image. It replaces the
  remote-shell entrypoint with a normal batch entrypoint.
- `entrypoint.sh`: maps Beaker replica metadata to the existing eight-node
  Prime-RL launcher and validates GPUs, assets, and shared storage.
- `experiment.yaml.template`: 8-replica Titan experiment spec.
- `prepare_assets.sh`: idempotently stages the student and teacher models on
  WEKA and copies `per_turn.parquet`.
- `build_and_push.sh`: builds and optionally pushes the thin Beaker image.

## Default training configuration

- Context: 81,920 trainer tokens.
- Policy completion limit: 65,000 tokens; vLLM model length 90,112.
- Dataset mode: `single`, using pre-rendered `prove`, `verify`, `refine`, and
  `select` prompts from `per_turn.parquet`.
- Optimizer batch: 64 packed sequences, approximately 5.24M padded tokens.
- Distillation: teacher hidden states, BF16 capture, `had_int6_blk32` transport,
  reconstructed full-vocabulary logits, reverse KL.
- Policy capacity: 288 in-flight rollouts, matching 48 per policy node.
- Checkpoints: every 100 optimizer steps, keep the last two.
- W&B: online.
- Trainer attention: automatically forced to `olmo3_sink_fa2` on B200/B300.
  Magi FA4 is not used because OLMo3 contains sliding-attention layers and the
  FA4 sink interface does not expose a sliding window.

## 1. Beaker prerequisites

Install/authenticate the Beaker CLI and choose the workspace that owns the
budget, WEKA bucket, and secrets. Titan is `ai2/titan-cirrascale`; it provides
96 B200 GPUs, so this experiment requests eight of its twelve 8-GPU nodes.

Create secrets in the target workspace. Do not place their values in the image
or YAML:

```bash
read -rsp 'HF token: ' SECRET_VALUE; echo
printf '%s' "${SECRET_VALUE}" | beaker secret write --workspace ai2/YOUR_WORKSPACE HF_TOKEN
unset SECRET_VALUE

read -rsp 'W&B API key: ' SECRET_VALUE; echo
printf '%s' "${SECRET_VALUE}" | beaker secret write --workspace ai2/YOUR_WORKSPACE WANDB_API_KEY
unset SECRET_VALUE

read -rsp 'GitHub token: ' SECRET_VALUE; echo
printf '%s' "${SECRET_VALUE}" | beaker secret write --workspace ai2/YOUR_WORKSPACE GITHUB_TOKEN
unset SECRET_VALUE
```

The GitHub token can be omitted from the spec when both repositories are
public. The HF token needs read access to the model repositories. W&B must be
writable to the configured project.

## 2. Build and publish the image

Build the full base image from the repository root when dependencies changed:

```bash
IMAGE_TAG=nguyen599/aimo-proof-pilot:cu128-torch211 \
  bash scripts/build_docker.sh
docker push nguyen599/aimo-proof-pilot:cu128-torch211
```

Then build the small Beaker-specific layer:

```bash
BASE_IMAGE=nguyen599/aimo-proof-pilot:cu128-torch211 \
IMAGE=nguyen599/aimo-proof-pilot:beaker-b200-cu128 \
PUSH=1 \
  bash beaker/build_and_push.sh
```

The Dockerfile performs import checks for Torch, vLLM, and Magi's FA2 sink
interface. It deliberately resets the parent image's remote-shell entrypoint so
Beaker can propagate failures and preemptions normally.

## 3. Stage assets on WEKA

Mount one writable WEKA bucket at `/weka/aimo-proof-pilot`. The final layout is:

```text
/weka/aimo-proof-pilot/
  models/opd-32b-deploy/opd-32b-deploy/config.json
  models/dpsk-v4-flash/config.json
  data/per_turn.parquet
  runs/<OPD_RUN_NAME>/
```

From a bare Beaker session with the WEKA bucket mounted, run:

```bash
OPD_SHARED_ROOT=/weka/aimo-proof-pilot \
OPD_DATASET_SOURCE=/path/to/per_turn.parquet \
  /usr/local/bin/beaker-opd-prepare-assets
```

The script downloads:

- student: `ycchen/proof-pilot-deploy-bundle`, subdirectory
  `opd-32b-deploy/`;
- teacher: `deepseek-ai/DeepSeek-V4-Flash`.

The current `per_turn.parquet` is about 1.5 GB and is not committed to Git. Copy
it to the WEKA path above before launching. Asset preparation is resumable via
the Hugging Face cache and skips complete model directories.

## 4. Render and submit the experiment

Use a unique run name. Every replica receives the same value, which is required
for rendezvous and shared output paths.

```bash
export BEAKER_BUDGET=ai2/YOUR_BUDGET
export BEAKER_IMAGE=nguyen599/aimo-proof-pilot:beaker-b200-cu128
export WEKA_BUCKET=YOUR_WEKA_BUCKET
export OPD_RUN_NAME="opd-b200-$(date -u +%Y%m%dT%H%M%SZ)"

envsubst < beaker/experiment.yaml.template > /tmp/opd-b200.yaml
beaker experiment create --workspace ai2/YOUR_WORKSPACE /tmp/opd-b200.yaml
```

The spec uses the distributed-training requirements from the Beaker docs:
`replicas`, `leaderSelection`, `hostNetworking`, failure/preemption propagation,
a synchronized start timeout, and Titan's InfiniBand interface settings.

## 5. Validate startup

Follow all replica logs:

```bash
beaker experiment logs EXPERIMENT_ID
```

Expected lines include:

```text
[beaker-opd] topology trainer=0 policy=1-6 teacher=7
[prime-opd] trainer attention backend=olmo3_sink_fa2 gpu_names=NVIDIA B200...
[prime-opd-3node] max_inflight_rollouts=288
```

Then verify:

1. All replicas report 8 B200 GPUs and the same run name.
2. Replica 7 exposes the teacher on port 8001.
3. Replicas 1-6 expose policy APIs on port 8000.
4. Replica 0 discovers all six policy URLs and the teacher URL.
5. Trainer logs show `Optimizer step token batch` near 5.24M padded tokens.
6. The first backward passes the OLMo3Sink sink-gradient canary.
7. W&B shows numeric token graphs under keys like
   `train/proof_math/all/metrics/proof_opd_policy_generated_tokens/mean`.
8. NCCL logs use the expected IB interfaces where collective communication is
   active.

Persistent outputs are under:

```text
/weka/aimo-proof-pilot/runs/<OPD_RUN_NAME>/
  checkpoints/
  hidden_states/
  logs/
  output/
  rdzv/
```

No Beaker `result.path` is configured because checkpoints are large and already
live on WEKA.

## Overrides and smoke tests

All entrypoint defaults can be overridden with Beaker `envVars`. Useful smoke
settings are:

```yaml
- { name: MAX_TRAIN_STEPS, value: "3" }
- { name: PRIME_OPD_CTX_LEN, value: "8192" }
- { name: PRIME_OPD_VLLM_MAX_MODEL_LEN, value: "16384" }
- { name: PRIME_OPD_TEACHER_VLLM_MAX_MODEL_LEN, value: "16384" }
- { name: PRIME_OPD_COMPLETION_TOKENS, value: "8192" }
- { name: PRIME_PACKED_SEQUENCES_PER_STEP, value: "4" }
```

Keep `OPD_RUN_NAME` unique for every attempt. Reusing a name is rejected by
default so stale rendezvous files cannot connect unrelated jobs. Set
`OPD_ALLOW_EXISTING_RUN=1` only for deliberate recovery after inspecting the
shared run directory; checkpoint resume has not yet been validated for this
pipeline.
