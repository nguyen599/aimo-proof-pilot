set -euo pipefail

NODE_LABEL="${GLOBAL_RANK:-${NODE_RANK:-${RANK:-none}}}"
echo "convert_phase2_phase3_nvfp4_8gpu_upload host=$(hostname) node_label=${NODE_LABEL} pid=$$ started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
if [ "${NODE_LABEL}" != "0" ]; then
  echo "Skipping NVFP4 conversion/upload on non-primary node ${NODE_LABEL}."
  exit 0
fi

export GIT_TERMINAL_PROMPT=0
export GIT_LFS_SKIP_SMUDGE=1
if [ -n "${GITHUB_TOKEN:-}" ]; then
  export GIT_CONFIG_COUNT=1
  export GIT_CONFIG_KEY_0="url.https://${GITHUB_TOKEN}@github.com/.insteadOf"
  export GIT_CONFIG_VALUE_0="https://github.com/"
else
  export GIT_CONFIG_COUNT=0
fi

export HF_HUB_ENABLE_HF_TRANSFER=0
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export MODEL_OPT_SITE="/tmp/modelopt_site"
export MODEL_OPT_ROOT="/tmp/Model-Optimizer"
export MODEL_OPT_DIR="${MODEL_OPT_ROOT}/examples/llm_ptq"
export PYTHONPATH="${MODEL_OPT_ROOT}:${MODEL_OPT_SITE}:${PYTHONPATH:-}"
export PATH="${MODEL_OPT_SITE}/bin:${PATH}"
export TRITON_PTXAS_PATH="/usr/local/cuda/bin/ptxas"
mkdir -p "${MODEL_OPT_SITE}" /tmp/hf_home_modelopt /tmp/olmo3_rl/data

PHASE2_ROOT="/tmp/olmo3_phase2/outputs/phase2_32b_tp8_pp3_seq65536/phase2_32b_tp8_pp3_seq65536/.hf_converted_checkpoints"
PHASE3_ROOT="/tmp/olmo3_rl/outputs/imo1959_2024_grpo_step1100hf_long4x8_vllm020_vllm080_cpuadam_5c84b85/cmd_2812a6de5075c2d3"
PHASE3_MODEL="${PHASE3_ROOT}/olmo3_32b_rl_2gen_1train_long__42__1781876839_checkpoints/step_15"
PHASE3_EXPORT="${PHASE3_ROOT}/step15-hf-nvfp4"
DATA_PATH="/groups/gcg51557/experiments/0371_aimo/containers/team1/train_phase2.parquet"
CALIB_PATH="/tmp/olmo3_rl/data/phase2_modelopt_calib_6144.jsonl"
SUBMISSIONS_DIR="/tmp/aimo-proof-pilot-runtime"
WORK_ROOT="/tmp/olmo3_nvfp4_parallel_$(date -u +%Y%m%d_%H%M%S)"
mkdir -p "${WORK_ROOT}/logs" "${WORK_ROOT}/status"

echo "WORK_ROOT=${WORK_ROOT}"
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader || true
[ -f "${DATA_PATH}" ] || { echo "ERROR: train_phase2 parquet missing: ${DATA_PATH}"; exit 3; }

if [ -d "${SUBMISSIONS_DIR}/.git" ]; then
  echo "Updating submissions repo"
  git -C "${SUBMISSIONS_DIR}" fetch --depth 1 origin main
  git -C "${SUBMISSIONS_DIR}" checkout --force FETCH_HEAD
else
  echo "Cloning submissions repo"
  rm -rf "${SUBMISSIONS_DIR}"
  git clone --depth 1 --branch main https://github.com/nguyen599/aimo-proof-pilot.git "${SUBMISSIONS_DIR}"
fi
PREPARE_SCRIPT="${SUBMISSIONS_DIR}/scripts/prepare_modelopt_calib.py"
[ -f "${PREPARE_SCRIPT}" ] || { echo "ERROR: missing ${PREPARE_SCRIPT}"; exit 5; }

if [ -d "${MODEL_OPT_ROOT}/.git" ]; then
  echo "Using existing ModelOpt repo ${MODEL_OPT_ROOT}"
  git -C "${MODEL_OPT_ROOT}" fetch --depth 1 origin main || true
else
  echo "Cloning NVIDIA ModelOpt to ${MODEL_OPT_ROOT}"
  rm -rf "${MODEL_OPT_ROOT}"
  git clone --depth 1 https://github.com/NVIDIA/TensorRT-Model-Optimizer.git "${MODEL_OPT_ROOT}"
fi
[ -f "${MODEL_OPT_DIR}/hf_ptq.py" ] || { echo "ERROR: hf_ptq.py not found"; find "${MODEL_OPT_ROOT}" -maxdepth 6 -type f -name hf_ptq.py -print || true; exit 6; }

python - <<'PY'
import importlib.util
import os
import subprocess
import sys

print("python", sys.executable, sys.version, flush=True)
print("target_site", os.environ["MODEL_OPT_SITE"], flush=True)
packages = []
for mod, pkg in [("scipy", "scipy"), ("pulp", "pulp")]:
    if importlib.util.find_spec(mod) is None:
        packages.append(pkg)
if packages:
    cmd = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--break-system-packages",
        "--target",
        os.environ["MODEL_OPT_SITE"],
        "--no-deps",
        "-U",
        *packages,
    ]
    print("RUN", " ".join(cmd), flush=True)
    subprocess.check_call(cmd)
for mod in ["scipy", "pulp", "modelopt", "torch"]:
    spec = importlib.util.find_spec(mod)
    print(f"{mod}_spec={spec.origin if spec else None}", flush=True)
PY

python -m pip install --break-system-packages --target "${MODEL_OPT_SITE}" --no-deps --force-reinstall 'apache-tvm-ffi<0.1.12'

python - <<'PATCHPY'
from pathlib import Path
import os
import re

plugin_init = Path(os.environ["MODEL_OPT_ROOT"]) / "modelopt/torch/nas/plugins/__init__.py"
if not plugin_init.exists():
    plugin_init = Path(os.environ["MODEL_OPT_SITE"]) / "modelopt/torch/nas/plugins/__init__.py"
text = plugin_init.read_text(encoding="utf-8")
pattern = r'with import_plugin\("megatron"\):\n(?:    .+\n)+?\n(?=with import_plugin\("megatron\.bridge"\):)'
replacement = (
    "# Skipped in this runtime: optional Megatron NAS plugin imports "
    "mamba_ssm/tilelang and can abort in this container.\n"
)
new_text = re.sub(pattern, replacement, text)
if new_text != text:
    plugin_init.write_text(new_text, encoding="utf-8")
print(plugin_init.read_text(encoding="utf-8"), flush=True)
PATCHPY

python - <<'IMPORTCHECKPY'
import importlib
for mod in ["modelopt.torch", "modelopt.torch.quantization"]:
    print("import_check", mod, flush=True)
    obj = importlib.import_module(mod)
    print("import_ok", mod, getattr(obj, "__file__", None), flush=True)
IMPORTCHECKPY

if [ -s "${CALIB_PATH}" ] && [ "$(wc -l < "${CALIB_PATH}")" -ge 6144 ]; then
  echo "Reusing existing calibration data: ${CALIB_PATH}"
else
  echo "Preparing calibration data: ${CALIB_PATH}"
  python "${PREPARE_SCRIPT}" --input "${DATA_PATH}" --output "${CALIB_PATH}" --limit 6144 --seed 17
fi
wc -l "${CALIB_PATH}" || true
ls -lh "${CALIB_PATH}" || true

check_model_dir() {
  local model_path="$1"
  [ -d "${model_path}" ] || { echo "ERROR: model dir missing: ${model_path}"; return 1; }
  [ -f "${model_path}/config.json" ] || { echo "ERROR: config missing: ${model_path}/config.json"; return 1; }
  [ -f "${model_path}/model.safetensors.index.json" ] || { echo "ERROR: index missing: ${model_path}/model.safetensors.index.json"; return 1; }
  [ -f "${model_path}/model-00001-of-00002.safetensors" ] || { echo "ERROR: shard1 missing: ${model_path}"; return 1; }
  [ -f "${model_path}/model-00002-of-00002.safetensors" ] || { echo "ERROR: shard2 missing: ${model_path}"; return 1; }
}

run_convert_one() {
  local gpu="$1"
  local label="$2"
  local model_path="$3"
  local export_path="$4"
  local log_path="${WORK_ROOT}/logs/${label}.log"
  local status_path="${WORK_ROOT}/status/${label}.status"
  (
    set -euo pipefail
    export CUDA_VISIBLE_DEVICES="${gpu}"
    export HF_HOME="/tmp/hf_home_modelopt_gpu${gpu}"
    export HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"
    mkdir -p "${HF_HOME}" "${HUGGINGFACE_HUB_CACHE}"
    echo "START label=${label} gpu=${gpu} host=$(hostname) utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "MODEL_PATH=${model_path}"
    echo "EXPORT_PATH=${export_path}"
    check_model_dir "${model_path}"
    rm -rf "${export_path}.tmp"
    mkdir -p "${export_path}.tmp"
    cd "${MODEL_OPT_DIR}"
    python hf_ptq.py \
      --pyt_ckpt_path "${model_path}" \
      --qformat nvfp4 \
      --kv_cache_qformat none \
      --export_path "${export_path}.tmp" \
      --trust_remote_code \
      --dataset "${CALIB_PATH}" \
      --calib_size 128 \
      --calib_seq 63512 \
      --batch_size 1
    rm -rf "${export_path}"
    mv "${export_path}.tmp" "${export_path}"
    echo "DONE label=${label} gpu=${gpu} utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    find "${export_path}" -maxdepth 2 -type f -printf '%s %p\n' | sort -nr | head -80 || true
    du -sh "${export_path}" || true
  ) >"${log_path}" 2>&1
  local rc=$?
  echo "${rc}" > "${status_path}"
  if [ "${rc}" -ne 0 ]; then
    echo "FAILED label=${label} rc=${rc}; tail ${log_path}"
    tail -120 "${log_path}" || true
    return "${rc}"
  fi
  echo "SUCCESS label=${label}; tail ${log_path}"
  tail -40 "${log_path}" || true
}

declare -a LABELS=()
declare -a MODEL_PATHS=()
declare -a EXPORT_PATHS=()
declare -a REPOS=()
declare -a TARGETS=()

LABELS+=("phase3_step15")
MODEL_PATHS+=("${PHASE3_MODEL}")
EXPORT_PATHS+=("${PHASE3_EXPORT}")
REPOS+=("nguyen599/olmo3-ckpt-phase3")
TARGETS+=("checkpoints/imo1959_2024_grpo_step1100hf_long4x8_vllm020_vllm080_cpuadam_5c84b85/cmd_2812a6de5075c2d3/step15-hf-nvfp4")

for step in 400 500 600 700 800 900 1000; do
  LABELS+=("phase2_step${step}")
  MODEL_PATHS+=("${PHASE2_ROOT}/step${step}-hf")
  EXPORT_PATHS+=("${PHASE2_ROOT}/step${step}-hf-nvfp4")
  REPOS+=("nguyen599/olmo3-ckpt-phase2")
  TARGETS+=("checkpoints/phase2_32b_tp8_pp3_seq65536/step${step}-hf-nvfp4")
done

for idx in "${!LABELS[@]}"; do
  check_model_dir "${MODEL_PATHS[$idx]}"
done

echo "Launching ${#LABELS[@]} conversions: GPU0 phase3 step15; GPU1-7 phase2 steps 400..1000."
pids=()
for idx in "${!LABELS[@]}"; do
  gpu="${idx}"
  run_convert_one "${gpu}" "${LABELS[$idx]}" "${MODEL_PATHS[$idx]}" "${EXPORT_PATHS[$idx]}" &
  pids+=("$!")
  sleep 5
done

failure=0
while true; do
  running=0
  for pid in "${pids[@]}"; do
    if kill -0 "${pid}" 2>/dev/null; then
      running=$((running + 1))
    fi
  done
  echo "CONVERT_STATUS utc=$(date -u +%Y-%m-%dT%H:%M:%SZ) running=${running}/${#pids[@]}"
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader || true
  [ "${running}" -gt 0 ] || break
  sleep 120
done

for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    failure=1
  fi
done
[ "${failure}" -eq 0 ] || { echo "ERROR: at least one conversion failed"; exit 20; }

echo "All conversions finished; starting sequential HF uploads."
python - <<'PY'
from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path

from huggingface_hub import HfApi

labels = ["phase3_step15"] + [f"phase2_step{step}" for step in [400, 500, 600, 700, 800, 900, 1000]]
export_paths = [
    "/tmp/olmo3_rl/outputs/imo1959_2024_grpo_step1100hf_long4x8_vllm020_vllm080_cpuadam_5c84b85/cmd_2812a6de5075c2d3/step15-hf-nvfp4",
    *[
        f"/tmp/olmo3_phase2/outputs/phase2_32b_tp8_pp3_seq65536/phase2_32b_tp8_pp3_seq65536/.hf_converted_checkpoints/step{step}-hf-nvfp4"
        for step in [400, 500, 600, 700, 800, 900, 1000]
    ],
]
repos = ["nguyen599/olmo3-ckpt-phase3"] + ["nguyen599/olmo3-ckpt-phase2"] * 7
targets = [
    "checkpoints/imo1959_2024_grpo_step1100hf_long4x8_vllm020_vllm080_cpuadam_5c84b85/cmd_2812a6de5075c2d3/step15-hf-nvfp4",
    *[f"checkpoints/phase2_32b_tp8_pp3_seq65536/step{step}-hf-nvfp4" for step in [400, 500, 600, 700, 800, 900, 1000]],
]
token = os.environ.get("HF_TOKEN")
api = HfApi(token=token)

for label, export_path_s, repo_id, target in zip(labels, export_paths, repos, targets):
    export_path = Path(export_path_s)
    files = [p for p in export_path.rglob("*") if p.is_file()]
    size_bytes = sum(p.stat().st_size for p in files)
    manifest = {
        "label": label,
        "source": str(export_path),
        "repo_id": repo_id,
        "target": target,
        "file_count": len(files),
        "size_bytes": size_bytes,
        "size_gib": round(size_bytes / (1024**3), 3),
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (export_path / "_upload_manifest_nvfp4.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("UPLOAD_MANIFEST", json.dumps(manifest, sort_keys=True), flush=True)
    api.create_repo(repo_id=repo_id, repo_type="dataset", private=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False) as handle:
        handle.write(json.dumps({**manifest, "marker": "upload_started"}, indent=2, sort_keys=True) + "\n")
        marker_path = handle.name
    try:
        api.upload_file(
            repo_id=repo_id,
            repo_type="dataset",
            path_or_fileobj=marker_path,
            path_in_repo=f"{target}/_upload_started.json",
            commit_message=f"Create {label} NVFP4 folder marker",
        )
    finally:
        Path(marker_path).unlink(missing_ok=True)
    start = time.monotonic()
    for attempt in range(1, 6):
        try:
            print(f"UPLOAD_START label={label} attempt={attempt} repo={repo_id} target={target}", flush=True)
            result = api.upload_folder(
                repo_id=repo_id,
                repo_type="dataset",
                folder_path=str(export_path),
                path_in_repo=target,
                commit_message=f"Upload {label} NVFP4 {time.strftime('%Y%m%d_%H%M%S', time.gmtime())}",
            )
            print(f"UPLOAD_OK label={label} elapsed_seconds={time.monotonic() - start:.1f} result={result}", flush=True)
            break
        except Exception as exc:
            print(f"UPLOAD_FAILED label={label} attempt={attempt}/5 type={type(exc).__name__}: {exc}", flush=True)
            if attempt >= 5:
                raise
            time.sleep(min(180, 30 * attempt))
PY

echo "convert_phase2_phase3_nvfp4_8gpu_upload finished_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
