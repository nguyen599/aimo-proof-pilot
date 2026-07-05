set -euo pipefail

NODE_LABEL="${GLOBAL_RANK:-${NODE_RANK:-${RANK:-none}}}"
echo "convert_phase3_step15_nvfp4_upload_modelopt host=$(hostname) node_label=${NODE_LABEL} pid=$$ started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
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

export MODEL_OPT_SITE="/tmp/modelopt_site"
mkdir -p "${MODEL_OPT_SITE}"
export PYTHONPATH="/tmp/Model-Optimizer:${MODEL_OPT_SITE}:${PYTHONPATH:-}"
export PATH="${MODEL_OPT_SITE}/bin:${PATH}"
export HF_HUB_ENABLE_HF_TRANSFER=0
export HF_HOME="/tmp/hf_home_modelopt"
export HUGGINGFACE_HUB_CACHE="/tmp/hf_home_modelopt/hub"
mkdir -p "${HF_HOME}" "${HUGGINGFACE_HUB_CACHE}"

RUN_ROOT="/tmp/olmo3_rl/outputs/imo1959_2024_grpo_step1100hf_long4x8_vllm020_vllm080_cpuadam_5c84b85/cmd_2812a6de5075c2d3"
MODEL_PATH="${RUN_ROOT}/olmo3_32b_rl_2gen_1train_long__42__1781876839_checkpoints/step_15"
EXPORT_PATH="${RUN_ROOT}/step15-hf-nvfp4"
DATA_PATH="/groups/gcg51557/experiments/0371_aimo/containers/team1/train_phase2.parquet"
CALIB_PATH="/tmp/olmo3_rl/data/phase2_modelopt_calib_6144.jsonl"
SUBMISSIONS_DIR="/tmp/aimo-proof-pilot-runtime"
export MODEL_OPT_ROOT="/tmp/Model-Optimizer"
export MODEL_OPT_DIR="${MODEL_OPT_ROOT}/examples/llm_ptq"
HF_TARGET="checkpoints/imo1959_2024_grpo_step1100hf_long4x8_vllm020_vllm080_cpuadam_5c84b85/cmd_2812a6de5075c2d3/step15-hf-nvfp4"

[ -d "${MODEL_PATH}" ] || { echo "ERROR: model path missing: ${MODEL_PATH}"; exit 2; }
[ -f "${DATA_PATH}" ] || { echo "ERROR: train_phase2 parquet missing: ${DATA_PATH}"; exit 3; }
mkdir -p "$(dirname "${CALIB_PATH}")"

if [ -d "${SUBMISSIONS_DIR}/.git" ]; then
  echo "Updating submissions repo with exact token"
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

echo "Installing pinned apache-tvm-ffi<0.1.12 into ${MODEL_OPT_SITE}"
python -m pip install --break-system-packages --target "${MODEL_OPT_SITE}" --no-deps --force-reinstall 'apache-tvm-ffi<0.1.12'
export PYTHONPATH="${MODEL_OPT_ROOT}:${MODEL_OPT_SITE}:${PYTHONPATH:-}"

echo "Patching ModelOpt optional Megatron NAS plugin to avoid mamba/tilelang import abort"
python - <<'PATCHPY'
from pathlib import Path
import os

plugin_init = Path(os.environ["MODEL_OPT_ROOT"]) / "modelopt/torch/nas/plugins/__init__.py"
if not plugin_init.exists():
    plugin_init = Path(os.environ["MODEL_OPT_SITE"]) / "modelopt/torch/nas/plugins/__init__.py"
text = plugin_init.read_text(encoding="utf-8")
old = 'with import_plugin("megatron"):\n    from .megatron import *\n\n'
new = (
    "# Skipped in this runtime: optional Megatron NAS plugin imports "
    "mamba_ssm/tilelang and aborts in this container.\n"
    '# with import_plugin("megatron"):\n'
    "#     from .megatron import *\n\n"
)
if old in text:
    plugin_init.write_text(text.replace(old, new), encoding="utf-8")
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

echo "Running ModelOpt NVFP4 PTQ"
echo "MODEL_PATH=${MODEL_PATH}"
echo "EXPORT_PATH=${EXPORT_PATH}"
rm -rf "${EXPORT_PATH}.tmp"
mkdir -p "${EXPORT_PATH}.tmp"
cd "${MODEL_OPT_DIR}"
export TRITON_PTXAS_PATH="/usr/local/cuda/bin/ptxas"
python hf_ptq.py \
  --pyt_ckpt_path "${MODEL_PATH}" \
  --qformat nvfp4 \
  --kv_cache_qformat none \
  --export_path "${EXPORT_PATH}.tmp" \
  --trust_remote_code \
  --dataset "${CALIB_PATH}" \
  --calib_size 128 \
  --calib_seq 63512 \
  --batch_size 1
rm -rf "${EXPORT_PATH}"
mv "${EXPORT_PATH}.tmp" "${EXPORT_PATH}"

echo "NVFP4 export complete. Files:"
find "${EXPORT_PATH}" -maxdepth 2 -type f -printf '%s %p\n' | sort -nr | head -80 || true
du -sh "${EXPORT_PATH}" || true

python - <<'PY'
from __future__ import annotations

import inspect
import json
import os
import tempfile
import time
from pathlib import Path

from huggingface_hub import HfApi

export_path = Path("/tmp/olmo3_rl/outputs/imo1959_2024_grpo_step1100hf_long4x8_vllm020_vllm080_cpuadam_5c84b85/cmd_2812a6de5075c2d3/step15-hf-nvfp4")
repo_id = "nguyen599/olmo3-ckpt-phase3"
target = "checkpoints/imo1959_2024_grpo_step1100hf_long4x8_vllm020_vllm080_cpuadam_5c84b85/cmd_2812a6de5075c2d3/step15-hf-nvfp4"
token = os.environ.get("HF_TOKEN")
files = [p for p in export_path.rglob("*") if p.is_file()]
size_bytes = sum(p.stat().st_size for p in files)
manifest = {
    "source": str(export_path),
    "repo_id": repo_id,
    "target": target,
    "file_count": len(files),
    "size_bytes": size_bytes,
    "size_gib": round(size_bytes / (1024**3), 3),
    "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
}
(export_path / "_upload_manifest_step15_nvfp4.json").write_text(
    json.dumps(manifest, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)
api = HfApi(token=token)
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
        commit_message="Create phase3 step15 NVFP4 folder marker",
    )
    print(f"HF_FOLDER_MARKER_OK {repo_id}/{target}/_upload_started.json", flush=True)
finally:
    Path(marker_path).unlink(missing_ok=True)
start = time.monotonic()
for attempt in range(1, 6):
    try:
        large_upload = getattr(api, "upload_large_folder", None)
        if large_upload is not None and "path_in_repo" in inspect.signature(large_upload).parameters:
            print(f"attempt {attempt}: upload_large_folder {export_path} -> {repo_id}/{target}", flush=True)
            result = large_upload(
                repo_id=repo_id,
                repo_type="dataset",
                folder_path=str(export_path),
                path_in_repo=target,
                num_workers=int(os.environ.get("HF_UPLOAD_WORKERS", "8")),
            )
        else:
            print(f"attempt {attempt}: upload_folder {export_path} -> {repo_id}/{target}", flush=True)
            result = api.upload_folder(
                repo_id=repo_id,
                repo_type="dataset",
                folder_path=str(export_path),
                path_in_repo=target,
                commit_message=f"Upload phase3 step15 NVFP4 {time.strftime('%Y%m%d_%H%M%S', time.gmtime())}",
            )
        print(f"UPLOAD_OK elapsed_seconds={time.monotonic() - start:.1f} result={result}", flush=True)
        break
    except Exception as exc:
        print(f"UPLOAD_FAILED attempt={attempt}/5 type={type(exc).__name__}: {exc}", flush=True)
        if attempt >= 5:
            raise
        time.sleep(min(120, 15 * attempt))
PY

echo "convert_phase3_step15_nvfp4_upload_modelopt finished_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
