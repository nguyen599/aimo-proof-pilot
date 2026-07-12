#!/usr/bin/env bash
set -euo pipefail

SHARED_ROOT="${OPD_SHARED_ROOT:-/weka/aimo-proof-pilot}"
MODEL_ROOT="${OPD_MODEL_ROOT:-${SHARED_ROOT}/models}"
DATA_ROOT="${OPD_DATA_ROOT:-${SHARED_ROOT}/data}"
STUDENT_ROOT="${OPD_STUDENT_MODEL_PATH:-${MODEL_ROOT}/opd-32b-deploy/opd-32b-deploy}"
TEACHER_ROOT="${OPD_TEACHER_MODEL_PATH:-${MODEL_ROOT}/dpsk-v4-flash}"
DATASET_TARGET="${OPD_DATASET_PATH:-${DATA_ROOT}/per_turn.parquet}"
STUDENT_REPO="${OPD_STUDENT_HF_REPO:-chankhavu/yccchen-olmo3-deploy}"
TEACHER_REPO="${OPD_TEACHER_HF_REPO:-deepseek-ai/DeepSeek-V4-Flash}"
DATASET_REPO="${OPD_DATASET_HF_REPO:-ycchen/dsflash-proof-distill-v2-test}"
DATASET_FILENAME="${OPD_DATASET_HF_FILENAME:-data/per_turn.parquet}"

mkdir -p "${MODEL_ROOT}" "${DATA_ROOT}"

if [[ ! -f "${STUDENT_ROOT}/config.json" ]]; then
  HF_XET_HIGH_PERFORMANCE=1 hf download \
    --repo-type model \
    --local-dir "${STUDENT_ROOT}" \
    "${STUDENT_REPO}"
fi

if [[ ! -f "${TEACHER_ROOT}/config.json" ]]; then
  HF_XET_HIGH_PERFORMANCE=1 hf download \
    --repo-type model \
    --local-dir "${TEACHER_ROOT}" \
    "${TEACHER_REPO}"
fi

if [[ ! -f "${DATASET_TARGET}" ]]; then
  if [[ -n "${OPD_DATASET_SOURCE:-}" && -f "${OPD_DATASET_SOURCE}" ]]; then
    cp --reflink=auto "${OPD_DATASET_SOURCE}" "${DATASET_TARGET}"
  else
    DATASET_REPO="${DATASET_REPO}" \
    DATASET_FILENAME="${DATASET_FILENAME}" \
    DATASET_TARGET="${DATASET_TARGET}" \
      python - <<'PY'
import os
import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download

target = Path(os.environ["DATASET_TARGET"])
downloaded = Path(
    hf_hub_download(
        repo_id=os.environ["DATASET_REPO"],
        repo_type="dataset",
        filename=os.environ["DATASET_FILENAME"],
        token=os.environ.get("HF_TOKEN") or None,
    )
)
target.parent.mkdir(parents=True, exist_ok=True)
temporary = target.with_name(f".{target.name}.tmp-{os.getpid()}")
try:
    os.link(downloaded, temporary)
except OSError:
    shutil.copyfile(downloaded, temporary)
temporary.replace(target)
PY
  fi
fi

test -f "${STUDENT_ROOT}/config.json"
test -f "${TEACHER_ROOT}/config.json"
test -s "${DATASET_TARGET}"
touch "${SHARED_ROOT}/.opd-assets-ready"

du -sh "${STUDENT_ROOT}" "${TEACHER_ROOT}" "${DATASET_TARGET}"
echo "[beaker-opd] assets ready under ${SHARED_ROOT}"
