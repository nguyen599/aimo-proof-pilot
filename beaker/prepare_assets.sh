#!/usr/bin/env bash
set -euo pipefail

SHARED_ROOT="${OPD_SHARED_ROOT:-/weka/aimo-proof-pilot}"
MODEL_ROOT="${OPD_MODEL_ROOT:-${SHARED_ROOT}/models}"
DATA_ROOT="${OPD_DATA_ROOT:-${SHARED_ROOT}/data}"
STUDENT_ROOT="${MODEL_ROOT}/opd-32b-deploy"
TEACHER_ROOT="${MODEL_ROOT}/dpsk-v4-flash"
DATASET_TARGET="${OPD_DATASET_PATH:-${DATA_ROOT}/per_turn.parquet}"

mkdir -p "${MODEL_ROOT}" "${DATA_ROOT}"

if [[ ! -f "${STUDENT_ROOT}/opd-32b-deploy/config.json" ]]; then
  HF_XET_HIGH_PERFORMANCE=1 hf download \
    --repo-type model \
    --local-dir "${STUDENT_ROOT}" \
    ycchen/proof-pilot-deploy-bundle \
    --include 'opd-32b-deploy/*'
fi

if [[ ! -f "${TEACHER_ROOT}/config.json" ]]; then
  HF_XET_HIGH_PERFORMANCE=1 hf download \
    --repo-type model \
    --local-dir "${TEACHER_ROOT}" \
    deepseek-ai/DeepSeek-V4-Flash
fi

if [[ ! -f "${DATASET_TARGET}" ]]; then
  if [[ -n "${OPD_DATASET_SOURCE:-}" && -f "${OPD_DATASET_SOURCE}" ]]; then
    cp --reflink=auto "${OPD_DATASET_SOURCE}" "${DATASET_TARGET}"
  else
    echo "Missing ${DATASET_TARGET}. Set OPD_DATASET_SOURCE to the local per_turn.parquet path and rerun." >&2
    exit 2
  fi
fi

test -f "${STUDENT_ROOT}/opd-32b-deploy/config.json"
test -f "${TEACHER_ROOT}/config.json"
test -s "${DATASET_TARGET}"
touch "${SHARED_ROOT}/.opd-assets-ready"

du -sh "${STUDENT_ROOT}" "${TEACHER_ROOT}" "${DATASET_TARGET}"
echo "[beaker-opd] assets ready under ${SHARED_ROOT}"
