#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASE_IMAGE="${BASE_IMAGE:-nguyen599/aimo-proof-pilot:cu128-torch211}"
IMAGE="${IMAGE:-nguyen599/aimo-proof-pilot:beaker-b200-cu128}"

docker build \
  --build-arg "BASE_IMAGE=${BASE_IMAGE}" \
  -f "${REPO_DIR}/beaker/Dockerfile" \
  -t "${IMAGE}" \
  "${REPO_DIR}"

if [[ "${PUSH:-0}" == "1" ]]; then
  docker push "${IMAGE}"
else
  echo "Built ${IMAGE}. Set PUSH=1 to push it."
fi
