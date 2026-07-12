#!/usr/bin/env bash
set -euo pipefail

MAGI_ATTENTION_REPO="${MAGI_ATTENTION_REPO:-https://github.com/SandAI-org/MagiAttention.git}"
MAGI_ATTENTION_REF="${MAGI_ATTENTION_REF:-efaabdbcbc53928debf2fcde189c45d6646210c6}"
MAGI_ATTENTION_DIR="${MAGI_ATTENTION_DIR:-/opt/MagiAttention}"
PYTHON_BIN="${PYTHON_BIN:-python}"

python_install() {
    if command -v uv >/dev/null 2>&1; then
        uv pip install --system "$@"
    else
        "${PYTHON_BIN}" -m pip install --break-system-packages "$@"
    fi
}

if [ ! -d "${MAGI_ATTENTION_DIR}/.git" ]; then
    git clone "${MAGI_ATTENTION_REPO}" "${MAGI_ATTENTION_DIR}"
fi
git -C "${MAGI_ATTENTION_DIR}" fetch origin
git -C "${MAGI_ATTENTION_DIR}" checkout --detach "${MAGI_ATTENTION_REF}"

# Current Magi extensions import an optional DSA test helper from package
# __init__. debugpy and expecttest satisfy its undeclared NGC-image assumptions.
python_install --no-cache-dir debugpy expecttest

# Sink correction uses Magi's Python/Triton utilities. Prime-RL does not use
# Magi's distributed CUDA/communication extensions, so reuse the image's
# FA2/FA3/FA4 wheels and avoid a long, architecture-specific Magi CUDA build.
MAGI_ATTENTION_SKIP_CUDA_BUILD=1 \
    python_install --no-build-isolation --no-deps "${MAGI_ATTENTION_DIR}"
python_install --no-build-isolation --no-deps "${MAGI_ATTENTION_DIR}/extensions"

"${PYTHON_BIN}" - <<'PY'
from magi_attn_extensions.fa2_interface_with_sink import fa2_varlen_func_with_sink

assert callable(fa2_varlen_func_with_sink)
print("MagiAttention FA2 sink extension import OK")
PY
