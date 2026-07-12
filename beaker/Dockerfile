# syntax=docker/dockerfile:1.7

ARG BASE_IMAGE=nguyen599/aimo-proof-pilot:cu128-torch211
FROM ${BASE_IMAGE}

USER root

COPY beaker/entrypoint.sh /usr/local/bin/beaker-opd-entrypoint.sh
COPY beaker/prepare_assets.sh /usr/local/bin/beaker-opd-prepare-assets
COPY operator_commands/prime_rl_opd_8node_full_vocab_dpsk_ctx81920_nodes345.sh \
    /opt/aimo-proof-pilot-beaker/prime_rl_opd_8node_full_vocab_dpsk_ctx81920_nodes345.sh

RUN chmod 0755 \
        /usr/local/bin/beaker-opd-entrypoint.sh \
        /usr/local/bin/beaker-opd-prepare-assets \
        /opt/aimo-proof-pilot-beaker/prime_rl_opd_8node_full_vocab_dpsk_ctx81920_nodes345.sh && \
    python - <<'PY'
import torch
import vllm
from magi_attn_extensions.fa2_interface_with_sink import fa2_varlen_func_with_sink

assert callable(fa2_varlen_func_with_sink)
print("Beaker OPD image preflight OK")
print("torch", torch.__version__, "cuda", torch.version.cuda, "vllm", vllm.__version__)
PY

# The parent image runs the remote-shell supervisor as PID 1. Beaker batch
# experiments need the training bootstrap instead, and must terminate normally
# when Beaker propagates a failure or preemption.
ENTRYPOINT []
CMD ["/usr/local/bin/beaker-opd-entrypoint.sh"]
