set -euo pipefail

RELAY_SPACE="${RELAY_SPACE:-${REMOTE_SHELL_SPACE:-${CONTROL_PANEL_SPACE:-imo2026-challenge/control-panel-nguyen}}}"
SPACE_SLUG="${RELAY_SPACE//\//_}"
SPACE_SLUG="${SPACE_SLUG//[^A-Za-z0-9_.-]/_}"
NODE_LABEL="${GLOBAL_RANK:-${NODE_RANK:-${SLURM_NODEID:-${RANK:-none}}}}"
case "$NODE_LABEL" in
    0) DEFAULT_RELAY_MEMBER="vu" ;;
    1) DEFAULT_RELAY_MEMBER="bogo" ;;
    2) DEFAULT_RELAY_MEMBER="yi" ;;
    3) DEFAULT_RELAY_MEMBER="nguyen" ;;
    *) DEFAULT_RELAY_MEMBER="" ;;
esac
RELAY_MEMBER="${RELAY_MEMBER:-${TEAM_MEMBER:-$DEFAULT_RELAY_MEMBER}}"
HOST="$(hostname 2>/dev/null || echo unknown-host)"
NODE_RUNTIME_ROOT="${REMOTE_SHELL_RUNTIME_ROOT:-/tmp/imochallenge/node${NODE_LABEL}}"
LOGS="${NODE_RUNTIME_ROOT}/logs"
RUN="${NODE_RUNTIME_ROOT}/run"
PYTHON_SITE="${NODE_RUNTIME_ROOT}/python_site"
mkdir -p "$LOGS" "$RUN" "$PYTHON_SITE"

PY="$(command -v python3 || command -v python || echo /usr/bin/python3)"
export NODE_RUNTIME_ROOT LOGS RUN PYTHON_SITE
export PYTHONPATH="${PYTHON_SITE}:${PYTHONPATH:-}"
export GRADIO_ANALYTICS_ENABLED=False
export HF_HUB_DISABLE_TELEMETRY=1
export DO_NOT_TRACK=1
export POLL_INTERVAL="${POLL_INTERVAL:-5}"
export CLIENT_ID="${CLIENT_ID:-node${NODE_LABEL}-${HOST}}"
export RELAY_MEMBER RELAY_SPACE

echo "remote-shell daemon hotfix node=${NODE_LABEL} host=${HOST}"
echo "relay_space=${RELAY_SPACE}"
echo "relay_member=${RELAY_MEMBER:-unassigned}"
echo "python=${PY}"
echo "python_site=${PYTHON_SITE}"
echo "hf_token_present=$([ -n "${HF_TOKEN:-${HUGGINGFACE_TOKEN:-}}" ] && echo yes || echo no)"

if ! "$PY" - <<'PY'
import gradio_client
print("gradio_client import OK", getattr(gradio_client, "__version__", "unknown"))
PY
then
    echo "installing gradio_client into user base"
    if ! "$PY" -m pip --version >/dev/null 2>&1; then
        "$PY" -m ensurepip --user || true
    fi
    "$PY" -m pip install --target "$PYTHON_SITE" --upgrade --break-system-packages --no-cache-dir "gradio_client>=1.3"
fi

"$PY" - <<'PY'
import gradio_client
print("gradio_client import OK", getattr(gradio_client, "__version__", "unknown"))
PY

CLIENT_SRC="${REMOTE_SHELL_CLIENT_PY:-}"
if [ -z "$CLIENT_SRC" ]; then
    CLIENT_REPO_DIR="${REMOTE_SHELL_CLIENT_REPO_DIR:-${NODE_RUNTIME_ROOT}/aimo-proof-pilot-remote-shell}"
    CLIENT_REPO_URL="${SUBMISSIONS_REPO:-https://github.com/nguyen599/aimo-proof-pilot.git}"
    CLIENT_REPO_REF="${SUBMISSIONS_REF:-main}"
    if [ ! -d "$CLIENT_REPO_DIR/.git" ]; then
        rm -rf "$CLIENT_REPO_DIR"
        git clone "$CLIENT_REPO_URL" "$CLIENT_REPO_DIR"
    fi
    git -C "$CLIENT_REPO_DIR" fetch origin "$CLIENT_REPO_REF"
    git -C "$CLIENT_REPO_DIR" checkout --force FETCH_HEAD
    CLIENT_SRC="$CLIENT_REPO_DIR/remote-shell/daemon/client.py"
fi
if [ ! -f "$CLIENT_SRC" ]; then
    echo "remote-shell client missing: $CLIENT_SRC" >&2
    exit 1
fi
export REMOTE_SHELL_CLIENT_PY="$CLIENT_SRC"
echo "remote_shell_client=${REMOTE_SHELL_CLIENT_PY}"

START_SCRIPT="${RUN}/start-relay-daemon-${SPACE_SLUG}.sh"
LOG_FILE="${LOGS}/relay-daemon-${SPACE_SLUG}.log"
PID_FILE="${RUN}/relay-daemon-${SPACE_SLUG}.pid"

EXISTING_FILE="${RUN}/remote-shell-client-existing-${SPACE_SLUG}.txt"
if "$PY" - "$RELAY_SPACE" "$CLIENT_ID" > "$EXISTING_FILE" <<'PY'
import os
import sys

target = sys.argv[1]
expected_client_id = sys.argv[2]
matches = []
for pid in os.listdir("/proc"):
    if not pid.isdigit():
        continue
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as handle:
            cmd = handle.read().replace(b"\0", b" ").decode("utf-8", "replace")
        if "remote-shell/daemon/client.py" not in cmd or "operator_client" in cmd:
            continue
        with open(f"/proc/{pid}/environ", "rb") as handle:
            env_items = handle.read().split(b"\0")
        env = {}
        for item in env_items:
            if b"=" in item:
                key, value = item.split(b"=", 1)
                env[key.decode("utf-8", "replace")] = value.decode("utf-8", "replace")
        space = env.get("RELAY_SPACE") or env.get("REMOTE_SHELL_SPACE") or env.get("CONTROL_PANEL_SPACE")
        if space == target and env.get("CLIENT_ID") == expected_client_id:
            matches.append((pid, cmd))
    except OSError:
        continue
for pid, cmd in matches:
    print(pid, cmd)
sys.exit(0 if matches else 1)
PY
then
    echo "remote-shell daemon already running for ${RELAY_SPACE}:"
    cat "$EXISTING_FILE"
    exit 0
fi

WRAPPER_EXISTING_FILE="${RUN}/remote-shell-wrapper-existing.txt"
if pgrep -af "$START_SCRIPT" >"$WRAPPER_EXISTING_FILE" 2>/dev/null; then
    echo "remote-shell restart wrapper already running for ${RELAY_SPACE}:"
    cat "$WRAPPER_EXISTING_FILE"
    exit 0
fi

cat > "$START_SCRIPT" <<'EOF'
#!/bin/bash
set -uo pipefail
RUN="${RUN:?RUN must be exported by the relay launcher}"
PYTHON_SITE="${PYTHON_SITE:?PYTHON_SITE must be exported by the relay launcher}"
PY="${PY:-$(command -v python3 || command -v python || echo /usr/bin/python3)}"
export PYTHONPATH="${PYTHON_SITE}:${PYTHONPATH:-}"
export GRADIO_ANALYTICS_ENABLED=False
export HF_HUB_DISABLE_TELEMETRY=1
export DO_NOT_TRACK=1
export POLL_INTERVAL="${POLL_INTERVAL:-5}"
export RELAY_SPACE="${RELAY_SPACE:-${REMOTE_SHELL_SPACE:-${CONTROL_PANEL_SPACE:-imo2026-challenge/control-panel-nguyen}}}"
export RELAY_MEMBER="${RELAY_MEMBER:-${TEAM_MEMBER:-}}"
NODE_LABEL="${GLOBAL_RANK:-${NODE_RANK:-${SLURM_NODEID:-${RANK:-none}}}}"
HOST="$(hostname 2>/dev/null || echo unknown-host)"
export CLIENT_ID="${CLIENT_ID:-node${NODE_LABEL}-${HOST}}"
CLIENT_PY="${REMOTE_SHELL_CLIENT_PY:-/app/remote-shell/daemon/client.py}"
if [ ! -f "$CLIENT_PY" ]; then
    CLIENT_PY="/tmp/aimo-proof-pilot/remote-shell/daemon/client.py"
fi
SPACE_SLUG="${RELAY_SPACE//\//_}"
SPACE_SLUG="${SPACE_SLUG//[^A-Za-z0-9_.-]/_}"
LOCK_FILE="${RUN}/relay-daemon-${SPACE_SLUG}.lock"
RESTART_DELAY="${REMOTE_SHELL_RESTART_DELAY:-5}"
exec 9>"$LOCK_FILE"
if ! flock -n 9; then
    echo "$(date -u +%FT%TZ) remote-shell restart wrapper already holds lock=${LOCK_FILE}; exiting"
    exit 0
fi
attempt=0
while true; do
    attempt=$((attempt + 1))
    echo "$(date -u +%FT%TZ) starting remote-shell client attempt=${attempt} relay_space=${RELAY_SPACE} relay_member=${RELAY_MEMBER:-unassigned} client_id=${CLIENT_ID} client_py=${CLIENT_PY}"
    "$PY" "$CLIENT_PY"
    rc=$?
    echo "$(date -u +%FT%TZ) remote-shell client exited rc=${rc}; restarting in ${RESTART_DELAY}s"
    sleep "$RESTART_DELAY"
done
EOF
chmod +x "$START_SCRIPT"

echo "starting remote-shell daemon without touching existing entrypoint/operator processes"
setsid nohup "$START_SCRIPT" >> "$LOG_FILE" 2>&1 < /dev/null &
DAEMON_PID=$!
echo "$DAEMON_PID" > "$PID_FILE"
sleep 3

if kill -0 "$DAEMON_PID" 2>/dev/null; then
    echo "remote-shell daemon started pid=${DAEMON_PID}"
else
    echo "remote-shell daemon exited during startup; tail follows"
    tail -80 "$LOG_FILE" || true
    exit 1
fi

echo "process check:"
pgrep -af "python.*remote-shell/daemon/client.py" || true
echo "log tail:"
tail -60 "$LOG_FILE" || true
