# Operator Mode

Operator mode lets a long-running `train.py` process poll a command repo, run
commands on each node, and upload live output back to GitHub. Use it when the
cluster does not expose SSH directly.

## Start Operators On Nodes

Start one operator process inside each container/node. Prefer launching through
`train.py`, not `train_operator.py`, because `train.py` fetches the newest
`aimo-proof-pilot` code before entering operator mode.

Example:

```bash
GITHUB_TOKEN="$GITHUB_TOKEN" \
HF_TOKEN="$HF_TOKEN" \
python /app/train.py \
  --fetch-update \
  --submissions-repo https://github.com/nguyen599/aimo-proof-pilot.git \
  --submissions-ref main \
  --backend open_instruct_wrapper \
  --model_path /tmp/unused \
  --dataset_path /tmp/unused \
  --output_path /tmp/olmo_train_logs/operator_github_command \
  --logdir /tmp/olmo_train_logs/operator_github_command \
  --operator_mode true \
  --operator_backend github \
  --operator_command_repo nguyen599/command \
  --operator_command_file command.sh \
  --operator_key_file key.txt \
  --operator_github_command_download_mode raw \
  --operator_github_api_refresh_interval_seconds 10 \
  --operator_poll_interval_seconds 2 \
  --operator_live_upload_interval_seconds 30 \
  --operator_github_output_branch_template 'operator-output-node-{node}' \
  --operator_output_upload_queue_path /tmp/olmo_operator_output_upload.lock \
  --operator_output_upload_queue_timeout_seconds 180
```

Node labels come from `GLOBAL_RANK`, `NODE_RANK`, `SLURM_NODEID`, or `RANK`.
For a six-node cluster, clients should usually pass `--nodes 0,1,2,3,4,5`.

Operator logs on each node are written under the `--logdir`, commonly:

```bash
/tmp/olmo_train_logs/operator_github_command/train.log
/tmp/olmo_train_logs/operator_github_command/operator_restarts/
```

## Send Commands

Run commands from the local checkout:

```bash
export GITHUB_TOKEN="$GITHUB_TOKEN"

python aimo-proof-pilot/scripts/operator_client.py \
  --backend github \
  --repo nguyen599/command \
  --nodes 0,1,2,3,4,5 \
  send --command 'hostname && nvidia-smi'
```

For larger jobs, write a `.sh` file and send it:

```bash
python aimo-proof-pilot/scripts/operator_client.py \
  --backend github \
  --repo nguyen599/command \
  --nodes 0,1,2,3,4,5 \
  send --file aimo-proof-pilot/operator_commands/prime_rl_opd_3node_full_vocab_dpsk_ctx81920_nodes345.sh
```

The client prints a six-character `command_id`. Outputs for that command are
uploaded as:

```text
output_node<N>_<command_id>.txt
```

By default, outputs are written to per-node branches:

```text
operator-output-node-0
operator-output-node-1
...
```

This avoids most Git push conflicts when many nodes upload at the same time.

## Fetch Logs

Fetch all nodes for one command:

```bash
python aimo-proof-pilot/scripts/operator_client.py \
  --backend github \
  --repo nguyen599/command \
  --nodes 0,1,2,3,4,5 \
  fetch --node all --command-id f77a76 --out-dir /tmp/operator_fetch_f77a76 --no-print
```

Print only useful tail lines:

```bash
for f in /tmp/operator_fetch_f77a76/output_node*; do
  echo "===== $(basename "$f") ====="
  grep -E 'ERROR|Traceback|finished_utc|return_code|vLLM|Prime-RL|ready|timeout|full-vocab|teacher|policy|orchestrator' "$f" | tail -120
done
```

Tail live output:

```bash
python aimo-proof-pilot/scripts/operator_client.py \
  --backend github \
  --repo nguyen599/command \
  --nodes 0,1,2,3,4,5 \
  tail --node all --command-id f77a76 --interval 30 --delta-first
```

Expect 60-90 seconds of delay in practice because each node uploads through Git.

## Re-running A Command

The operator command ID is the first six hex characters of the SHA-256 hash of
the uploaded command text. If you need to run the same command again
immediately, add a harmless nonce comment:

```bash
python aimo-proof-pilot/scripts/operator_client.py \
  --backend github \
  --repo nguyen599/command \
  --nodes 0,1,2,3,4,5 \
  send --command $'hostname && nvidia-smi\n# nonce='"$(date +%s%N)"
```

## Restart Operators

Restart starts a fresh `train.py --operator_mode true` process. The old operator
stays alive but becomes passive if its key no longer matches `key.txt`.

```bash
python aimo-proof-pilot/scripts/operator_client.py \
  --backend github \
  --repo nguyen599/command \
  --nodes 0,1,2,3,4,5 \
  restart
```

Use this after pushing fixes to `aimo-proof-pilot` when you need all nodes to
fetch the latest operator code. Do not kill remote-shell or endpoint daemons
unless you explicitly intend to disable that separate control path.

## Stop Or Kill Jobs

Cancel active commands but keep operator mode alive:

```bash
python aimo-proof-pilot/scripts/operator_client.py \
  --backend github \
  --repo nguyen599/command \
  --nodes 0,1,2,3,4,5 \
  cancel
```

Kill one command by command ID:

```bash
python aimo-proof-pilot/scripts/operator_client.py \
  --backend github \
  --repo nguyen599/command \
  --nodes 0,1,2,3,4,5 \
  kill f77a76 --node all --wait-seconds 90
```

`kill` targets child process groups for that command. It preserves operator
processes, `operator_client.py`, and `train.py --operator_mode true`.

Stop operator mode completely:

```bash
python aimo-proof-pilot/scripts/operator_client.py \
  --backend github \
  --repo nguyen599/command \
  --nodes 0,1,2,3,4,5 \
  stop
```

Only use `stop` when you intentionally want operators to exit.

## Troubleshooting

Check active operators:

```bash
python aimo-proof-pilot/scripts/operator_client.py \
  --backend github \
  --repo nguyen599/command \
  --nodes 0,1,2,3,4,5 \
  send --command 'echo node=${GLOBAL_RANK:-${NODE_RANK:-none}} host=$(hostname); pgrep -af "train.py .*--operator_mode|train_operator.py" || true'
```

Check training or vLLM processes without killing them:

```bash
python aimo-proof-pilot/scripts/operator_client.py \
  --backend github \
  --repo nguyen599/command \
  --nodes 0,1,2,3,4,5 \
  send --command 'echo node=${GLOBAL_RANK:-${NODE_RANK:-none}} host=$(hostname); ps -eo pid,ppid,etime,stat,cmd | grep -E "[p]rime_rl|[v]llm|[t]rain.py|[t]rain_engine"'
```

If output upload is delayed by Git conflicts, wait for the queue to drain before
sending another diagnostic command. Large log files slow down every fetch; clean
old short or oversized output files from the command repo when needed.

## Gradio Relay

Use the Gradio relay for low-latency diagnostics, background job submission, and
log polling. Keep the GitHub operator available as the durable fallback when the
relay Space or daemon is unavailable.

> **Critical:** Never kill, restart, or broadly match the main endpoint process,
> `/app/entrypoint.sh`, or the daemon connected to
> `imo2026-challenge/control-panel`. The main endpoint provides cluster access
> and may be shared by teammates. When cleaning up a job, target only its saved
> PID or an exact job-specific command line.

The project relay is:

```text
imo2026-challenge/control-panel-nguyen
```

The daemon defaults to `imo2026-challenge/control-panel`. Set `RELAY_SPACE`
explicitly so it does not register with the wrong Space. Do not stop the main
relay daemon when starting the project relay; teammates may still depend on it.

### Start The Relay Daemon

The restart wrapper installs missing client dependencies, fetches the current
daemon code, and keeps reconnecting after network failures:

```bash
RELAY_SPACE=imo2026-challenge/control-panel-nguyen \
  bash operator_commands/hotfix_start_remote_shell_daemon.sh
```

The four-node fallback mapping is `node0=vu`, `node1=bogo`, `node2=yi`, and
`node3=nguyen`. Set `RELAY_MEMBER` explicitly when launching a node with a
nonstandard rank or client label.

The underlying client is `remote-shell/daemon/client.py`. Useful environment
variables are `HF_TOKEN`, `CLIENT_ID`, `RELAY_SPACE`, `POLL_INTERVAL`, and
`CMD_TIMEOUT`.

### Connect And Discover Nodes

Use cached Hugging Face authentication rather than putting a token in a command,
because commands and history are visible in the relay UI.

```python
import time

from gradio_client import Client
from huggingface_hub import get_token

SPACE = "imo2026-challenge/control-panel-nguyen"
client = Client(SPACE, token=get_token())

result = client.predict(
    session=f"relay-check-{time.time_ns()}",
    command="echo relay-ok",
    timeout=120,
    api_name="/ui_broadcast",
)
print(result)
```

`/ui_broadcast` sends the command to every connected client and returns their
current labels. Use it only for harmless commands. Labels are Space-specific;
the project relay commonly uses labels such as `node2-hnode070`, while another
Space may expose `user@ip` labels.

### Send A Command To One Node

`/ui_send` acknowledges submission; it does not return the final shell output.
Read the result through `/ui_history`:

```python
import time

from gradio_client import Client
from huggingface_hub import get_token

client = Client(
    "imo2026-challenge/control-panel-nguyen",
    token=get_token(),
)
label = "node2-hnode070"
session = f"gpu-check-{time.time_ns()}"

print(
    client.predict(
        client_label=label,
        session=session,
        command="hostname && nvidia-smi",
        timeout=120,
        api_name="/ui_send",
    )
)

history = client.predict(client_label=label, api_name="/ui_history")
marker = f"[{session}]"
start = history.find(marker)
print(history[start:] if start >= 0 else history)
```

### Run On A Member's Node

The project relay routes each member name to one NII node:

| Member | Human nodes | Daemon ranks |
| --- | --- | --- |
| `vu` | 0 | `node0` |
| `bogo` | 1 | `node1` |
| `yi` | 2 | `node2` |
| `nguyen` | 3 | `node3` |
| `all` | 0-3 | `node0` through `node3` |

```python
from gradio_client import Client
from huggingface_hub import get_token

client = Client(
    "imo2026-challenge/control-panel-nguyen",
    token=get_token(),
)
print(
    client.predict(
        member="vu",
        session="main",
        command="nvidia-smi",
        timeout=120,
        api_name="/ui_team_broadcast",
    )
)
```

The Space UI exposes the same routing through **Run on member node**. Use
`/ui_team_clients` to inspect the currently resolved node. Member routing does
not replace access control: every user with access to the private Space can
still select any node, member, or the all-node broadcast.
Selecting `all` exposes every registered container and sends member-scoped
commands to every online node.

Commands in the same `session` share one persistent bash process, so `cd`,
exports, and virtual-environment activation persist. They also execute in order.
Use a unique session for independent work; different sessions can run
concurrently.

### Submit Long Jobs Safely

Do not leave a training or upload command in the foreground. Start it with
`nohup`, write PID/log/status files, and let the relay command return quickly:

```python
command = r'''LOG=/tmp/my-job.log
PIDFILE=/tmp/my-job.pid
STATUS=/tmp/my-job.status
rm -f "$STATUS"
nohup bash -lc 'set +e; your-command; rc=$?; echo "$rc" > /tmp/my-job.status' \
  > "$LOG" 2>&1 < /dev/null &
echo $! > "$PIDFILE"
echo "started pid=$(cat "$PIDFILE") log=$LOG status=$STATUS"'''
```

Poll it with a separate short command:

```bash
PID=$(cat /tmp/my-job.pid 2>/dev/null || true)
if [ -n "$PID" ] && kill -0 "$PID" 2>/dev/null; then
  echo RUNNING=yes
else
  echo RUNNING=no
fi
echo -n 'status='
cat /tmp/my-job.status 2>/dev/null || echo pending
tail -n 80 /tmp/my-job.log 2>/dev/null || true
```

Never put Hugging Face, GitHub, W&B, or other secrets in a relay command. Avoid
broad `pkill` patterns. In particular, preserve `/app/entrypoint.sh`,
the main endpoint, `train.py --operator_mode true`, and
`remote-shell/daemon/client.py`; target an exact PID file or command-specific
process instead.

### Automatic SFT Checkpoint Uploads

The current SFT run writes checkpoints under:

```text
/tmp/prime-sft-runs/prime_sft_mixed_nemotron_ctx131072_gb64_nooptimoffload_8node_20260715T085639Z/checkpoints/weights
```

Two independent runtime watchers preserve completed checkpoints:

- The personal watcher uploads checkpoints after step 425 to public model
  repositories named `nguyen599/olmo3-opd-sft-<step>`. Step 425 is handled by
  its own one-time upload job.
- The organization watcher backfills every complete checkpoint and uploads all
  future checkpoints to
  `fieldsmodelorg/Olmo-3.1-32B-Think-OPD-SFT-IMO`. Each checkpoint is a complete
  model under its own `step_<step>/` folder in that single repository.

Runtime files are:

```text
/tmp/watch_olmo3_opd_sft_checkpoints.py
/tmp/watch_olmo3_opd_sft_checkpoints.pid
/tmp/watch_olmo3_opd_sft_checkpoints.log
/tmp/prime-sft-runs/prime_sft_mixed_nemotron_ctx131072_gb64_nooptimoffload_8node_20260715T085639Z/checkpoints/weights/hf_auto_upload_state.json
/tmp/watch_fieldsmodelorg_opd_sft_checkpoints.py
/tmp/watch_fieldsmodelorg_opd_sft_checkpoints.pid
/tmp/watch_fieldsmodelorg_opd_sft_checkpoints.log
/tmp/prime-sft-runs/prime_sft_mixed_nemotron_ctx131072_gb64_nooptimoffload_8node_20260715T085639Z/checkpoints/weights/hf_org_auto_upload_state.json
```

Before a full upload starts, the watcher requires:

- A `STABLE` marker.
- The same 21-file manifest as the completed step-425 checkpoint.
- Exactly 14 safetensor shards referenced by `model.safetensors.index.json`.
- Exactly `65,071,239,917` bytes across checkpoint files.
- No checkpoint file newer than the `STABLE` marker.

After upload, each watcher compares every local filename and size with the Hub
manifest before recording the step as complete. The organization watcher uses
hard-linked staging directories, so nested `step_<step>/` uploads do not copy
another 65 GB locally. Failed uploads retry after five minutes, and the
large-folder uploader resumes already uploaded content.

Check watcher state without stopping it:

```bash
PID=$(cat /tmp/watch_olmo3_opd_sft_checkpoints.pid)
kill -0 "$PID" && echo running
tail -n 100 /tmp/watch_olmo3_opd_sft_checkpoints.log
cat /tmp/prime-sft-runs/prime_sft_mixed_nemotron_ctx131072_gb64_nooptimoffload_8node_20260715T085639Z/checkpoints/weights/hf_auto_upload_state.json

ORG_PID=$(cat /tmp/watch_fieldsmodelorg_opd_sft_checkpoints.pid)
kill -0 "$ORG_PID" && echo org-watcher-running
tail -n 100 /tmp/watch_fieldsmodelorg_opd_sft_checkpoints.log
cat /tmp/prime-sft-runs/prime_sft_mixed_nemotron_ctx131072_gb64_nooptimoffload_8node_20260715T085639Z/checkpoints/weights/hf_org_auto_upload_state.json
```

Request a clean stop with:

```bash
touch /tmp/watch_olmo3_opd_sft_checkpoints.stop
touch /tmp/watch_fieldsmodelorg_opd_sft_checkpoints.stop
```

Watcher processes must be relaunched after a node restart, even if the shared
`/tmp` scripts and state files survive.
