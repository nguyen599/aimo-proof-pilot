#!/usr/bin/env python3
"""
NII Relay daemon — runs inside a no-SSH container, polls the HF Space relay, and
executes shell commands in persistent named bash sessions.

Run it (foreground, or under nohup / tmux / systemd):

    export HF_TOKEN="hf_..."                       # token with access to the private Space (this is the auth)
    export CLIENT_ID="$(hostname)"                 # how it shows up in the UI (optional)
    export RELAY_SPACE="imo2026-challenge/control-panel-nguyen"  # optional
    export RELAY_MEMBER="vu"                       # optional team routing metadata
    python client.py

The relay Space repo id defaults to imo2026-challenge/control-panel, but can be
overridden with RELAY_SPACE, REMOTE_SHELL_SPACE, or CONTROL_PANEL_SPACE.

Each command carries a `session` name. Commands with the same session name share
one long-lived bash process, so `cd`, exports, and activated venvs persist.
A new session name spins up a fresh shell. Different sessions run concurrently
(one worker thread each), so a long command in one session never blocks another
session or the poll loop that keeps this client marked online.

Airgapped/restricted-network friendly: outbound HTTPS only (it reaches
huggingface.co AND <user>-<space>.hf.space), never binds a port, and disables
all telemetry by default.
"""

import json
import os

# Disable all phone-home before importing gradio_client / huggingface_hub, so a
# restricted-egress container never tries to reach api.gradio.app etc.
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("DO_NOT_TRACK", "1")

import time
import signal
import socket
import shutil
import getpass
import threading
import subprocess
import uuid
import queue as queuelib
from collections import defaultdict

from gradio_client import Client, handle_file


def _primary_ip() -> str:
    """Best-effort local IP. The UDP 'connect' just picks the outbound route's
    source address; no packets are sent and the host need not be reachable."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except OSError:
        try:
            return socket.gethostbyname(socket.gethostname())
        except OSError:
            return os.uname().nodename
    finally:
        s.close()


def _default_client_id() -> str:
    try:
        user = getpass.getuser()
    except Exception:
        user = os.environ.get("USER") or "user"
    return f"{user}@{_primary_ip()}"


DEFAULT_SPACE = "imo2026-challenge/control-panel"
SPACE = (
    os.environ.get("RELAY_SPACE")
    or os.environ.get("REMOTE_SHELL_SPACE")
    or os.environ.get("CONTROL_PANEL_SPACE")
    or DEFAULT_SPACE
)
HF_TOKEN = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
CLIENT_ID = os.environ.get("CLIENT_ID") or _default_client_id()
RELAY_MEMBER = (
    os.environ.get("RELAY_MEMBER")
    or os.environ.get("TEAM_MEMBER")
    or ""
).strip().lower()
NODE_RANK = (
    os.environ.get("AIMO_NODE_RANK")
    or os.environ.get("GLOBAL_RANK")
    or os.environ.get("NODE_RANK")
    or os.environ.get("SLURM_NODEID")
    or ""
).strip()
POLL_INTERVAL = float(os.environ.get("POLL_INTERVAL", "5"))
DEFAULT_TIMEOUT = int(os.environ.get("CMD_TIMEOUT", "120"))
MAX_OUTPUT_CHARS = 200_000
MAX_FILE_BYTES = 100 * 1024 * 1024  # 100 MB cap on transfers
MAX_SHELLS = int(os.environ.get("MAX_SHELLS", "48"))
SHELL_IDLE_TTL = int(os.environ.get("SHELL_IDLE_TTL", "1800"))  # reap idle shells after 30 min

_log_lock = threading.Lock()


def log(*a):
    with _log_lock:
        print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


# --------------------------------------------------------------------------- #
# Process-tree helpers (used to interrupt a command without killing bash)      #
# --------------------------------------------------------------------------- #
def _descendants(pid: int) -> list[int]:
    """All descendant pids of `pid`, via /proc. Bash itself is NOT included."""
    children: dict[int, list[int]] = defaultdict(list)
    try:
        entries = os.listdir("/proc")
    except OSError:
        return []
    for d in entries:
        if not d.isdigit():
            continue
        try:
            with open(f"/proc/{d}/stat") as f:
                data = f.read()
            # format: "pid (comm) state ppid ..."; comm may contain spaces/parens
            rest = data[data.rfind(")") + 2:].split()
            ppid = int(rest[1])
        except (OSError, IndexError, ValueError):
            continue
        children[ppid].append(int(d))
    out, stack = [], [pid]
    while stack:
        for c in children.get(stack.pop(), []):
            out.append(c)
            stack.append(c)
    return out


# --------------------------------------------------------------------------- #
# Persistent bash session                                                      #
# --------------------------------------------------------------------------- #
class Shell:
    """A long-lived bash process. Commands run serially within the session;
    output is captured up to a unique sentinel line that carries the exit code."""

    def __init__(self, name: str):
        self.name = name
        self.busy = False
        self.last_used = time.time()
        self._start()

    def _start(self):
        # Fresh queue + reader per incarnation so a previous reader's leftover
        # output (or its EOF marker) can never leak into the new shell.
        self._q: "queuelib.Queue[str | None]" = queuelib.Queue()
        self.proc = subprocess.Popen(
            ["bash", "--norc", "--noprofile"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, bufsize=1,
            start_new_session=True,
            env={**os.environ, "PS1": "", "TERM": "dumb"},
        )
        t = threading.Thread(target=self._pump, args=(self.proc, self._q), daemon=True)
        t.start()

    @staticmethod
    def _pump(proc, q):
        for line in proc.stdout:
            q.put(line)
        q.put(None)  # EOF sentinel

    def _drain(self):
        while True:
            try:
                self._q.get_nowait()
            except queuelib.Empty:
                return

    def _interrupt(self):
        """Stop the running command but keep bash alive: signal only bash's
        descendants (the command + its children), never bash itself."""
        for sig in (signal.SIGINT, signal.SIGTERM):
            kids = _descendants(self.proc.pid)
            if not kids:
                return
            for k in kids:
                try:
                    os.kill(k, sig)
                except OSError:
                    pass
            time.sleep(0.4)
        for k in _descendants(self.proc.pid):
            try:
                os.kill(k, signal.SIGKILL)
            except OSError:
                pass

    def close(self):
        try:
            self.proc.terminate()
        except Exception:
            pass

    def run(self, command: str, timeout: int) -> tuple[str, int]:
        if self.proc.poll() is not None:  # shell died (e.g. `exit`) — respawn
            log(f"session {self.name}: shell exited, restarting")
            self._start()

        self.busy = True
        try:
            self._drain()
            marker = "__NII_END_" + uuid.uuid4().hex + "__"
            # Run the command as a group with stdin from /dev/null so commands
            # that read stdin (cat, a REPL, ssh, read) can't swallow the sentinel
            # and hang. The group is NOT a subshell, so cd/exports still persist.
            # Then print the sentinel + exit code on its own line.
            script = (
                "{\n" + command + "\n} </dev/null\n"
                + f"printf '\\n%s %s\\n' '{marker}' \"$?\"\n"
            )
            try:
                self.proc.stdin.write(script)
                self.proc.stdin.flush()
            except (BrokenPipeError, ValueError):
                return ("[relay: shell pipe broken]", 127)

            out_lines: list[str] = []
            exit_code = 0
            deadline = time.monotonic() + max(1, timeout)
            killed = False
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0 and not killed:
                    self._interrupt()
                    out_lines.append(f"\n[relay: timed out after {timeout}s, interrupted]\n")
                    killed = True
                    deadline = time.monotonic() + 8  # grace to collect the sentinel
                    continue
                try:
                    wait = max(0.1, (deadline - time.monotonic()) if not killed else 0.5)
                    line = self._q.get(timeout=wait)
                except queuelib.Empty:
                    if killed and time.monotonic() > deadline:
                        # bash never produced the sentinel — give up, respawn.
                        self.close()
                        self._start()
                        exit_code = 124
                        break
                    continue
                if line is None:  # shell EOF
                    if not killed:
                        out_lines.append("\n[relay: shell closed]\n")
                    exit_code = 137 if not killed else 130
                    break
                stripped = line.rstrip("\n")
                # Exact sentinel match: "<marker> <int>" only.
                if stripped.startswith(marker + " "):
                    tail = stripped[len(marker) + 1:].strip()
                    if tail.isdigit():
                        exit_code = int(tail)
                        break
                out_lines.append(line)

            output = "".join(out_lines)
            if output.endswith("\n"):  # drop the single newline our printf injected
                output = output[:-1]
            if len(output) > MAX_OUTPUT_CHARS:
                output = output[:MAX_OUTPUT_CHARS] + "\n[relay: output truncated]"
            return (output, exit_code)
        finally:
            self.busy = False
            self.last_used = time.time()


SHELLS: dict[str, Shell] = {}
SHELLS_LOCK = threading.Lock()


def get_shell(name: str) -> Shell:
    with SHELLS_LOCK:
        sh = SHELLS.get(name)
        if sh is not None and sh.proc.poll() is not None:
            sh.close()
            sh = None
        if sh is None:
            # cap total shells: evict the least-recently-used idle one
            if len(SHELLS) >= MAX_SHELLS:
                idle = [(s.last_used, n) for n, s in SHELLS.items() if not s.busy]
                if idle:
                    _, victim = min(idle)
                    SHELLS.pop(victim).close()
                    log(f"evicted idle session (cap): {victim}")
            sh = SHELLS[name] = Shell(name)
            log(f"opened bash session: {name}")
        return sh


def _shell_reaper():
    while True:
        time.sleep(120)
        now = time.time()
        with SHELLS_LOCK:
            for name in list(SHELLS):
                sh = SHELLS[name]
                if not sh.busy and (now - sh.last_used) > SHELL_IDLE_TTL:
                    sh.close()
                    SHELLS.pop(name, None)
                    log(f"reaped idle session: {name}")


# --------------------------------------------------------------------------- #
# Relay client (shared, swapped atomically on reconnect)                       #
# --------------------------------------------------------------------------- #
CLIENT: Client | None = None


def _make_client() -> Client:
    # gradio_client renamed the auth kwarg from `hf_token` to `token` (~v2.x).
    try:
        return Client(SPACE, token=HF_TOKEN, verbose=False)
    except TypeError:
        return Client(SPACE, hf_token=HF_TOKEN, verbose=False)


def _meta() -> str:
    payload = {
        "system": os.uname().sysname,
        "release": os.uname().release,
        "pid": os.getpid(),
    }
    if RELAY_MEMBER:
        payload["member"] = RELAY_MEMBER
    if NODE_RANK.isdigit():
        payload["node_rank"] = int(NODE_RANK)
    return json.dumps(payload, separators=(",", ":"), sort_keys=True)


def connect() -> Client:
    global CLIENT
    while True:
        try:
            log(f"connecting to Space {SPACE} …")
            c = _make_client()
            c.predict(CLIENT_ID, _meta(), api_name="/register")
            CLIENT = c
            log(f"registered as '{CLIENT_ID}'")
            return c
        except Exception as e:
            # Don't log raw exception — it can include request payload (the secret).
            log(f"connect failed: {type(e).__name__} — retrying in 10s")
            time.sleep(10)


def _post(*args, **kwargs):
    c = CLIENT
    if c is None:
        raise RuntimeError("no relay connection")
    return c.predict(*args, **kwargs)


def _safe_result(cid: str, code: int, output: str, session: str):
    try:
        _post(cid, CLIENT_ID, code, output, session, api_name="/result")
    except Exception as e:
        log(f"result post failed for {session}: {type(e).__name__}: {e}")


# --------------------------------------------------------------------------- #
# Command handlers (run on per-session worker threads)                         #
# --------------------------------------------------------------------------- #
def _expand(p: str) -> str:
    return os.path.expanduser(os.path.expandvars(p or ""))


def _result_path(ret) -> str:
    """gradio_client returns a File output as a path str, or a FileData dict."""
    if isinstance(ret, dict):
        return ret.get("path") or ret.get("name") or ""
    if isinstance(ret, (list, tuple)) and ret:
        return _result_path(ret[0])
    return ret if isinstance(ret, str) else ""


def do_put(cmd: dict):
    """Server -> client: download the payload from the relay and write it to dest."""
    cid = cmd["id"]
    dest = _expand(cmd.get("dest") or cmd.get("filename") or "downloaded.bin")
    log(f"[file] put → {dest}")
    try:
        ret = _post(cmd["file_id"], api_name="/fetch_file")
        src = _result_path(ret)
        if not src or not os.path.isfile(src):
            raise FileNotFoundError("relay returned no file")
        size = os.path.getsize(src)
        if size > MAX_FILE_BYTES:
            raise ValueError(f"{size} bytes exceeds the {MAX_FILE_BYTES}-byte limit")
        if os.path.isdir(dest) or dest.endswith(os.sep):
            dest = os.path.join(dest, cmd.get("filename") or os.path.basename(src))
        os.makedirs(os.path.dirname(os.path.abspath(dest)) or ".", exist_ok=True)
        shutil.copy(src, dest)
        out, code = (f"wrote {os.path.getsize(dest)} bytes to {dest}", 0)
    except Exception as e:
        out, code = (f"[relay: put failed: {e}]", 1)
    _safe_result(cid, code, out, "file")
    log(f"[file] put → exit {code}")


def do_get(cmd: dict):
    """Client -> server: read a local file and upload it to the relay."""
    cid = cmd["id"]
    remote = _expand(cmd.get("remote_path"))
    log(f"[file] get {remote}")
    try:
        if not os.path.isfile(remote):
            raise FileNotFoundError(f"not a file: {remote}")
        size = os.path.getsize(remote)
        if size > MAX_FILE_BYTES:
            raise ValueError(f"{size} bytes exceeds the {MAX_FILE_BYTES}-byte limit")
    except Exception as e:
        try:
            _post(cid, CLIENT_ID, None, f"{e}", api_name="/upload_result")
        except Exception as e2:
            log(f"error report failed: {type(e2).__name__}: {e2}")
        log(f"[file] get → error: {e}")
        return
    try:
        _post(cid, CLIENT_ID, handle_file(remote), "", api_name="/upload_result")
        log(f"[file] get → sent {size} bytes")
    except Exception as e:
        log(f"[file] get upload failed: {type(e).__name__}: {e}")


def handle(cmd: dict):
    typ = cmd.get("type", "exec")
    if typ == "put":
        return do_put(cmd)
    if typ == "get":
        return do_get(cmd)
    cid = cmd["id"]
    session = cmd.get("session") or "main"
    command = cmd["command"]
    timeout = int(cmd.get("timeout") or DEFAULT_TIMEOUT)
    log(f"[{session}] $ {command}")
    try:
        output, code = get_shell(session).run(command, timeout)
    except Exception as e:
        output, code = (f"[relay: daemon error: {e!r}]", 1)
    _safe_result(cid, code, output, session)
    log(f"[{session}] → exit {code} ({len(output)} bytes)")


# --------------------------------------------------------------------------- #
# Per-session workers: commands for a session run on that session's thread,    #
# so the poll loop never blocks and sessions run concurrently.                 #
# --------------------------------------------------------------------------- #
class Worker:
    def __init__(self, name: str):
        self.name = name
        self.q: "queuelib.Queue[dict]" = queuelib.Queue()
        threading.Thread(target=self._loop, daemon=True).start()

    def submit(self, cmd: dict):
        self.q.put(cmd)

    def _loop(self):
        while True:
            cmd = self.q.get()
            try:
                handle(cmd)
            except Exception as e:
                log(f"worker[{self.name}] error: {type(e).__name__}: {e}")


WORKERS: dict[str, Worker] = {}
WORKERS_LOCK = threading.Lock()


def get_worker(name: str) -> Worker:
    with WORKERS_LOCK:
        w = WORKERS.get(name)
        if w is None:
            w = WORKERS[name] = Worker(name)
        return w


def _shutdown():
    with SHELLS_LOCK:
        for sh in list(SHELLS.values()):
            sh.close()


def main():
    routing = f", member='{RELAY_MEMBER}'" if RELAY_MEMBER else ""
    log(
        f"daemon starting — client_id='{CLIENT_ID}'{routing}, "
        f"poll every {POLL_INTERVAL}s"
    )
    threading.Thread(target=_shell_reaper, daemon=True).start()
    connect()
    while True:
        try:
            res = CLIENT.predict(CLIENT_ID, api_name="/poll")
            for cmd in (res or {}).get("commands", []):
                get_worker(cmd.get("session") or "main").submit(cmd)
        except KeyboardInterrupt:
            log("bye")
            _shutdown()
            return
        except Exception as e:
            log(f"poll error: {type(e).__name__} — reconnecting")
            connect()
        time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        _shutdown()
