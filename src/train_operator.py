from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import queue
import random
import re
import secrets
import shutil
import shlex
import signal
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

from train_logging import configure_logging, log_dependency_versions
from train_utils import path_name_token, sanitize_slug_part, truncate_slug

ENV_ASSIGNMENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")
WRAPPER_HANDOFF_ENV_KEYS = (
    "TRAIN_WRAPPER_REEXECUTED",
    "TRAIN_WRAPPER_PREPARED_ENGINE_PATH",
    "TRAIN_WRAPPER_PREPARED_OPEN_INSTRUCT_DIR",
    "TRAIN_WRAPPER_PREPARED_OLMO_CORE_DIR",
)

# os.environ["GPG_TTY"] = "/dev/stdout"
os.environ["GNUPGHOME"] = os.path.expanduser("~/.gnupg")

os.environ.pop("GPG_TTY", None)
os.environ["GIT_CONFIG_COUNT"] = "1"
os.environ["GIT_CONFIG_KEY_0"] = "commit.gpgsign"
os.environ["GIT_CONFIG_VALUE_0"] = "false"

GITHUB_GIT_LOCKS_GUARD = threading.Lock()
GITHUB_GIT_LOCKS: dict[str, threading.RLock] = {}


class GitPushConflict(RuntimeError):
    pass


def add_operator_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--operator_mode",
        default="false",
        choices=("true", "false"),
        help=(
            "Poll a Hugging Face dataset for operator commands instead of starting training. "
            "New commands run concurrently with active commands; only CANCEL or STOP terminates them. "
            "Plain command text is parsed with shlex and executed with Popen argv; use bash -lc for shell features."
        ),
    )
    parser.add_argument(
        "--operator_backend",
        default="github",
        choices=("hf", "github", "auto"),
        help="Storage backend for operator files. 'github' uses git pull/push only; 'auto' uses git and falls back to HF.",
    )
    parser.add_argument(
        "--operator_command_repo",
        default="nguyen599/command",
        help="Repo containing the operator command file. Use owner/name for HF datasets or GitHub repos.",
    )
    parser.add_argument(
        "--operator_command_file",
        default="command.sh",
        help=(
            "Command file inside --operator_command_repo. May contain STOP, CANCEL, restart_operator, "
            "plain command text, a JSON action object, or a shell script when the filename ends in .sh."
        ),
    )
    parser.add_argument(
        "--operator_key_file",
        default="key.txt",
        help=(
            "Leader-key file inside --operator_command_repo. Each operator process claims a random "
            "6-character key for its node label; only the matching key executes non-control commands."
        ),
    )
    parser.add_argument(
        "--operator_output_repo",
        default="",
        help="Repo for output_node<N>.txt. Defaults to --operator_command_repo.",
    )
    parser.add_argument(
        "--operator_github_branch",
        default="main",
        help="GitHub branch for --operator_backend github.",
    )
    parser.add_argument(
        "--operator_output_prefix",
        default="",
        help="Optional path prefix inside the output repo.",
    )
    parser.add_argument(
        "--operator_poll_interval_seconds",
        type=float,
        default=2.0,
        help="Seconds between command-file polls in --operator_mode.",
    )
    parser.add_argument(
        "--operator_live_upload_interval_seconds",
        type=float,
        default=30.0,
        help=(
            "Seconds between live uploads of output_node<N>.txt while an operator child command is running. "
            "Set <=0 to disable live uploads and upload only after command completion."
        ),
    )
    parser.add_argument(
        "--operator_upload_inactive_outputs",
        default="false",
        choices=("true", "false"),
        help=(
            "Upload an output file when an older operator process sees a command but no longer owns "
            "the node key. Disabled by default because stale operators can create heavy Git push contention."
        ),
    )
    parser.add_argument(
        "--operator_exit_on_key_mismatch",
        default="true",
        choices=("true", "false"),
        help=(
            "Exit an older operator process after a newer operator has claimed the node key. "
            "This keeps restarts from leaving passive processes that keep polling and uploading."
        ),
    )
    parser.add_argument(
        "--operator_output_upload_max_bytes",
        type=int,
        default=8 * 1024 * 1024,
        help=(
            "Maximum bytes published for each operator output file. Larger local logs are "
            "uploaded as a header plus recent tail while the complete log remains in "
            "--operator_work_dir. Set <=0 to upload complete files."
        ),
    )
    parser.add_argument(
        "--operator_work_dir",
        default="/tmp/olmo_operator",
        help="Writable local work directory for operator command downloads and output files.",
    )


def operator_run_name(args: argparse.Namespace) -> str:
    backend = getattr(args, "operator_backend", "github")
    return truncate_slug(f"operator_{backend}_" + path_name_token(args.operator_command_repo, "command"))


def operator_prefers_github(args: argparse.Namespace) -> bool:
    return getattr(args, "operator_backend", "github") in {"github", "auto"}


def github_git_env() -> dict[str, str]:
    env = os.environ.copy()
    token = env.get("GITHUB_TOKEN", "").strip()
    env["GIT_TERMINAL_PROMPT"] = "0"
    env["GIT_LFS_SKIP_SMUDGE"] = "1"
    env.pop("GPG_TTY", None)
    git_config_entries = [
        ("commit.gpgsign", "false"),
        ("tag.gpgsign", "false"),
    ]
    if token:
        git_config_entries.append((f"url.https://{token}@github.com/.insteadOf", "https://github.com/"))
    env["GIT_CONFIG_COUNT"] = str(len(git_config_entries))
    for idx, (key, value) in enumerate(git_config_entries):
        env[f"GIT_CONFIG_KEY_{idx}"] = key
        env[f"GIT_CONFIG_VALUE_{idx}"] = value
    return env


def run_github_git(args: list[str], cwd: Path | None = None) -> str:
    timeout = float(os.environ.get("OPERATOR_GIT_TIMEOUT_SECONDS", "90"))
    try:
        process = subprocess.run(
            ["git", *args],
            cwd=str(cwd) if cwd else None,
            env=github_git_env(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f"git {' '.join(args)} timed out after {timeout:.1f}s"
        ) from exc
    if process.returncode != 0:
        detail = (process.stderr or process.stdout).strip()
        raise RuntimeError(f"git {' '.join(args)} failed with exit code {process.returncode}: {detail}")
    return process.stdout.strip()


def operator_github_git_dir(args: argparse.Namespace, repo: str) -> Path:
    node_label = operator_node_label(args)
    repo_slug = hashlib.sha256(repo.encode("utf-8")).hexdigest()[:12]
    return Path(args.operator_work_dir).expanduser() / f"node{node_label}" / "github_git" / f"{repo_slug}_pid{os.getpid()}"


def github_git_lock(args: argparse.Namespace, repo: str, branch: str) -> threading.RLock:
    key = f"{operator_node_label(args)}:{repo}:{branch}:{os.getpid()}"
    with GITHUB_GIT_LOCKS_GUARD:
        lock = GITHUB_GIT_LOCKS.get(key)
        if lock is None:
            lock = threading.RLock()
            GITHUB_GIT_LOCKS[key] = lock
        return lock


@contextmanager
def locked_github_git_repo(args: argparse.Namespace, repo: str, branch: str):
    with github_git_lock(args, repo, branch):
        yield


def ensure_github_git_repo(args: argparse.Namespace, repo: str, branch: str) -> Path:
    repo_dir = operator_github_git_dir(args, repo)
    repo_url = f"https://github.com/{repo}.git"
    if not (repo_dir / ".git").is_dir():
        repo_dir.parent.mkdir(parents=True, exist_ok=True)
        if repo_dir.exists():
            import shutil

            shutil.rmtree(repo_dir)
        run_github_git(["clone", "--depth", "1", "--branch", branch, repo_url, str(repo_dir)])
        run_github_git(["config", "user.email", "operator@local"], cwd=repo_dir)
        run_github_git(["config", "user.name", "operator"], cwd=repo_dir)
        run_github_git(["config", "commit.gpgsign", "false"], cwd=repo_dir)
        return repo_dir
    try:
        run_github_git(["remote", "set-url", "origin", repo_url], cwd=repo_dir)
        run_github_git(["config", "commit.gpgsign", "false"], cwd=repo_dir)
        run_github_git(["reset", "--hard"], cwd=repo_dir)
        run_github_git(["clean", "-fd"], cwd=repo_dir)
        run_github_git(["fetch", "--depth", "1", "origin", branch], cwd=repo_dir)
        run_github_git(["checkout", "-B", branch, f"origin/{branch}"], cwd=repo_dir)
        run_github_git(["reset", "--hard", f"origin/{branch}"], cwd=repo_dir)
    except Exception:
        import shutil

        shutil.rmtree(repo_dir, ignore_errors=True)
        run_github_git(["clone", "--depth", "1", "--branch", branch, repo_url, str(repo_dir)])
        run_github_git(["config", "user.email", "operator@local"], cwd=repo_dir)
        run_github_git(["config", "user.name", "operator"], cwd=repo_dir)
        run_github_git(["config", "commit.gpgsign", "false"], cwd=repo_dir)
    return repo_dir


def force_sync_github_git_repo(args: argparse.Namespace, repo: str, branch: str, reason: str) -> None:
    with locked_github_git_repo(args, repo, branch):
        repo_dir = operator_github_git_dir(args, repo)
        logging.warning(
            "Force-syncing operator Git repo %s at %s after failure: %s",
            repo,
            repo_dir,
            reason,
        )
        shutil.rmtree(repo_dir, ignore_errors=True)
        ensure_github_git_repo(args, repo, branch)


def force_sync_operator_github_repos(args: argparse.Namespace, reason: str) -> None:
    if not operator_prefers_github(args):
        return
    repos = {
        args.operator_command_repo.strip(),
        operator_output_repo(args).strip(),
    }
    for repo in sorted(repo for repo in repos if repo):
        try:
            force_sync_github_git_repo(args, repo, args.operator_github_branch, reason)
        except Exception:
            logging.exception("Operator Git force-sync failed for repo %s.", repo)


def download_github_git_text(args: argparse.Namespace, repo: str, path_in_repo: str, branch: str) -> str:
    with locked_github_git_repo(args, repo, branch):
        repo_dir = ensure_github_git_repo(args, repo, branch)
        path = repo_dir / path_in_repo
        if not path.is_file():
            raise FileNotFoundError(f"{repo}/{path_in_repo} not found in git worktree")
        return path.read_text(encoding="utf-8", errors="replace")


def upload_github_git_file(
    args: argparse.Namespace,
    repo: str,
    path_in_repo: str,
    local_path: Path,
    branch: str,
    message: str,
    max_attempts: int = 8,
) -> None:
    last_error: BaseException | None = None
    with locked_github_git_repo(args, repo, branch):
        for attempt in range(1, max_attempts + 1):
            repo_dir = ensure_github_git_repo(args, repo, branch)
            target = repo_dir / path_in_repo
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(local_path.read_bytes())
            run_github_git(["add", path_in_repo], cwd=repo_dir)
            try:
                run_github_git(["diff", "--cached", "--quiet"], cwd=repo_dir)
                return
            except Exception:
                pass
            try:
                run_github_git(["commit", "-m", message], cwd=repo_dir)
            except RuntimeError as exc:
                if "nothing to commit" not in str(exc).lower():
                    raise
                logging.warning(
                    "Git commit reported nothing to commit for %s/%s; attempting push anyway.",
                    repo,
                    path_in_repo,
                )
            try:
                run_github_git(["push", "origin", branch], cwd=repo_dir)
                return
            except Exception as exc:
                last_error = exc
                try:
                    run_github_git(["fetch", "--depth", "1", "origin", branch], cwd=repo_dir)
                    run_github_git(["rebase", f"origin/{branch}"], cwd=repo_dir)
                    run_github_git(["push", "origin", branch], cwd=repo_dir)
                    return
                except Exception as rebase_exc:
                    last_error = rebase_exc
                    try:
                        run_github_git(["rebase", "--abort"], cwd=repo_dir)
                    except Exception:
                        pass
                if attempt >= max_attempts:
                    raise GitPushConflict(str(last_error)) from last_error
                logging.warning(
                    "Git push conflict for %s/%s; refreshing worktree and retrying (%d/%d).",
                    repo,
                    path_in_repo,
                    attempt,
                    max_attempts,
                )
                shutil.rmtree(repo_dir, ignore_errors=True)
                backoff_seconds = min(45.0, 1.5 * attempt)
                time.sleep(backoff_seconds + random.uniform(0.0, 12.0))
    raise GitPushConflict(f"Git upload failed after {max_attempts} attempts: {last_error}")


def cli_has_flag(argv: list[str], flag: str) -> bool:
    return any(token == flag or token.startswith(f"{flag}=") for token in argv)


def git_commit_for_path(path: Path) -> str:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=path.parent,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        if completed.returncode == 0:
            return completed.stdout.strip()
    except Exception:
        pass
    return "unknown"


def command_to_text(command: list[str]) -> str:
    return " ".join(str(part) for part in command)


def operator_output_repo(args: argparse.Namespace) -> str:
    return args.operator_output_repo.strip() or args.operator_command_repo.strip()


def operator_node_label(args: argparse.Namespace) -> str:
    node_rank = getattr(args, "node_rank", None)
    if node_rank is not None:
        return str(node_rank)
    for name in ("GLOBAL_RANK", "NODE_RANK", "SLURM_NODEID", "RANK"):
        value = os.environ.get(name)
        if value not in {None, ""}:
            return str(value)
    return "none"


def operator_output_repo_path(args: argparse.Namespace, node_label: str) -> str:
    filename = f"output_node{node_label}.txt"
    prefix = "/".join(
        sanitize_slug_part(part)
        for part in (args.operator_output_prefix or "").strip("/").split("/")
        if part.strip()
    )
    return f"{prefix}/{filename}" if prefix else filename


def operator_inactive_output_repo_path(
    args: argparse.Namespace,
    node_label: str,
    operator_key: str,
    command_hash: str,
) -> str:
    filename = f"output_node{node_label}_inactive_{operator_key}_{command_hash[:6]}.txt"
    prefix = "/".join(
        sanitize_slug_part(part)
        for part in (args.operator_output_prefix or "").strip("/").split("/")
        if part.strip()
    )
    return f"{prefix}/{filename}" if prefix else filename


def generate_operator_key() -> str:
    return secrets.token_hex(3)


def normalize_plain_command_text(raw_text: str) -> str:
    parts: list[str] = []
    pending = ""
    for raw_line in raw_text.strip().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.endswith("\\"):
            pending += line[:-1].strip() + " "
            continue
        parts.append((pending + line).strip())
        pending = ""
    if pending.strip():
        parts.append(pending.strip())
    return " ".join(parts)


def split_leading_env_assignments(tokens: list[str]) -> tuple[dict[str, str], list[str]]:
    env: dict[str, str] = {}
    index = 0
    while index < len(tokens) and ENV_ASSIGNMENT_RE.match(tokens[index]):
        key, value = tokens[index].split("=", 1)
        env[key] = value
        index += 1
    return env, tokens[index:]


def split_leading_cwd(tokens: list[str]) -> tuple[str | None, list[str]]:
    if len(tokens) >= 4 and tokens[0] == "cd" and tokens[2] in {"&&", ";"}:
        return tokens[1], tokens[3:]
    return None, tokens


def parse_plain_operator_command(raw_text: str) -> dict[str, object]:
    text = normalize_plain_command_text(raw_text)
    if not text:
        return {"action": "noop"}
    try:
        tokens = shlex.split(text, posix=True)
    except ValueError as exc:
        raise ValueError(f"Could not parse operator command text: {exc}") from exc
    env, command_tokens = split_leading_env_assignments(tokens)
    cwd, command_tokens = split_leading_cwd(command_tokens)
    if not command_tokens:
        return {"action": "noop", "env": env, "cwd": cwd}

    command_name = Path(command_tokens[0]).name
    lower_command = command_name.lower()
    if lower_command in {"stop", "cancel"}:
        return {"action": lower_command.upper(), "env": env, "cwd": cwd}
    if lower_command in {"kill_command", "kill-command", "killcmd"}:
        if len(command_tokens) < 2:
            raise ValueError(f"{command_name} requires a command id.")
        return {
            "action": "kill_command",
            "command_id": command_tokens[1],
            "env": env,
            "cwd": cwd,
        }
    if lower_command in {"noop", "env_report", "restart_operator"}:
        return {"action": lower_command.replace("-", "_"), "env": env, "cwd": cwd}

    if lower_command.startswith("python"):
        script_name = Path(command_tokens[1]).name if len(command_tokens) >= 2 else ""
        script_args = command_tokens[2:] if len(command_tokens) >= 2 else []
    else:
        script_name = command_name
        script_args = command_tokens[1:]

    if script_name == "train.py":
        return {"action": "run_train", "args": script_args, "env": env, "cwd": cwd}
    if script_name == "train_engine.py":
        return {"action": "run_train_engine", "args": script_args, "env": env, "cwd": cwd}

    return {"action": "run_command", "command": command_tokens, "env": env, "cwd": cwd}


def parse_operator_command(raw_text: str, source_path: str = "") -> dict[str, object]:
    text = raw_text.strip()
    if not text:
        return {"action": "noop"}
    if text in {"STOP", "CANCEL"}:
        return {"action": text}
    if text[0] in "[{\"":
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            payload = None
        if payload is not None:
            if isinstance(payload, str):
                return {"action": payload}
            if not isinstance(payload, dict):
                raise ValueError("Operator command JSON must be an object or string action.")
            return payload
    if source_path.endswith(".sh"):
        non_comment_lines = [
            line.strip()
            for line in text.splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
        if len(non_comment_lines) == 1:
            control_payload = parse_plain_operator_command(non_comment_lines[0])
            control_action = str(control_payload.get("action", "")).strip()
            if control_action in {"STOP", "CANCEL", "kill_command", "restart_operator", "noop", "env_report"}:
                return control_payload
        return {"action": "run_script", "script": raw_text}
    return parse_plain_operator_command(text)


def inject_operator_run_dir_name(command_payload: dict[str, object], command_hash: str) -> None:
    action = str(command_payload.get("action", "")).strip()
    if action not in {"run_train", "dry_run_train", "run_train_engine", "dry_run_launch", "run_command", "run_script"}:
        return
    raw_env = command_payload.get("env")
    if raw_env is None:
        raw_env = {}
        command_payload["env"] = raw_env
    if not isinstance(raw_env, dict):
        return
    raw_env.setdefault("OLMO_RUN_DIR_NAME", f"cmd_{command_hash[:16]}")


def operator_env_report(args: argparse.Namespace) -> str:
    source_path = Path(__file__).resolve()
    report = {
        "node_label": operator_node_label(args),
        "host": os.uname().nodename if hasattr(os, "uname") else "unknown",
        "pid": os.getpid(),
        "cwd": os.getcwd(),
        "time": datetime.utcnow().isoformat() + "Z",
        "train_operator": str(source_path),
        "git_commit": git_commit_for_path(source_path),
        "operator_key": getattr(args, "_operator_key", None),
        "env": {
            key: os.environ.get(key)
            for key in (
                "GLOBAL_RANK",
                "NODE_RANK",
                "RANK",
                "LOCAL_RANK",
                "WORLD_SIZE",
                "MASTER_ADDR",
                "MASTER_PORT",
                "CUDA_VISIBLE_DEVICES",
            )
        },
    }
    return json.dumps(report, indent=2, sort_keys=True)


def operator_env_overrides(command_payload: dict[str, object]) -> dict[str, str]:
    raw_env = command_payload.get("env", {})
    if raw_env is None or raw_env == "":
        return {}
    if not isinstance(raw_env, dict):
        raise ValueError("Operator command env must be an object of string values.")
    env: dict[str, str] = {}
    for key, value in raw_env.items():
        if not isinstance(key, str) or not ENV_ASSIGNMENT_RE.match(f"{key}="):
            raise ValueError(f"Invalid operator env key: {key!r}")
        env[key] = str(value)
    return env


def operator_cwd(command_payload: dict[str, object]) -> str | None:
    raw_cwd = command_payload.get("cwd")
    if raw_cwd is None or raw_cwd == "":
        return None
    if not isinstance(raw_cwd, str):
        raise ValueError("Operator command cwd must be a string.")
    return raw_cwd


def run_operator_subprocess(
    command: list[str],
    timeout_seconds: int | None = None,
    env_overrides: dict[str, str] | None = None,
    cwd: str | None = None,
    live_output_path: Path | None = None,
    live_upload: Callable[[Path], None] | None = None,
    live_upload_interval_seconds: float = 0.0,
    cancel_event: threading.Event | None = None,
    on_process_started: Callable[[int], None] | None = None,
) -> tuple[int, str]:
    logging.info("Operator running command: %s", command_to_text(command))
    env = os.environ.copy()
    if env_overrides:
        env.update(env_overrides)
    for key in WRAPPER_HANDOFF_ENV_KEYS:
        env.pop(key, None)
    env.setdefault("PYTHONUNBUFFERED", "1")
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        errors="replace",
        bufsize=1,
        env=env,
        cwd=cwd,
        start_new_session=True,
    )
    if on_process_started is not None:
        try:
            on_process_started(process.pid)
        except Exception:
            logging.exception("Operator process-start callback failed; continuing child command.")
    assert process.stdout is not None

    output_queue: queue.Queue[str | None] = queue.Queue()

    def drain_stdout() -> None:
        try:
            for line in process.stdout:
                output_queue.put(line)
        finally:
            output_queue.put(None)

    reader = threading.Thread(target=drain_stdout, name="operator-subprocess-stdout", daemon=True)
    reader.start()

    capture_output = live_output_path is None
    output_parts: list[str] = []
    stdout_closed = False
    started = time.monotonic()
    last_uploaded = started
    last_uploaded_size = live_output_path.stat().st_size if live_output_path and live_output_path.exists() else 0

    def append_live_output(text: str) -> None:
        if live_output_path is None:
            return
        with live_output_path.open("a", encoding="utf-8", errors="replace") as handle:
            handle.write(text)

    def maybe_upload_live(force: bool = False) -> None:
        nonlocal last_uploaded, last_uploaded_size
        if live_output_path is None or live_upload is None or live_upload_interval_seconds <= 0:
            return
        now = time.monotonic()
        if not force and now - last_uploaded < live_upload_interval_seconds:
            return
        try:
            current_size = live_output_path.stat().st_size
        except FileNotFoundError:
            return
        if not force and current_size == last_uploaded_size:
            return
        try:
            live_upload(live_output_path)
            last_uploaded = now
            last_uploaded_size = current_size
        except Exception:
            logging.exception("Operator live output upload failed; continuing child command.")

    def terminate_process_group(reason: str, return_code: int) -> tuple[int, str]:
        logging.warning("%s; terminating operator process group pid=%s.", reason, process.pid)
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            except OSError:
                process.terminate()
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                logging.error("Operator process group did not exit after SIGTERM; sending SIGKILL.")
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                except OSError:
                    process.kill()
                process.wait(timeout=30)
        reader.join(timeout=2)
        message = f"\n{reason}\n"
        if capture_output:
            output_parts.append(message)
        else:
            append_live_output(message)
            maybe_upload_live(force=True)
        return return_code, "".join(output_parts)

    while True:
        try:
            item = output_queue.get(timeout=0.2)
        except queue.Empty:
            item = ""

        if item is None:
            stdout_closed = True
        elif item:
            if capture_output:
                output_parts.append(item)
            else:
                append_live_output(item)
            logging.info("[operator child] %s", item.rstrip())
            maybe_upload_live()

        return_code = process.poll()
        if return_code is not None and stdout_closed:
            reader.join(timeout=1)
            maybe_upload_live(force=True)
            return return_code, "".join(output_parts)

        if cancel_event is not None and cancel_event.is_set():
            return terminate_process_group("Cancelled by an operator control command", 130)

        if timeout_seconds is not None and time.monotonic() - started > timeout_seconds:
            return terminate_process_group(f"Timed out after {timeout_seconds}s", 124)


def current_train_engine_path() -> Path:
    return Path(__file__).resolve().with_name("train_engine.py")


def current_train_wrapper_path() -> Path:
    return Path(__file__).resolve().with_name("train.py")


def replacement_operator_command(args: argparse.Namespace) -> list[str]:
    command = [
        sys.executable,
        str(current_train_wrapper_path()),
        "--fetch-update",
        "--operator_mode",
        "true",
        "--operator_backend",
        str(getattr(args, "operator_backend", "github")),
        "--operator_command_repo",
        str(args.operator_command_repo),
        "--operator_command_file",
        str(args.operator_command_file),
        "--operator_key_file",
        str(args.operator_key_file),
        "--operator_github_branch",
        str(args.operator_github_branch),
        "--operator_poll_interval_seconds",
        str(args.operator_poll_interval_seconds),
        "--operator_live_upload_interval_seconds",
        str(args.operator_live_upload_interval_seconds),
        "--operator_upload_inactive_outputs",
        str(getattr(args, "operator_upload_inactive_outputs", "false")),
        "--operator_exit_on_key_mismatch",
        str(getattr(args, "operator_exit_on_key_mismatch", "true")),
        "--operator_output_upload_max_bytes",
        str(args.operator_output_upload_max_bytes),
        "--operator_work_dir",
        str(args.operator_work_dir),
        "--logdir",
        str(args.logdir),
        "--output_path",
        str(args.output_path),
    ]
    if args.operator_output_repo:
        command.extend(["--operator_output_repo", str(args.operator_output_repo)])
    if args.operator_output_prefix:
        command.extend(["--operator_output_prefix", str(args.operator_output_prefix)])
    return command


def spawn_replacement_operator(
    args: argparse.Namespace,
    env_overrides: dict[str, str] | None = None,
    cwd: str | None = None,
) -> tuple[int, Path, list[str]]:
    restart_dir = Path(args.logdir).expanduser().resolve() / "operator_restarts"
    restart_dir.mkdir(parents=True, exist_ok=True)
    node_label = operator_node_label(args)
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    log_path = restart_dir / f"operator_restart_node{node_label}_{timestamp}_pid{os.getpid()}.log"
    command = replacement_operator_command(args)
    env = os.environ.copy()
    if env_overrides:
        env.update(env_overrides)
    for key in WRAPPER_HANDOFF_ENV_KEYS:
        env.pop(key, None)
    env.setdefault("PYTHONUNBUFFERED", "1")
    with log_path.open("a", encoding="utf-8", errors="replace") as handle:
        handle.write(
            "".join(
                [
                    f"started_utc={datetime.utcnow().isoformat()}Z\n",
                    f"old_operator_pid={os.getpid()}\n",
                    f"command={command_to_text(command)}\n\n",
                ]
            )
        )
        process = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
            errors="replace",
            env=env,
            cwd=cwd,
            start_new_session=True,
            close_fds=True,
        )
    return process.pid, log_path, command


def run_operator_action(
    args: argparse.Namespace,
    command_payload: dict[str, object],
    live_output_path: Path | None = None,
    live_upload: Callable[[Path], None] | None = None,
    cancel_event: threading.Event | None = None,
    on_process_started: Callable[[int], None] | None = None,
    active_actions: dict[str, "ActiveOperatorAction"] | None = None,
) -> tuple[int, str, bool]:
    action = str(command_payload.get("action", "")).strip()
    env_overrides = operator_env_overrides(command_payload)
    cwd = operator_cwd(command_payload)
    live_upload_interval = max(0.0, float(getattr(args, "operator_live_upload_interval_seconds", 0.0)))
    if action in {"", "noop", "NOOP"}:
        return 0, "NOOP\n", False
    if action == "STOP":
        return 0, "STOP received; exiting operator loop.\n", True
    if action == "CANCEL":
        return 0, "CANCEL received; active commands were signalled.\n", False
    if action in {"kill_command", "KILL_COMMAND"}:
        return_code, output = kill_operator_command(args, command_payload, active_actions)
        return return_code, output, False
    if action in {"restart_operator", "RESTART_OPERATOR"}:
        pid, log_path, command = spawn_replacement_operator(args, env_overrides, cwd)
        output = "".join(
            [
                "Replacement operator started; current operator will stay alive and become passive "
                "after the replacement claims the key file.\n",
                f"new_operator_pid={pid}\n",
                f"new_operator_log={log_path}\n",
                f"new_operator_command={command_to_text(command)}\n",
            ]
        )
        return 0, output, False
    if action == "env_report":
        return 0, operator_env_report(args) + "\n", False
    if action in {"dry_run_launch", "run_train_engine", "run_train", "dry_run_train"}:
        argv = command_payload.get("args", [])
        if not isinstance(argv, list) or not all(isinstance(item, str) for item in argv):
            raise ValueError(f"{action} requires field 'args' as a list of strings.")
        if cli_has_flag(argv, "--operator_mode"):
            raise ValueError(f"{action} actions cannot include --operator_mode.")
        if action in {"run_train", "dry_run_train"}:
            command = [sys.executable, str(current_train_wrapper_path()), *argv]
        else:
            command = [sys.executable, str(current_train_engine_path()), *argv]
        if action in {"dry_run_launch", "dry_run_train"} and "--dry_run_launch" not in argv:
            command.append("--dry_run_launch")
        timeout_value = command_payload.get("timeout_seconds")
        timeout_seconds = int(timeout_value) if timeout_value is not None else None
        return_code, output = run_operator_subprocess(
            command,
            timeout_seconds=timeout_seconds,
            env_overrides=env_overrides,
            cwd=cwd,
            live_output_path=live_output_path,
            live_upload=live_upload,
            live_upload_interval_seconds=live_upload_interval,
            cancel_event=cancel_event,
            on_process_started=on_process_started,
        )
        return return_code, output, False
    if action == "run_script":
        script_text = command_payload.get("script")
        if not isinstance(script_text, str):
            raise ValueError("run_script requires field 'script' as a string.")
        script_dir = live_output_path.parent if live_output_path is not None else Path(tempfile.mkdtemp())
        script_dir.mkdir(parents=True, exist_ok=True)
        script_path = script_dir / "command.sh"
        script_path.write_text(script_text.rstrip() + "\n", encoding="utf-8")
        script_path.chmod(0o700)
        timeout_value = command_payload.get("timeout_seconds")
        timeout_seconds = int(timeout_value) if timeout_value is not None else None
        return_code, output = run_operator_subprocess(
            ["bash", str(script_path)],
            timeout_seconds=timeout_seconds,
            env_overrides=env_overrides,
            cwd=cwd,
            live_output_path=live_output_path,
            live_upload=live_upload,
            live_upload_interval_seconds=live_upload_interval,
            cancel_event=cancel_event,
            on_process_started=on_process_started,
        )
        return return_code, output, False
    if action == "run_command":
        command = command_payload.get("command", [])
        if not isinstance(command, list) or not all(isinstance(item, str) for item in command):
            raise ValueError("run_command requires field 'command' as a list of strings.")
        if not command:
            return 0, "NOOP: empty run_command.\n", False
        timeout_value = command_payload.get("timeout_seconds")
        timeout_seconds = int(timeout_value) if timeout_value is not None else None
        return_code, output = run_operator_subprocess(
            command,
            timeout_seconds=timeout_seconds,
            env_overrides=env_overrides,
            cwd=cwd,
            live_output_path=live_output_path,
            live_upload=live_upload,
            live_upload_interval_seconds=live_upload_interval,
            cancel_event=cancel_event,
            on_process_started=on_process_started,
        )
        return return_code, output, False
    raise ValueError(
        f"Unsupported operator action {action!r}. "
        "Allowed actions: STOP, CANCEL, restart_operator, noop, env_report, "
        "kill_command, dry_run_launch, run_train_engine, dry_run_train, run_train, run_command, run_script."
    )


def download_operator_repo_text(args: argparse.Namespace, work_dir: Path, path_in_repo: str) -> str:
    if operator_prefers_github(args):
        try:
            return download_github_git_text(
                args,
                args.operator_command_repo,
                path_in_repo,
                args.operator_github_branch,
            )
        except Exception as git_exc:
            if getattr(args, "operator_backend", "github") != "auto":
                raise
            logging.warning(
                "Git operator download failed for %s/%s; falling back to HF because --operator_backend auto: %s",
                args.operator_command_repo,
                path_in_repo,
                git_exc,
            )
    return download_hf_operator_text(args.operator_command_repo, path_in_repo, work_dir)


def download_hf_operator_text(repo_id: str, path_in_repo: str, work_dir: Path) -> str:
    from huggingface_hub import hf_hub_download

    local_path = hf_hub_download(
        repo_id=repo_id,
        filename=path_in_repo,
        repo_type="dataset",
        cache_dir=str(work_dir / "hf_cache"),
        force_download=True,
        token=os.environ.get("HF_TOKEN"),
    )
    return Path(local_path).read_text(encoding="utf-8")


def upload_operator_repo_text(
    args: argparse.Namespace,
    path_in_repo: str,
    text: str,
    message: str,
    max_attempts: int = 5,
) -> None:
    if operator_prefers_github(args):
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False) as handle:
            handle.write(text)
            tmp_path = Path(handle.name)
        try:
            upload_github_git_file(
                args,
                args.operator_command_repo,
                path_in_repo,
                tmp_path,
                args.operator_github_branch,
                message,
                max_attempts=max_attempts,
            )
            return
        except Exception as git_exc:
            if getattr(args, "operator_backend", "github") != "auto":
                raise
            logging.warning(
                "Git operator upload failed for %s/%s; falling back to HF because --operator_backend auto: %s",
                args.operator_command_repo,
                path_in_repo,
                git_exc,
            )
            upload_hf_operator_file(args.operator_command_repo, path_in_repo, tmp_path, message)
            return
        finally:
            tmp_path.unlink(missing_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False) as handle:
        handle.write(text)
        tmp_path = Path(handle.name)
    try:
        upload_hf_operator_file(args.operator_command_repo, path_in_repo, tmp_path, message)
    finally:
        tmp_path.unlink(missing_ok=True)


def upload_hf_operator_file(repo_id: str, path_in_repo: str, local_path: Path, message: str) -> None:
    from huggingface_hub import HfApi

    api = HfApi(token=os.environ.get("HF_TOKEN"))
    api.create_repo(repo_id=repo_id, repo_type="dataset", private=True, exist_ok=True)
    api.upload_file(
        repo_id=repo_id,
        repo_type="dataset",
        path_or_fileobj=str(local_path),
        path_in_repo=path_in_repo,
        commit_message=message,
    )


def download_operator_command(args: argparse.Namespace, work_dir: Path) -> str:
    return download_operator_repo_text(args, work_dir, args.operator_command_file)


def parse_operator_key_registry(raw_text: str, node_label: str) -> dict[str, object]:
    text = raw_text.strip()
    if not text:
        return {"version": 1, "operators": {}}
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return {
            "version": 1,
            "operators": {
                node_label: {
                    "key": text,
                    "format": "legacy_plain_text",
                }
            },
        }
    if not isinstance(payload, dict):
        return {"version": 1, "operators": {}}
    operators = payload.get("operators")
    if not isinstance(operators, dict):
        payload["operators"] = {}
    return payload


def operator_key_registry_text(registry: dict[str, object]) -> str:
    return json.dumps(registry, indent=2, sort_keys=True) + "\n"


def download_operator_key_registry(
    args: argparse.Namespace,
    work_dir: Path,
    node_label: str,
) -> dict[str, object]:
    try:
        raw_text = download_operator_repo_text(args, work_dir, args.operator_key_file)
    except Exception as exc:
        if (
            "HTTP 404" not in str(exc)
            and "EntryNotFound" not in type(exc).__name__
            and not isinstance(exc, FileNotFoundError)
            and "not found in git worktree" not in str(exc)
        ):
            raise
        return {"version": 1, "operators": {}}
    return parse_operator_key_registry(raw_text, node_label)


def operator_registry_key_for_node(registry: dict[str, object], node_label: str) -> str | None:
    raw_record = operator_registry_record_for_node(registry, node_label)
    if isinstance(raw_record, str):
        return raw_record
    if isinstance(raw_record, dict):
        raw_key = raw_record.get("key")
        if isinstance(raw_key, str) and raw_key:
            return raw_key
    return None


def operator_registry_record_for_node(
    registry: dict[str, object],
    node_label: str,
) -> object | None:
    operators = registry.get("operators")
    if not isinstance(operators, dict):
        return None
    return operators.get(node_label)


def parse_operator_started_utc(value: object) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip()
    if text.endswith("Z"):
        text = f"{text[:-1]}+00:00"
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


def operator_record_started_utc(raw_record: object) -> str | None:
    if isinstance(raw_record, dict):
        value = raw_record.get("started_utc")
        if isinstance(value, str) and value:
            return value
    return None


def operator_record_is_stale_for_local(
    raw_record: object | None,
    operator_started_utc: str,
) -> bool:
    if raw_record is None:
        return True
    remote_started_utc = operator_record_started_utc(raw_record)
    if not remote_started_utc:
        return True
    local_started = parse_operator_started_utc(operator_started_utc)
    remote_started = parse_operator_started_utc(remote_started_utc)
    if local_started is None or remote_started is None:
        return operator_started_utc > remote_started_utc
    return local_started > remote_started


def operator_key_record(node_label: str, operator_key: str, started_utc: str) -> dict[str, object]:
    return {
        "key": operator_key,
        "host": os.uname().nodename if hasattr(os, "uname") else "unknown",
        "pid": os.getpid(),
        "started_utc": started_utc,
        "train_operator": str(Path(__file__).resolve()),
        "git_commit": git_commit_for_path(Path(__file__).resolve()),
    }


def set_operator_key_record(
    registry: dict[str, object],
    node_label: str,
    operator_key: str,
    started_utc: str,
) -> None:
    operators = registry.setdefault("operators", {})
    if not isinstance(operators, dict):
        operators = {}
        registry["operators"] = operators
    registry["version"] = 1
    registry["updated_utc"] = datetime.utcnow().isoformat() + "Z"
    operators[node_label] = operator_key_record(node_label, operator_key, started_utc)


def claim_operator_key(
    args: argparse.Namespace,
    work_dir: Path,
    node_label: str,
    operator_key: str,
    started_utc: str,
    max_attempts: int = 20,
) -> None:
    last_error: BaseException | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            registry = download_operator_key_registry(args, work_dir, node_label)
            set_operator_key_record(registry, node_label, operator_key, started_utc)
            upload_operator_repo_text(
                args,
                args.operator_key_file,
                operator_key_registry_text(registry),
                f"Claim operator key for node {node_label}",
            )
            return
        except Exception as exc:
            last_error = exc
            if attempt >= max_attempts:
                break
            time.sleep(min(5.0, 0.5 * attempt))
    raise RuntimeError(
        f"Failed to claim operator key for node {node_label} after {max_attempts} attempts: {last_error}"
    )


def remote_operator_key(
    args: argparse.Namespace,
    work_dir: Path,
    node_label: str,
) -> str | None:
    registry = download_operator_key_registry(args, work_dir, node_label)
    return operator_registry_key_for_node(registry, node_label)


def active_or_reclaimed_operator_key(
    args: argparse.Namespace,
    work_dir: Path,
    node_label: str,
    operator_key: str,
    operator_started_utc: str,
) -> str | None:
    registry = download_operator_key_registry(args, work_dir, node_label)
    raw_record = operator_registry_record_for_node(registry, node_label)
    active_key = operator_registry_key_for_node(registry, node_label)
    if active_key == operator_key:
        return active_key
    if not operator_record_is_stale_for_local(raw_record, operator_started_utc):
        return active_key
    logging.warning(
        (
            "Reclaiming stale operator key for node=%s; local_key=%s local_started=%s "
            "remote_key=%s remote_started=%s"
        ),
        node_label,
        operator_key,
        operator_started_utc,
        active_key or "<missing>",
        operator_record_started_utc(raw_record) or "<missing>",
    )
    claim_operator_key(args, work_dir, node_label, operator_key, operator_started_utc)
    return operator_key


def upload_operator_output(
    args: argparse.Namespace,
    output_path: Path,
    node_label: str,
    path_in_repo: str | None = None,
) -> None:
    target_path = path_in_repo or operator_output_repo_path(args, node_label)
    repo_id = operator_output_repo(args)
    message = f"Update operator output for node {node_label}"
    upload_path = prepare_operator_output_upload_snapshot(
        output_path,
        max_bytes=int(getattr(args, "operator_output_upload_max_bytes", 0)),
    )
    try:
        if operator_prefers_github(args):
            try:
                upload_github_git_file(
                    args,
                    repo_id,
                    target_path,
                    upload_path,
                    args.operator_github_branch,
                    message,
                    max_attempts=20,
                )
                return
            except Exception as git_exc:
                if getattr(args, "operator_backend", "github") != "auto":
                    raise
                logging.warning(
                    "Git operator output upload failed for %s/%s; falling back to HF because --operator_backend auto: %s",
                    repo_id,
                    target_path,
                    git_exc,
                )
        upload_hf_operator_file(repo_id, target_path, upload_path, message)
    finally:
        if upload_path != output_path:
            upload_path.unlink(missing_ok=True)


def upload_operator_poll_error(
    args: argparse.Namespace,
    work_dir: Path,
    node_label: str,
    operator_key: str,
    exc: BaseException,
) -> None:
    now = datetime.utcnow()
    output_path = work_dir / f"output_node{node_label}_poll_error.txt"
    output_path.write_text(
        "".join(
            [
                f"node={node_label}\n",
                f"host={os.uname().nodename if hasattr(os, 'uname') else 'unknown'}\n",
                f"operator_key={operator_key}\n",
                f"operator_pid={os.getpid()}\n",
                f"time_utc={now.isoformat()}Z\n",
                f"error_type={type(exc).__name__}\n",
                f"error={exc}\n\n",
                traceback.format_exc(),
            ]
        ),
        encoding="utf-8",
    )
    upload_operator_output(
        args,
        output_path,
        node_label,
        path_in_repo=operator_poll_error_output_repo_path(args, node_label),
    )


def recover_operator_poll_failure(
    args: argparse.Namespace,
    work_dir: Path,
    node_label: str,
    operator_key: str,
    exc: BaseException,
    upload_lock: threading.Lock,
    last_poll_error_upload: float,
) -> float:
    now = time.monotonic()
    reason = f"{type(exc).__name__}: {exc}"
    with upload_lock:
        # Always rebuild the local Git checkout after a poll/upload failure. The
        # operator must stay alive even when git index/lock/push state is corrupt.
        force_sync_operator_github_repos(args, reason=reason)
        if now - last_poll_error_upload >= 60.0:
            upload_operator_poll_error(
                args,
                work_dir,
                node_label,
                operator_key,
                exc,
            )
            return now
    return last_poll_error_upload


def prepare_operator_output_upload_snapshot(output_path: Path, max_bytes: int) -> Path:
    if max_bytes <= 0:
        return output_path
    try:
        file_size = output_path.stat().st_size
    except FileNotFoundError:
        return output_path
    if file_size <= max_bytes:
        return output_path

    header_bytes = min(256 * 1024, max_bytes // 4)
    marker = (
        "\n\n[operator output truncated for remote upload]\n"
        f"local_file={output_path}\n"
        f"local_size_bytes={file_size}\n"
        f"remote_max_bytes={max_bytes}\n"
        "[showing file header and most recent tail]\n\n"
    ).encode("utf-8")
    tail_bytes = max(0, max_bytes - header_bytes - len(marker))
    snapshot_dir = output_path.parent / ".upload_snapshots"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    snapshot_path = snapshot_dir / (
        f"{output_path.name}.pid{os.getpid()}.thread{threading.get_ident()}.tmp"
    )
    with output_path.open("rb") as source, snapshot_path.open("wb") as destination:
        destination.write(source.read(header_bytes))
        destination.write(marker)
        if tail_bytes > 0:
            source.seek(max(0, file_size - tail_bytes))
            destination.write(source.read(tail_bytes))
    return snapshot_path


def operator_command_output_repo_path(
    args: argparse.Namespace,
    node_label: str,
    command_hash: str,
) -> str:
    filename = f"output_node{node_label}_{command_hash[:6]}.txt"
    parts = [
        *(part for part in (args.operator_output_prefix or "").strip("/").split("/") if part),
        filename,
    ]
    return "/".join(sanitize_slug_part(part) for part in parts)


def operator_poll_error_output_repo_path(args: argparse.Namespace, node_label: str) -> str:
    filename = f"output_node{node_label}_poll_error.txt"
    parts = [
        *(part for part in (args.operator_output_prefix or "").strip("/").split("/") if part),
        filename,
    ]
    return "/".join(sanitize_slug_part(part) for part in parts)


@dataclass
class ActiveOperatorAction:
    command_hash: str
    command_payload: dict[str, object]
    started: datetime
    output_path: Path
    cancel_event: threading.Event
    thread: threading.Thread | None = None
    process_pid: int | None = None
    return_code: int | None = None
    output: str = ""
    should_stop: bool = False
    error: BaseException | None = None
    finalized: bool = False


def normalize_operator_command_id(command_id: object) -> str:
    value = str(command_id or "").strip().lower()
    if len(value) < 6 or re.fullmatch(r"[0-9a-f]+", value) is None:
        raise ValueError("kill_command requires command_id with at least 6 hexadecimal characters.")
    return value[:6]


def command_hash_matches_id(command_hash: str, command_id: str) -> bool:
    return command_hash.lower().startswith(command_id)


def process_rows() -> list[tuple[int, int, int, str, str, str]]:
    rows = subprocess.check_output(
        ["ps", "-eo", "pid=,ppid=,pgid=,stat=,etime=,cmd="],
        text=True,
        errors="replace",
    ).splitlines()
    parsed: list[tuple[int, int, int, str, str, str]] = []
    for row in rows:
        parts = row.strip().split(None, 5)
        if len(parts) < 6:
            continue
        pid_s, ppid_s, pgid_s, stat, etime, cmd = parts
        try:
            parsed.append((int(pid_s), int(ppid_s), int(pgid_s), stat, etime, cmd))
        except ValueError:
            continue
    return parsed


def process_environment(pid: int) -> str:
    try:
        return (
            (Path("/proc") / str(pid) / "environ")
            .read_bytes()
            .replace(b"\0", b" ")
            .decode("utf-8", "replace")
        )
    except (FileNotFoundError, PermissionError, ProcessLookupError, OSError):
        return ""


def process_descendant_pids(root_pids: Iterable[int]) -> set[int]:
    children_by_parent: dict[int, list[int]] = {}
    for pid, ppid, _pgid, _stat, _etime, _cmd in process_rows():
        children_by_parent.setdefault(ppid, []).append(pid)
    pending = list(root_pids)
    descendants: set[int] = set()
    while pending:
        parent = pending.pop()
        for child in children_by_parent.get(parent, []):
            if child in descendants:
                continue
            descendants.add(child)
            pending.append(child)
    return descendants


def operator_process_should_be_preserved(cmd: str) -> bool:
    return "--operator_mode" in cmd or "train_operator.py" in cmd or "operator_client.py" in cmd


def scan_command_processes(
    command_id: str,
    extra_identifiers: Iterable[str] = (),
    descendant_pids: Iterable[int] = (),
) -> list[tuple[int, int, int, str, str, str]]:
    own_pid = os.getpid()
    parent_pid = os.getppid()
    try:
        own_pgid = os.getpgrp()
    except Exception:
        own_pgid = -1
    identifiers = {
        ident.lower()
        for ident in (command_id, *extra_identifiers)
        if isinstance(ident, str) and len(ident.strip()) >= 6
    }
    descendants = set(descendant_pids)
    matches: list[tuple[int, int, int, str, str, str]] = []
    for pid, ppid, pgid, stat, etime, cmd in process_rows():
        if pid in {own_pid, parent_pid} or ppid in {own_pid, parent_pid} or pgid == own_pgid:
            continue
        if operator_process_should_be_preserved(cmd):
            continue
        cmd_haystack = cmd.lower()
        if pid not in descendants and not any(ident in cmd_haystack for ident in identifiers):
            env_haystack = process_environment(pid).lower()
            if not any(ident in env_haystack for ident in identifiers):
                continue
        matches.append((pid, ppid, pgid, stat, etime, cmd))
    return matches


def kill_command_identifiers(
    command_id: str,
    matching_active: list[ActiveOperatorAction],
    extra_identifiers: Iterable[str] = (),
) -> set[str]:
    identifiers = {command_id}
    identifiers.update(str(ident) for ident in extra_identifiers if len(str(ident).strip()) >= 6)
    for active in matching_active:
        identifiers.add(active.command_hash)
        identifiers.add(active.command_hash[:12])
        env = active.command_payload.get("env")
        if isinstance(env, dict):
            run_name = env.get("OLMO_RUN_DIR_NAME")
            if isinstance(run_name, str):
                identifiers.add(run_name)
        command = active.command_payload.get("command")
        if isinstance(command, list):
            identifiers.update(str(part) for part in command if len(str(part)) >= 12)
        args = active.command_payload.get("args")
        if isinstance(args, list):
            identifiers.update(str(part) for part in args if len(str(part)) >= 12)
    return identifiers


def signal_process_group(pid: int, sig: signal.Signals, output: list[str]) -> None:
    try:
        pgid = os.getpgid(pid)
    except ProcessLookupError:
        return
    except OSError as exc:
        output.append(f"could not resolve process group for pid={pid}: {exc}\n")
        return
    try:
        os.killpg(pgid, sig)
        output.append(f"sent {sig.name} pgid={pgid} from pid={pid}\n")
    except ProcessLookupError:
        return
    except PermissionError as exc:
        output.append(f"permission_error {sig.name} pgid={pgid}: {exc}\n")
    except OSError as exc:
        output.append(f"error {sig.name} pgid={pgid}: {exc}\n")


def local_operator_process_matches(args: argparse.Namespace, pid: int, cmd: str) -> bool:
    if pid == os.getpid():
        return False
    if "--operator_mode" not in cmd and "train_operator.py" not in cmd:
        return False
    repo = str(getattr(args, "operator_command_repo", "") or "")
    command_file = str(getattr(args, "operator_command_file", "") or "")
    if repo and repo not in cmd:
        return False
    if command_file and command_file not in cmd:
        return False
    return True


def retire_stale_local_operator_processes(
    args: argparse.Namespace,
    work_dir: Path,
    node_label: str,
    operator_key: str,
) -> None:
    delay_seconds = float(os.environ.get("OLMO_OPERATOR_RETIRE_STALE_DELAY_SECONDS", "8"))
    if delay_seconds <= 0:
        return
    time.sleep(delay_seconds)
    active_key = remote_operator_key(args, work_dir, node_label)
    if active_key != operator_key:
        logging.info(
            "Skipping stale-operator retirement because local key is no longer active: node=%s local=%s remote=%s",
            node_label,
            operator_key,
            active_key or "<missing>",
        )
        return

    candidates = [
        (pid, stat, etime, cmd)
        for pid, _ppid, _pgid, stat, etime, cmd in process_rows()
        if local_operator_process_matches(args, pid, cmd)
    ]
    if not candidates:
        logging.info("No stale local operator processes found for node=%s key=%s.", node_label, operator_key)
        return

    logging.warning(
        "Retiring %d stale local operator process(es) for node=%s active_key=%s.",
        len(candidates),
        node_label,
        operator_key,
    )
    for pid, stat, etime, cmd in candidates:
        logging.warning("Sending SIGTERM to stale operator pid=%s stat=%s etime=%s cmd=%s", pid, stat, etime, cmd[:500])
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            continue
        except PermissionError as exc:
            logging.warning("Permission error sending SIGTERM to stale operator pid=%s: %s", pid, exc)
        except OSError as exc:
            logging.warning("Error sending SIGTERM to stale operator pid=%s: %s", pid, exc)
    time.sleep(2.0)
    remaining = [
        (pid, stat, etime, cmd)
        for pid, _ppid, _pgid, stat, etime, cmd in process_rows()
        if local_operator_process_matches(args, pid, cmd)
    ]
    for pid, stat, etime, cmd in remaining:
        logging.warning("Sending SIGKILL to stale operator pid=%s stat=%s etime=%s cmd=%s", pid, stat, etime, cmd[:500])
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            continue
        except PermissionError as exc:
            logging.warning("Permission error sending SIGKILL to stale operator pid=%s: %s", pid, exc)
        except OSError as exc:
            logging.warning("Error sending SIGKILL to stale operator pid=%s: %s", pid, exc)


def kill_operator_command(
    args: argparse.Namespace,
    command_payload: dict[str, object],
    active_actions: dict[str, ActiveOperatorAction] | None,
) -> tuple[int, str]:
    command_id = normalize_operator_command_id(command_payload.get("command_id"))
    grace_seconds = float(command_payload.get("grace_seconds", 10.0) or 0.0)
    force = bool(command_payload.get("force", True))
    raw_extra_identifiers = command_payload.get("extra_identifiers", [])
    if isinstance(raw_extra_identifiers, str):
        extra_identifiers = [raw_extra_identifiers]
    elif isinstance(raw_extra_identifiers, list) and all(
        isinstance(item, str) for item in raw_extra_identifiers
    ):
        extra_identifiers = raw_extra_identifiers
    else:
        raise ValueError("kill_command extra_identifiers must be a string or list of strings.")
    node_label = operator_node_label(args)
    lines = [
        f"kill_command target_command_id={command_id} node={node_label} host={os.uname().nodename}\n"
    ]

    matching_active = [
        active
        for active in (active_actions or {}).values()
        if command_hash_matches_id(active.command_hash, command_id)
    ]
    lines.append(f"active_matches={len(matching_active)}\n")
    for active in matching_active:
        lines.append(
            (
                f"active command_hash={active.command_hash[:12]} "
                f"process_pid={active.process_pid or '<unknown>'} "
                f"started_utc={active.started.isoformat()}Z\n"
            )
        )
        active.cancel_event.set()
        if active.process_pid is not None:
            signal_process_group(active.process_pid, signal.SIGTERM, lines)

    identifiers = kill_command_identifiers(command_id, matching_active, extra_identifiers)
    descendant_pids = process_descendant_pids(
        active.process_pid for active in matching_active if active.process_pid is not None
    )
    lines.append(f"scan_identifiers={','.join(sorted(identifiers))[:1000]}\n")
    lines.append(f"descendant_pids={len(descendant_pids)}\n")

    scanned_before = scan_command_processes(command_id, identifiers, descendant_pids)
    lines.append(f"scanned_matches_before={len(scanned_before)}\n")
    signalled_pids: set[int] = set()
    for pid, _ppid, _pgid, stat, etime, cmd in scanned_before:
        if pid in signalled_pids:
            continue
        signalled_pids.add(pid)
        lines.append(f"scan_target pid={pid} stat={stat} etime={etime} cmd={cmd[:500]}\n")
        signal_process_group(pid, signal.SIGTERM, lines)

    if grace_seconds > 0:
        time.sleep(grace_seconds)
    for active in matching_active:
        if active.thread is not None:
            active.thread.join(timeout=2)

    scanned_after_term = scan_command_processes(command_id, identifiers, descendant_pids)
    live_active = [active for active in matching_active if active.thread is not None and active.thread.is_alive()]
    lines.append(
        f"after_sigterm active_alive={len(live_active)} scanned_matches={len(scanned_after_term)}\n"
    )

    if force:
        for active in live_active:
            if active.process_pid is not None:
                signal_process_group(active.process_pid, signal.SIGKILL, lines)
        signalled_pids.clear()
        for pid, _ppid, _pgid, stat, etime, cmd in scanned_after_term:
            if pid in signalled_pids:
                continue
            signalled_pids.add(pid)
            lines.append(f"scan_force_target pid={pid} stat={stat} etime={etime} cmd={cmd[:500]}\n")
            signal_process_group(pid, signal.SIGKILL, lines)
        time.sleep(1.0)
        for active in matching_active:
            if active.thread is not None:
                active.thread.join(timeout=2)

    scanned_final = scan_command_processes(command_id, identifiers, descendant_pids)
    live_active_final = [
        active for active in matching_active if active.thread is not None and active.thread.is_alive()
    ]
    lines.append(
        f"final active_alive={len(live_active_final)} scanned_matches={len(scanned_final)}\n"
    )
    for pid, _ppid, _pgid, stat, etime, cmd in scanned_final:
        lines.append(f"remain pid={pid} stat={stat} etime={etime} cmd={cmd[:500]}\n")

    try:
        smi = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            errors="replace",
            check=False,
            timeout=20,
        )
        lines.append("nvidia_smi=\n")
        lines.append(smi.stdout)
        if smi.stdout and not smi.stdout.endswith("\n"):
            lines.append("\n")
    except Exception as exc:
        lines.append(f"nvidia_smi_failed={type(exc).__name__}: {exc}\n")

    return_code = 0 if not live_active_final and not scanned_final else 1
    return return_code, "".join(lines)


def load_operator_command_hash_state(state_path: Path) -> tuple[str | None, set[str]]:
    try:
        hashes = [
            line.strip()
            for line in state_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except FileNotFoundError:
        return None, set()
    if len(hashes) == 1:
        return hashes[0], set()
    return None, set(hashes)


def persist_operator_command_hash(state_path: Path, command_hash: str) -> None:
    state_path.write_text(command_hash + "\n", encoding="utf-8")


def start_operator_action(
    args: argparse.Namespace,
    command_hash: str,
    command_payload: dict[str, object],
    node_label: str,
    work_dir: Path,
    latest_command_hash: dict[str, str | None],
    upload_lock: threading.Lock,
    active_actions: dict[str, ActiveOperatorAction] | None = None,
) -> ActiveOperatorAction:
    operator_key = sanitize_slug_part(str(getattr(args, "_operator_key", "nokey")))
    output_path = (
        work_dir
        / "commands"
        / command_hash
        / f"output_node{node_label}_{operator_key}_pid{os.getpid()}_{command_hash[:6]}.txt"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    started = datetime.utcnow()
    output_path.write_text(
        "".join(
            [
                f"node={node_label}\n",
                f"host={os.uname().nodename if hasattr(os, 'uname') else 'unknown'}\n",
                f"started_utc={started.isoformat()}Z\n",
                f"command_hash={command_hash}\n",
                f"operator_key={operator_key}\n",
                f"operator_pid={os.getpid()}\n",
                "command_payload=\n",
                json.dumps(command_payload, indent=2, sort_keys=True),
                "\n\noutput=\n",
            ]
        ),
        encoding="utf-8",
    )
    live_upload_interval = max(0.0, float(args.operator_live_upload_interval_seconds))

    def live_upload(path: Path) -> None:
        with upload_lock:
            upload_operator_output(
                args,
                path,
                node_label,
                path_in_repo=operator_command_output_repo_path(args, node_label, command_hash),
            )
            if latest_command_hash["value"] == command_hash:
                upload_operator_output(args, path, node_label)

    if live_upload_interval > 0:
        try:
            live_upload(output_path)
        except Exception:
            logging.exception("Initial operator output upload failed; command will still run.")

    active = ActiveOperatorAction(
        command_hash=command_hash,
        command_payload=command_payload,
        started=started,
        output_path=output_path,
        cancel_event=threading.Event(),
    )

    def set_process_pid(pid: int) -> None:
        active.process_pid = pid

    def run_action() -> None:
        try:
            active.return_code, active.output, active.should_stop = run_operator_action(
                args,
                command_payload,
                live_output_path=output_path,
                live_upload=live_upload if live_upload_interval > 0 else None,
                cancel_event=active.cancel_event,
                on_process_started=set_process_pid,
                active_actions=active_actions,
            )
        except BaseException as exc:
            active.error = exc
            logging.exception("Operator action failed.")

    active.thread = threading.Thread(
        target=run_action,
        name=f"operator-action-{command_hash[:8]}",
        daemon=True,
    )
    active.thread.start()
    return active


def finalize_operator_action(
    args: argparse.Namespace,
    active: ActiveOperatorAction,
    node_label: str,
    latest_command_hash: dict[str, str | None],
    upload_lock: threading.Lock,
) -> None:
    assert active.thread is not None
    if active.thread.is_alive():
        raise RuntimeError("Cannot finalize an operator action while it is still running.")
    if not active.finalized:
        if active.error is not None:
            active.return_code = 1
            active.output = f"{type(active.error).__name__}: {active.error}\n"
        finished = datetime.utcnow()
        with active.output_path.open("a", encoding="utf-8", errors="replace") as handle:
            handle.write(
                "".join(
                    [
                        active.output if active.output else "",
                        "\n",
                        f"return_code={active.return_code if active.return_code is not None else 1}\n",
                        f"finished_utc={finished.isoformat()}Z\n",
                        f"duration_seconds={(finished - active.started).total_seconds():.3f}\n",
                    ]
                )
            )
        active.finalized = True
    with upload_lock:
        upload_operator_output(
            args,
            active.output_path,
            node_label,
            path_in_repo=operator_command_output_repo_path(
                args, node_label, active.command_hash
            ),
        )
        if latest_command_hash["value"] == active.command_hash:
            upload_operator_output(args, active.output_path, node_label)


def upload_inactive_operator_output(
    args: argparse.Namespace,
    work_dir: Path,
    node_label: str,
    operator_key: str,
    remote_key: str | None,
    command_hash: str,
    command_payload: dict[str, object],
    upload_lock: threading.Lock,
) -> None:
    if getattr(args, "operator_upload_inactive_outputs", "false") != "true":
        logging.info(
            (
                "Skipping inactive operator output upload for node=%s local_key=%s remote_key=%s "
                "command=%s because --operator_upload_inactive_outputs=false."
            ),
            node_label,
            operator_key,
            remote_key or "<missing>",
            command_hash[:12],
        )
        return
    output_path = (
        work_dir
        / "commands"
        / command_hash
        / f"output_node{node_label}_inactive_{operator_key}_{command_hash[:6]}.txt"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    now = datetime.utcnow()
    output_path.write_text(
        "".join(
            [
                f"node={node_label}\n",
                f"host={os.uname().nodename if hasattr(os, 'uname') else 'unknown'}\n",
                f"started_utc={now.isoformat()}Z\n",
                f"finished_utc={now.isoformat()}Z\n",
                f"command_hash={command_hash}\n",
                f"operator_key={operator_key}\n",
                f"remote_key={remote_key or '<missing>'}\n",
                "command_payload=\n",
                json.dumps(command_payload, indent=2, sort_keys=True),
                "\n\noutput=\n",
                "Operator key mismatch; this older operator skipped the command and stayed alive "
                "to observe STOP.\n",
                "return_code=0\n",
                "duration_seconds=0.000\n",
            ]
        ),
        encoding="utf-8",
    )
    with upload_lock:
        upload_operator_output(
            args,
            output_path,
            node_label,
            path_in_repo=operator_inactive_output_repo_path(
                args,
                node_label,
                operator_key,
                command_hash,
            ),
        )


def run_operator_mode(args: argparse.Namespace) -> None:
    logdir = Path(args.logdir).expanduser().resolve()
    configure_logging(logdir, args)
    logging.info(
        "Training source: train_operator=%s git_commit=%s",
        Path(__file__).resolve(),
        git_commit_for_path(Path(__file__).resolve()),
    )
    log_dependency_versions()
    node_label = operator_node_label(args)
    work_dir = Path(args.operator_work_dir).expanduser() / f"node{node_label}"
    work_dir.mkdir(parents=True, exist_ok=True)
    operator_key = generate_operator_key()
    operator_started_utc = datetime.utcnow().isoformat() + "Z"
    args._operator_key = operator_key
    claim_operator_key(args, work_dir, node_label, operator_key, operator_started_utc)
    retire_stale_local_operator_processes(args, work_dir, node_label, operator_key)
    state_path = work_dir / f"last_command_hash_{operator_key}.txt"
    previous_command_hash, legacy_seen_command_hashes = load_operator_command_hash_state(
        state_path
    )
    active_actions: dict[str, ActiveOperatorAction] = {}
    latest_command_hash: dict[str, str | None] = {"value": None}
    upload_lock = threading.Lock()
    poll_interval = max(0.5, float(args.operator_poll_interval_seconds))
    last_poll_error_upload = 0.0
    logging.info(
        (
            "Starting operator mode: backend=%s command_repo=%s command_file=%s output_repo=%s output_path=%s "
            "key_file=%s operator_key=%s poll_interval=%.2fs live_upload_interval=%.2fs node=%s"
        ),
        getattr(args, "operator_backend", "github"),
        args.operator_command_repo,
        args.operator_command_file,
        operator_output_repo(args),
        operator_output_repo_path(args, node_label),
        args.operator_key_file,
        operator_key,
        poll_interval,
        max(0.0, float(args.operator_live_upload_interval_seconds)),
        node_label,
    )
    while True:
        try:
            for active_hash, active in list(active_actions.items()):
                assert active.thread is not None
                if active.thread.is_alive():
                    continue
                finalize_operator_action(
                    args,
                    active,
                    node_label,
                    latest_command_hash,
                    upload_lock,
                )
                del active_actions[active_hash]

            raw_command = download_operator_command(args, work_dir)
            command_hash = hashlib.sha256(raw_command.encode("utf-8")).hexdigest()
            is_legacy_seen_command = command_hash in legacy_seen_command_hashes
            if (
                command_hash == previous_command_hash or is_legacy_seen_command
            ) and raw_command.strip() != "STOP":
                if is_legacy_seen_command:
                    previous_command_hash = command_hash
                    legacy_seen_command_hashes.clear()
                    persist_operator_command_hash(state_path, command_hash)
                time.sleep(poll_interval)
                continue

            command_payload = parse_operator_command(raw_command, args.operator_command_file)
            inject_operator_run_dir_name(command_payload, command_hash)
            action = str(command_payload.get("action", "")).strip().upper()
            is_global_control_action = action in {"STOP", "CANCEL", "KILL_COMMAND", "RESTART_OPERATOR"}
            previous_command_hash = command_hash
            legacy_seen_command_hashes.clear()
            persist_operator_command_hash(state_path, command_hash)
            latest_command_hash["value"] = command_hash
            if not is_global_control_action:
                active_key = active_or_reclaimed_operator_key(
                    args,
                    work_dir,
                    node_label,
                    operator_key,
                    operator_started_utc,
                )
                if active_key != operator_key:
                    logging.warning(
                        (
                            "Operator key mismatch for node=%s; local_key=%s remote_key=%s. "
                            "Skipping command %s action=%s and staying alive for control commands."
                        ),
                        node_label,
                        operator_key,
                        active_key or "<missing>",
                        command_hash[:12],
                        action or "<empty>",
                    )
                    upload_inactive_operator_output(
                        args,
                        work_dir,
                        node_label,
                        operator_key,
                        active_key,
                        command_hash,
                        command_payload,
                        upload_lock,
                    )
                    if getattr(args, "operator_exit_on_key_mismatch", "true") == "true":
                        logging.warning(
                            "Operator key mismatch for node=%s; exiting stale operator local_key=%s remote_key=%s.",
                            node_label,
                            operator_key,
                            active_key or "<missing>",
                        )
                        return
                    time.sleep(poll_interval)
                    continue

            if action in {"CANCEL", "STOP"}:
                logging.warning(
                    "%s received; signalling %d active operator command(s).",
                    action,
                    len(active_actions),
                )
                for active in active_actions.values():
                    active.cancel_event.set()

            control_action = start_operator_action(
                args,
                command_hash,
                command_payload,
                node_label,
                work_dir,
                latest_command_hash,
                upload_lock,
                active_actions=active_actions,
            )
            if action in {"CANCEL", "STOP", "RESTART_OPERATOR", "KILL_COMMAND"}:
                assert control_action.thread is not None
                control_timeout = 45 if action == "KILL_COMMAND" else 10
                control_action.thread.join(timeout=control_timeout)
                if control_action.thread.is_alive():
                    raise RuntimeError(f"Operator control action {action} did not finish promptly.")
                finalize_operator_action(
                    args,
                    control_action,
                    node_label,
                    latest_command_hash,
                    upload_lock,
                )
            else:
                active_actions[command_hash] = control_action
                logging.info(
                    "Started operator command %s concurrently; active_commands=%d.",
                    command_hash[:12],
                    len(active_actions),
                )

            if action == "STOP":
                for active in active_actions.values():
                    assert active.thread is not None
                    active.thread.join(timeout=65)
                    if active.thread.is_alive():
                        raise RuntimeError(
                            f"Active operator command {active.command_hash[:12]} did not stop after STOP."
                        )
                    finalize_operator_action(
                        args,
                        active,
                        node_label,
                        latest_command_hash,
                        upload_lock,
                    )
                active_actions.clear()
                logging.info("Operator mode exiting after STOP command.")
                return
            time.sleep(poll_interval)
        except Exception as exc:
            logging.exception("Operator poll/upload cycle failed; retrying after %.2fs.", poll_interval)
            try:
                last_poll_error_upload = recover_operator_poll_failure(
                    args,
                    work_dir,
                    node_label,
                    operator_key,
                    exc,
                    upload_lock,
                    last_poll_error_upload,
                )
            except Exception:
                logging.exception("Operator Git recovery/poll-error upload failed.")
            time.sleep(poll_interval)
