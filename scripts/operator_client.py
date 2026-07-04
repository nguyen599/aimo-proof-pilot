#!/usr/bin/env python
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download


DEFAULT_REPO = "nguyen599/command"
DEFAULT_COMMAND_FILE = "command.sh"


class GitPushConflict(RuntimeError):
    pass


def repo_file(prefix: str, filename: str) -> str:
    prefix = prefix.strip("/")
    return f"{prefix}/{filename}" if prefix else filename


def normalize_node_label(node: str) -> str:
    value = node.strip()
    if value.startswith("output_node") and value.endswith(".txt"):
        return value[len("output_node") : -len(".txt")]
    if value.startswith("node") and value[4:]:
        return value[4:]
    return value


def normalize_command_id(command_id: str) -> str:
    value = command_id.strip().lower()
    if not value:
        return ""
    if len(value) < 6 or re.fullmatch(r"[0-9a-f]+", value) is None:
        raise SystemExit("--command-id must contain at least 6 hexadecimal SHA-256 characters.")
    return value[:6]


def output_path_for_node(prefix: str, node: str, command_id: str = "") -> str:
    label = normalize_node_label(node)
    suffix = f"_{normalize_command_id(command_id)}" if command_id else ""
    return repo_file(prefix, f"output_node{label}{suffix}.txt")


def output_repo(args: argparse.Namespace) -> str:
    return args.output_repo or args.repo


def github_token() -> str:
    token = os.environ.get("GITHUB_TOKEN", "").strip()
    if not token:
        raise SystemExit("GITHUB_TOKEN is required for --backend github.")
    return token


def client_prefers_github(args: argparse.Namespace) -> bool:
    return args.backend in {"github", "auto"}


def github_git_env() -> dict[str, str]:
    env = os.environ.copy()
    token = env.get("GITHUB_TOKEN", "").strip()
    env["GIT_TERMINAL_PROMPT"] = "0"
    env["GIT_LFS_SKIP_SMUDGE"] = "1"
    if token:
        env["GIT_CONFIG_COUNT"] = "1"
        env["GIT_CONFIG_KEY_0"] = f"url.https://{token}@github.com/.insteadOf"
        env["GIT_CONFIG_VALUE_0"] = "https://github.com/"
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


def reset_github_git_worktree(repo_dir: Path, branch: str) -> None:
    run_github_git(["reset", "--hard"], cwd=repo_dir)
    run_github_git(["clean", "-fd"], cwd=repo_dir)
    run_github_git(["checkout", "-B", branch, f"origin/{branch}"], cwd=repo_dir)
    run_github_git(["reset", "--hard", f"origin/{branch}"], cwd=repo_dir)


def client_github_git_dir(args: argparse.Namespace, repo: str) -> Path:
    repo_hash = hashlib.sha256(repo.encode("utf-8")).hexdigest()[:12]
    return Path(args.cache_dir).expanduser() / "github_git" / f"{repo_hash}_pid{os.getpid()}"


def ensure_client_github_git_repo(
    args: argparse.Namespace,
    repo: str,
    branch: str,
    sparse_paths: list[str] | None = None,
) -> Path:
    repo_dir = client_github_git_dir(args, repo)
    repo_url = f"https://github.com/{repo}.git"
    paths = [path.strip("/") for path in (sparse_paths or []) if path.strip("/")]
    if not (repo_dir / ".git").is_dir():
        import shutil

        repo_dir.parent.mkdir(parents=True, exist_ok=True)
        shutil.rmtree(repo_dir, ignore_errors=True)
        run_github_git(
            [
                "clone",
                "--depth",
                "1",
                "--filter=blob:none",
                "--no-checkout",
                "--branch",
                branch,
                repo_url,
                str(repo_dir),
            ]
        )
        run_github_git(["config", "user.email", "operator-client@local"], cwd=repo_dir)
        run_github_git(["config", "user.name", "operator-client"], cwd=repo_dir)
        if paths:
            run_github_git(["sparse-checkout", "init", "--no-cone"], cwd=repo_dir)
            run_github_git(["sparse-checkout", "set", *paths], cwd=repo_dir)
            run_github_git(["checkout", "-B", branch, f"origin/{branch}"], cwd=repo_dir)
        return repo_dir
    try:
        run_github_git(["remote", "set-url", "origin", repo_url], cwd=repo_dir)
        run_github_git(["reset", "--hard"], cwd=repo_dir)
        run_github_git(["clean", "-fd"], cwd=repo_dir)
        run_github_git(["fetch", "--depth", "1", "origin", branch], cwd=repo_dir)
        if paths:
            run_github_git(["sparse-checkout", "init", "--no-cone"], cwd=repo_dir)
            run_github_git(["sparse-checkout", "set", *paths], cwd=repo_dir)
            reset_github_git_worktree(repo_dir, branch)
    except Exception:
        import shutil

        shutil.rmtree(repo_dir, ignore_errors=True)
        run_github_git(
            [
                "clone",
                "--depth",
                "1",
                "--filter=blob:none",
                "--no-checkout",
                "--branch",
                branch,
                repo_url,
                str(repo_dir),
            ]
        )
        run_github_git(["config", "user.email", "operator-client@local"], cwd=repo_dir)
        run_github_git(["config", "user.name", "operator-client"], cwd=repo_dir)
        if paths:
            run_github_git(["sparse-checkout", "init", "--no-cone"], cwd=repo_dir)
            run_github_git(["sparse-checkout", "set", *paths], cwd=repo_dir)
            run_github_git(["checkout", "-B", branch, f"origin/{branch}"], cwd=repo_dir)
    return repo_dir


def github_git_download_text(args: argparse.Namespace, repo: str, path_in_repo: str, branch: str) -> str:
    repo_dir = ensure_client_github_git_repo(args, repo, branch, [path_in_repo])
    path = repo_dir / path_in_repo
    if not path.is_file():
        raise FileNotFoundError(f"{repo}/{path_in_repo} not found in git worktree")
    return path.read_text(encoding="utf-8", errors="replace")


def github_git_list_files(args: argparse.Namespace, repo: str, branch: str) -> list[str]:
    repo_dir = ensure_client_github_git_repo(args, repo, branch)
    output = run_github_git(["ls-tree", "-r", "--name-only", f"origin/{branch}"], cwd=repo_dir)
    return [line for line in output.splitlines() if line]


def github_git_upload_text(
    args: argparse.Namespace,
    repo: str,
    path_in_repo: str,
    text: str,
    branch: str,
    message: str,
    max_attempts: int = 20,
) -> None:
    last_error: BaseException | None = None
    for attempt in range(1, max_attempts + 1):
        repo_dir = ensure_client_github_git_repo(args, repo, branch, [path_in_repo])
        path = repo_dir / path_in_repo
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
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
            print(
                (
                    f"[operator_client] Git commit reported nothing to commit for "
                    f"{repo}/{path_in_repo}; attempting push anyway"
                ),
                file=sys.stderr,
                flush=True,
            )
        try:
            run_github_git(["push", "origin", branch], cwd=repo_dir)
            return
        except Exception as exc:
            last_error = exc
            if attempt >= max_attempts:
                raise GitPushConflict(str(exc)) from exc
            print(
                (
                    f"[operator_client] Git push conflict for {repo}/{path_in_repo}; "
                    f"retrying {attempt}/{max_attempts}"
                ),
                file=sys.stderr,
                flush=True,
            )
            time.sleep(min(5.0, 0.5 * attempt))
    raise GitPushConflict(f"Git upload failed after {max_attempts} attempts: {last_error}")


def command_text_from_args(args: argparse.Namespace) -> str:
    if getattr(args, "file", None):
        return Path(args.file).read_text(encoding="utf-8")
    if getattr(args, "command_text", None):
        return args.command_text
    command = list(getattr(args, "command", []) or [])
    if command and command[0] == "--":
        command = command[1:]
    if command:
        return shlex.join(command)
    if not sys.stdin.isatty():
        return sys.stdin.read()
    raise SystemExit("No command provided. Use --command, --file, stdin, or arguments after '--'.")


def upload_text(repo_id: str, path_in_repo: str, text: str, message: str) -> None:
    api = HfApi(token=os.environ.get("HF_TOKEN"))
    api.create_repo(repo_id=repo_id, repo_type="dataset", private=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False) as handle:
        handle.write(text.rstrip() + "\n")
        tmp_path = handle.name
    try:
        api.upload_file(
            repo_id=repo_id,
            repo_type="dataset",
            path_or_fileobj=tmp_path,
            path_in_repo=path_in_repo,
            commit_message=message,
        )
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def send_command(args: argparse.Namespace, text: str | None = None) -> str:
    command_text = text if text is not None else command_text_from_args(args)
    if not command_text.strip():
        raise SystemExit("Refusing to upload an empty command.")
    uploaded_text = command_text.rstrip() + "\n"
    command_id = hashlib.sha256(uploaded_text.encode("utf-8")).hexdigest()[:6]
    used_backend = args.backend
    if client_prefers_github(args):
        try:
            github_git_upload_text(
                args,
                args.repo,
                args.command_file,
                uploaded_text,
                args.github_branch,
                f"Update operator command {args.command_file}",
            )
            used_backend = "github-git"
        except Exception as git_exc:
            if args.backend != "auto":
                raise
            print(
                (
                    f"[operator_client] Git upload failed for {args.repo}/{args.command_file}; "
                    f"falling back to HF because --backend auto: {git_exc}"
                ),
                file=sys.stderr,
                flush=True,
            )
            upload_text(
                repo_id=args.repo,
                path_in_repo=args.command_file,
                text=uploaded_text,
                message=f"Update operator command {args.command_file}",
            )
            used_backend = "hf"
    else:
        upload_text(
            repo_id=args.repo,
            path_in_repo=args.command_file,
            text=uploaded_text,
            message=f"Update operator command {args.command_file}",
        )
    print(
        f"uploaded command to {used_backend} {args.repo}/{args.command_file} command_id={command_id}"
    )
    return command_id


def list_output_paths(args: argparse.Namespace) -> list[str]:
    repo_id = output_repo(args)
    prefix = args.prefix.strip("/")
    if client_prefers_github(args):
        try:
            files = github_git_list_files(args, repo_id, args.github_branch)
        except Exception as git_exc:
            if args.backend != "auto":
                raise
            print(
                f"[operator_client] Git list failed for {repo_id}; falling back to HF because --backend auto: {git_exc}",
                file=sys.stderr,
                flush=True,
            )
            files = HfApi(token=os.environ.get("HF_TOKEN")).list_repo_files(
                repo_id=repo_id,
                repo_type="dataset",
            )
    else:
        files = HfApi(token=os.environ.get("HF_TOKEN")).list_repo_files(
            repo_id=repo_id,
            repo_type="dataset",
        )
    command_id = normalize_command_id(getattr(args, "command_id", ""))
    paths: list[str] = []
    for path in files:
        if prefix and not path.startswith(prefix + "/"):
            continue
        name = Path(path).name
        if name.startswith("output_node") and name.endswith(".txt"):
            if command_id and not name.endswith(f"_{command_id}.txt"):
                continue
            paths.append(path)
    return sorted(paths)


def download_repo_file(args: argparse.Namespace, repo_id: str, path_in_repo: str) -> str:
    if client_prefers_github(args):
        try:
            return github_git_download_text(args, repo_id, path_in_repo, args.github_branch)
        except Exception as git_exc:
            if args.backend != "auto":
                raise
            print(
                (
                    f"[operator_client] Git download failed for {repo_id}/{path_in_repo}; "
                    f"falling back to HF because --backend auto: {git_exc}"
                ),
                file=sys.stderr,
                flush=True,
            )
    cache_dir = Path(args.cache_dir).expanduser()
    cache_dir.mkdir(parents=True, exist_ok=True)
    local_path = hf_hub_download(
        repo_id=repo_id,
        filename=path_in_repo,
        repo_type="dataset",
        cache_dir=str(cache_dir),
        force_download=True,
    )
    return Path(local_path).read_text(encoding="utf-8", errors="replace")


def operator_nodes_from_key_file(args: argparse.Namespace) -> list[str]:
    try:
        raw_text = download_repo_file(args, args.repo, args.key_file)
        payload = json.loads(raw_text)
    except Exception:
        return []
    operators = payload.get("operators") if isinstance(payload, dict) else None
    if not isinstance(operators, dict):
        return []
    nodes = [str(node) for node in operators if str(node).isdigit()]
    return sorted(nodes, key=lambda value: int(value))


def selected_output_paths(args: argparse.Namespace) -> list[str]:
    if args.node == "all" and normalize_command_id(getattr(args, "command_id", "")):
        nodes = operator_nodes_from_key_file(args)
        if nodes:
            return [
                output_path_for_node(args.prefix, node, args.command_id)
                for node in nodes
            ]
    if args.node == "all":
        return list_output_paths(args)
    return [output_path_for_node(args.prefix, args.node, args.command_id)]


def fetch_outputs(args: argparse.Namespace) -> dict[str, str]:
    repo_id = output_repo(args)
    paths = selected_output_paths(args)
    if not paths:
        raise SystemExit(f"No output_node*.txt files found in dataset {repo_id}.")
    outputs: dict[str, str] = {}
    out_dir = Path(args.out_dir).expanduser() if args.out_dir else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
    for path in paths:
        try:
            text = download_repo_file(args, repo_id, path)
        except FileNotFoundError as exc:
            text = f"[operator_client] missing output file {repo_id}/{path}: {exc}\n"
        outputs[path] = text
        if out_dir:
            local_name = path.replace("/", "__")
            (out_dir / local_name).write_text(text, encoding="utf-8")
    return outputs


def print_outputs(outputs: dict[str, str], full: bool = True, previous: dict[str, str] | None = None) -> None:
    previous = previous or {}
    for path, text in outputs.items():
        old = previous.get(path, "")
        if not full and text.startswith(old):
            delta = text[len(old) :]
        else:
            delta = text
        if not delta:
            continue
        if len(outputs) > 1:
            print(f"\n===== {path} =====", flush=True)
        print(delta, end="" if delta.endswith("\n") else "\n", flush=True)


def cmd_send(args: argparse.Namespace) -> None:
    send_command(args)


def cmd_stop(args: argparse.Namespace) -> None:
    send_command(args, "STOP")


def cmd_cancel(args: argparse.Namespace) -> None:
    send_command(args, "CANCEL")


def expected_operator_node_count(args: argparse.Namespace) -> int:
    try:
        raw_text = download_repo_file(args, args.repo, args.key_file)
        payload = json.loads(raw_text)
        operators = payload.get("operators") if isinstance(payload, dict) else None
        if isinstance(operators, dict):
            numeric_nodes = [node for node in operators if str(node).isdigit()]
            return len(numeric_nodes) if numeric_nodes else len(operators)
    except Exception:
        pass
    return 0


def wait_for_command_outputs(
    args: argparse.Namespace,
    command_id: str,
    wait_seconds: float,
    expected_nodes: int = 0,
) -> None:
    tail_args = argparse.Namespace(**vars(args))
    tail_args.command_id = command_id
    tail_args.full_first = True
    tail_args.quiet_errors = False
    previous: dict[str, str] = {}
    first = True
    deadline = time.monotonic() + max(0.0, wait_seconds)
    while True:
        try:
            outputs = fetch_outputs(tail_args)
            print_outputs(outputs, full=first, previous=previous)
            previous = outputs
            first = False
            active_outputs = [
                text for path, text in outputs.items() if "_inactive_" not in Path(path).name
            ]
            enough_nodes = not expected_nodes or len(active_outputs) >= expected_nodes
            if enough_nodes and active_outputs and all("finished_utc=" in text for text in active_outputs):
                return
        except (Exception, SystemExit) as exc:
            print(f"[operator_client] fetch failed: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        if time.monotonic() >= deadline:
            return
        time.sleep(max(1.0, float(getattr(args, "interval", 5.0))))


def legacy_kill_command_payload(
    target_command_id: str,
    grace_seconds: float,
    force: bool,
) -> dict[str, object]:
    code = f'''
import os
import signal
import subprocess
import time
from pathlib import Path

target = {target_command_id!r}.lower()
grace_seconds = {float(grace_seconds)!r}
force = {bool(force)!r}
own_pid = os.getpid()
parent_pid = os.getppid()
own_pgid = os.getpgrp()

def read_process_text(pid):
    proc = Path("/proc") / str(pid)
    try:
        cmd = (proc / "cmdline").read_bytes().replace(b"\\0", b" ").decode("utf-8", "replace")
    except (FileNotFoundError, PermissionError, ProcessLookupError, OSError):
        cmd = ""
    try:
        env = (proc / "environ").read_bytes().replace(b"\\0", b" ").decode("utf-8", "replace")
    except (FileNotFoundError, PermissionError, ProcessLookupError, OSError):
        env = ""
    return cmd, env

def scan():
    matches = {{}}
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit():
            continue
        pid = int(proc.name)
        if pid in {{own_pid, parent_pid}}:
            continue
        try:
            pgid = os.getpgid(pid)
        except (ProcessLookupError, PermissionError, OSError):
            continue
        if pgid == own_pgid:
            continue
        cmd, env = read_process_text(pid)
        lowered_cmd = cmd.lower()
        lowered_env = env.lower()
        if "train_operator.py" in lowered_cmd and "--operator_mode" in lowered_cmd:
            continue
        if target not in lowered_cmd and target not in lowered_env:
            continue
        matches.setdefault(pgid, (pid, cmd, env))
    return matches

print(f"kill_command compatibility mode target_command_id={{target}} host={{os.uname().nodename}}")
matches = scan()
print(f"process_groups_before={{len(matches)}}")
for pgid, (pid, cmd, env) in sorted(matches.items()):
    marker = next((part for part in env.split() if part.startswith("OLMO_RUN_DIR_NAME=")), "")
    print(f"TARGET pgid={{pgid}} pid={{pid}} marker={{marker}} cmd={{cmd[:500]}}")
    try:
        os.killpg(pgid, signal.SIGTERM)
        print(f"sent SIGTERM pgid={{pgid}}")
    except ProcessLookupError:
        pass
    except Exception as exc:
        print(f"SIGTERM failed pgid={{pgid}}: {{type(exc).__name__}}: {{exc}}")

if grace_seconds > 0:
    time.sleep(grace_seconds)
remaining = scan()
print(f"process_groups_after_sigterm={{len(remaining)}}")
if force:
    for pgid, (pid, cmd, env) in sorted(remaining.items()):
        try:
            os.killpg(pgid, signal.SIGKILL)
            print(f"sent SIGKILL pgid={{pgid}}")
        except ProcessLookupError:
            pass
        except Exception as exc:
            print(f"SIGKILL failed pgid={{pgid}}: {{type(exc).__name__}}: {{exc}}")
    time.sleep(1)
remaining = scan()
print(f"process_groups_final={{len(remaining)}}")
subprocess.run(
    [
        "nvidia-smi",
        "--query-gpu=index,utilization.gpu,memory.used,memory.total",
        "--format=csv,noheader,nounits",
    ],
    check=False,
)
raise SystemExit(0 if not remaining else 1)
'''
    encoded = base64.b64encode(code.encode("utf-8")).decode("ascii")
    return {
        "action": "run_command",
        "command": [
            "python",
            "-c",
            f"import base64; exec(base64.b64decode({encoded!r}))",
        ],
        "nonce": time.time_ns(),
    }


def cmd_kill(args: argparse.Namespace) -> None:
    target_command_id = normalize_command_id(args.target_command_id)
    if args.legacy_scan:
        payload = legacy_kill_command_payload(
            target_command_id,
            args.grace_seconds,
            not args.no_force,
        )
    else:
        payload = {
            "action": "kill_command",
            "command_id": target_command_id,
            "grace_seconds": args.grace_seconds,
            "force": not args.no_force,
            "nonce": time.time_ns(),
        }
    expected_nodes = expected_operator_node_count(args) if args.node == "all" else 1
    kill_command_id = send_command(args, json.dumps(payload, sort_keys=True))
    print(
        f"target_command_id={target_command_id} kill_command_id={kill_command_id}",
        flush=True,
    )
    if args.no_tail:
        return
    wait_for_command_outputs(args, kill_command_id, args.wait_seconds, expected_nodes=expected_nodes)


def cmd_restart(args: argparse.Namespace) -> None:
    send_command(args, json.dumps({"action": "restart_operator", "nonce": time.time_ns()}, sort_keys=True))


def cmd_list(args: argparse.Namespace) -> None:
    for path in list_output_paths(args):
        print(path)


def cmd_fetch(args: argparse.Namespace) -> None:
    outputs = fetch_outputs(args)
    if args.print_output:
        print_outputs(outputs)


def cmd_tail(args: argparse.Namespace) -> None:
    previous: dict[str, str] = {}
    first = True
    while True:
        try:
            outputs = fetch_outputs(args)
            print_outputs(outputs, full=first and args.full_first, previous=previous)
            previous = outputs
            first = False
        except Exception as exc:
            if not args.quiet_errors:
                print(f"[operator_client] fetch failed: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        if args.once:
            return
        time.sleep(max(1.0, args.interval))


def cmd_run(args: argparse.Namespace) -> None:
    command_id = send_command(args)
    if args.no_tail:
        return
    tail_args = argparse.Namespace(**vars(args))
    tail_args.command_id = command_id
    tail_args.full_first = False
    tail_args.once = False
    tail_args.quiet_errors = False
    cmd_tail(tail_args)


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--backend",
        default="github",
        choices=("hf", "github", "auto"),
        help="Command/output backend. 'github' uses git pull/push only; 'auto' uses git and falls back to HF.",
    )
    parser.add_argument("--repo", default=DEFAULT_REPO, help="Repo containing the operator command file.")
    parser.add_argument("--command-file", default=DEFAULT_COMMAND_FILE, help="Path to command file inside --repo.")
    parser.add_argument("--key-file", default="key.txt", help="Path to operator key file inside --repo.")
    parser.add_argument("--output-repo", default="", help="Repo containing output files. Defaults to --repo.")
    parser.add_argument("--prefix", default="", help="Optional output prefix used by --operator_output_prefix.")
    parser.add_argument("--cache-dir", default="~/.cache/operator-client", help="Local HF/Git cache.")
    parser.add_argument("--github-branch", default="main", help="GitHub branch for --backend github.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Send operator commands and fetch live output_node<N>.txt logs.",
        epilog=(
            "Examples:\n"
            "  operator_client.py send -- python train.py --backend olmo_core_sft ...\n"
            "  operator_client.py send --file command.sh\n"
            "  operator_client.py tail --node all --interval 10\n"
            "  operator_client.py run --node 0 -- bash -lc 'cd /tmp && nvidia-smi'\n"
            "  operator_client.py kill 157a77 --node all\n"
            "  operator_client.py cancel\n"
            "  operator_client.py restart\n"
            "  operator_client.py stop\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    add_common_args(parser)
    subparsers = parser.add_subparsers(dest="command_name", required=True)

    send = subparsers.add_parser("send", help="Upload command text to the configured command file.")
    send.add_argument("-c", "--command", dest="command_text", default="", help="Command text to upload.")
    send.add_argument("--file", default="", help="Read command text from a local file.")
    send.add_argument("command", nargs=argparse.REMAINDER, help="Command tokens after '--'.")
    send.set_defaults(func=cmd_send)

    stop = subparsers.add_parser(
        "stop", help="Cancel active commands and exit operator mode."
    )
    stop.set_defaults(func=cmd_stop)

    cancel = subparsers.add_parser("cancel", help="Cancel active commands but keep operator mode running.")
    cancel.set_defaults(func=cmd_cancel)

    kill_cmd = subparsers.add_parser(
        "kill",
        help="Kill a specific operator command by the command_id printed by send/run.",
    )
    kill_cmd.add_argument("target_command_id", help="Command ID to kill, e.g. 157a77.")
    kill_cmd.add_argument("--node", default="all", help="Node label to fetch after upload, e.g. 0, 1, none, or all.")
    kill_cmd.add_argument("--interval", type=float, default=5.0, help="Output polling interval in seconds.")
    kill_cmd.add_argument("--wait-seconds", type=float, default=60.0, help="How long to wait for kill outputs.")
    kill_cmd.add_argument("--grace-seconds", type=float, default=10.0, help="Seconds before SIGKILL fallback.")
    kill_cmd.add_argument("--no-force", action="store_true", help="Only send SIGTERM; do not follow with SIGKILL.")
    kill_cmd.add_argument(
        "--legacy-scan",
        action="store_true",
        help="Use the old standalone /proc scanner instead of native operator kill_command.",
    )
    kill_cmd.add_argument("--no-tail", action="store_true", help="Only upload the kill command.")
    kill_cmd.add_argument("--out-dir", default="", help="Optional local directory to save fetched kill logs.")
    kill_cmd.set_defaults(func=cmd_kill)

    restart = subparsers.add_parser(
        "restart",
        help="Start a fresh train.py --operator_mode process; older operators remain alive but passive.",
    )
    restart.set_defaults(func=cmd_restart)

    list_cmd = subparsers.add_parser("list", help="List output_node*.txt files.")
    list_cmd.add_argument("--command-id", default="", help="Filter by the first 6 SHA-256 characters.")
    list_cmd.set_defaults(func=cmd_list)

    fetch = subparsers.add_parser("fetch", help="Fetch output logs once.")
    fetch.add_argument("--node", default="all", help="Node label, e.g. 0, 1, 2, none, or all.")
    fetch.add_argument("--command-id", default="", help="Exact command ID printed by send/run.")
    fetch.add_argument("--out-dir", default="", help="Optional local directory to save fetched logs.")
    fetch.add_argument("--print", dest="print_output", action="store_true", default=True, help="Print logs to stdout.")
    fetch.add_argument("--no-print", dest="print_output", action="store_false", help="Do not print logs.")
    fetch.set_defaults(func=cmd_fetch)

    tail = subparsers.add_parser("tail", help="Poll output logs and print new content.")
    tail.add_argument("--node", default="all", help="Node label, e.g. 0, 1, 2, none, or all.")
    tail.add_argument("--command-id", default="", help="Exact command ID printed by send/run.")
    tail.add_argument("--interval", type=float, default=10.0, help="Polling interval in seconds.")
    tail.add_argument("--out-dir", default="", help="Optional local directory to save fetched logs each poll.")
    tail.add_argument("--full-first", action="store_true", default=True, help="Print the full log on first fetch.")
    tail.add_argument("--delta-first", dest="full_first", action="store_false", help="Only print newly added text.")
    tail.add_argument("--once", action="store_true", help="Fetch once and exit.")
    tail.add_argument("--quiet-errors", action="store_true", help="Suppress transient fetch errors.")
    tail.set_defaults(func=cmd_tail)

    run = subparsers.add_parser("run", help="Upload a command, then tail output logs.")
    run.add_argument("--node", default="all", help="Node label to tail after upload.")
    run.add_argument("--command-id", default="", help=argparse.SUPPRESS)
    run.add_argument("--interval", type=float, default=10.0, help="Tail polling interval in seconds.")
    run.add_argument("--out-dir", default="", help="Optional local directory to save fetched logs.")
    run.add_argument("--no-tail", action="store_true", help="Only upload the command.")
    run.add_argument("-c", "--command", dest="command_text", default="", help="Command text to upload.")
    run.add_argument("--file", default="", help="Read command text from a local file.")
    run.add_argument("command", nargs=argparse.REMAINDER, help="Command tokens after '--'.")
    run.set_defaults(func=cmd_run)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
