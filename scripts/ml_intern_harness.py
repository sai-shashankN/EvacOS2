"""Codex-controlled harness for Hugging Face ml-intern-style ML review.

This wrapper lets either ml-intern or Codex CLI act as a top-level ML
specialist lane while Codex keeps the operational safety boundary. The local
ml-intern checkout can use the patched ``openai-codex/...`` model ids to route
through Codex OAuth. The wrapper writes a guarded task file, a YOLO approval
request template, and stdout/stderr logs. It runs from an isolated scratch
directory and avoids passing HF/GitHub credentials unless explicitly allowed.
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import shutil
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOG_ROOT = REPO_ROOT / "logs" / "ml_intern"
DEFAULT_ML_INTERN_DIR = REPO_ROOT / ".tmp_external_review" / "ml-intern"
DEFAULT_CODEX_MODEL = "gpt-5.5"
MAX_CONTEXT_CHARS = 50_000

BLOCKED_CONTEXT_NAMES = {
    ".env",
    ".env.local",
    ".env.production",
    ".env.development",
}
BLOCKED_CONTEXT_PARTS = {
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    "outputs",
    "wandb",
    "runs",
    "checkpoints",
}
STRIPPED_HF_ENV = {
    "HF_TOKEN",
    "HUGGINGFACE_HUB_TOKEN",
}
STRIPPED_CODEX_API_ENV = {
    "OPENAI_API_KEY",
    "OPENAI_BASE_URL",
    "OPENAI_ORG_ID",
    "OPENAI_ORGANIZATION",
    "OPENAI_PROJECT",
}


TOP_LEVEL_GUARDRAIL = """\
You are an ML specialist lane running under a Codex harness for the EvacOS2 repository.

You are allowed to operate as a top-level ML/Hugging Face specialist lane, using GPT-5.5 or another explicitly selected model. Treat the task like a serious ML engineering assignment: form hypotheses, inspect attached context, search public HF/docs/examples when your tools allow it, and produce actionable decisions.

Codex is still the operational harness and final committer.

Hard boundaries unless the Codex task text explicitly says otherwise:
- Do not start, rent, destroy, or modify paid compute, including HF Jobs, Spaces hardware, Vast, cloud GPUs, or sandboxes.
- Do not upload, delete, create, merge, or mutate Hugging Face or GitHub repositories.
- Do not edit, write, delete, move, or overwrite files in the EvacOS2 repository.
- Do not read or print secrets, tokens, .env files, credentials, SSH keys, or cached auth files.
- You may use temporary files and commands inside the scratch workspace for analysis.
- If a code change is needed, return a proposed patch or precise recommendation; Codex will apply and verify it.
- If a command would need paid compute, credentials, or repo mutation, list it under "Needs Codex approval" instead of running it.

YOLO-class approval protocol:
- If you believe a restricted action is important, write a request into ./approval_requests.md and summarize it under "Needs Codex approval".
- Each request must include: action, exact command/API call if known, target resource, expected cost/risk, rollback/stop plan, required credentials, and why it is worth doing now.
- Do not execute the restricted action yourself. Codex will inspect the request, ask the user if needed, and either execute it separately or re-run you with an explicit approval scope.

Required final shape:
1. Findings
2. Evidence / sources checked
3. Recommended Codex actions
4. Needs Codex approval
5. Residual risks
"""

READ_ONLY_GUARDRAIL = """\
You are an ML specialist lane running under a Codex harness for the EvacOS2 repository.

Codex is the supervisor. Your job is bounded ML/Hugging Face research or review, not autonomous repo control.

Hard boundaries unless the Codex task text explicitly says otherwise:
- Do not start, rent, destroy, or modify paid compute, including HF Jobs, Spaces hardware, Vast, cloud GPUs, or sandboxes.
- Do not upload, delete, create, merge, or mutate Hugging Face or GitHub repositories.
- Do not edit, write, delete, move, or overwrite files in the EvacOS2 repository.
- Do not read or print secrets, tokens, .env files, credentials, SSH keys, or cached auth files.
- Prefer read-only public HF docs, papers, dataset/model metadata, and code examples.
- If a code change is needed, return a proposed patch or precise recommendation; Codex will apply it.
- If a command would need paid compute, credentials, or repo mutation, list it under "Needs Codex approval" instead of running it.

YOLO-class approval protocol:
- If a restricted action looks necessary, write a request into ./approval_requests.md and summarize it under "Needs Codex approval".
- Include: action, exact command/API call if known, target resource, expected cost/risk, rollback/stop plan, required credentials, and why it is worth doing now.
- Do not execute the restricted action yourself.

Required final shape:
1. Findings
2. Evidence / sources checked
3. Recommended Codex actions
4. Needs Codex approval
5. Residual risks
"""

APPROVAL_REQUEST_TEMPLATE = """# ml-intern YOLO-Class Approval Requests

Use this file only for actions that need Codex/user approval before execution.

For each request, include:

- Action:
- Exact command/API call:
- Target resource:
- Expected cost/risk:
- Required credentials:
- Rollback/stop plan:
- Why now:

Codex will inspect this file after the run and decide whether to execute, reject,
or ask the user.
"""


def _timestamp() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _safe_repo_path(raw_path: str) -> Path:
    path = (REPO_ROOT / raw_path).resolve()
    try:
        path.relative_to(REPO_ROOT)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"context path must stay inside repo: {raw_path}"
        ) from exc
    if path.name in BLOCKED_CONTEXT_NAMES:
        raise argparse.ArgumentTypeError(f"refusing secret context file: {raw_path}")
    if any(part in BLOCKED_CONTEXT_PARTS for part in path.parts):
        raise argparse.ArgumentTypeError(f"refusing generated/secret context path: {raw_path}")
    if not path.exists():
        raise argparse.ArgumentTypeError(f"context path does not exist: {raw_path}")
    if path.is_dir():
        raise argparse.ArgumentTypeError(f"context path must be a file, not directory: {raw_path}")
    return path


def _read_context(path: Path) -> str:
    text = path.read_text(encoding="utf-8", errors="replace")
    if len(text) > MAX_CONTEXT_CHARS:
        text = text[:MAX_CONTEXT_CHARS]
        text += f"\n\n[truncated to {MAX_CONTEXT_CHARS} chars by Codex harness]\n"
    rel = path.relative_to(REPO_ROOT).as_posix()
    return f"## Context File: {rel}\n\n```text\n{text}\n```"


def _runner_identity(runner: str) -> str:
    if runner == "codex":
        return (
            "## Runner Identity\n\n"
            "This task is launched through `codex exec`, using the current "
            "Codex/ChatGPT authenticated session unless Codex is explicitly "
            "configured otherwise. Act as the ml-intern-style ML specialist "
            "lane, but keep the Codex safety boundaries below."
        )
    return (
        "## Runner Identity\n\n"
        "This task is launched through Hugging Face `ml-intern`. Codex is the "
        "operational harness and final committer."
    )


def _build_task(user_prompt: str, context_files: list[Path], profile: str, runner: str) -> str:
    context_block = "\n\n".join(_read_context(path) for path in context_files)
    if not context_block:
        context_block = "No repo context files were attached. Ask for specific files if needed."
    guardrail = TOP_LEVEL_GUARDRAIL if profile == "top-level" else READ_ONLY_GUARDRAIL
    return "\n\n".join(
        [
            guardrail,
            _runner_identity(runner),
            "## Codex Task",
            user_prompt.strip(),
            "## Attached Repo Context",
            context_block,
        ]
    ).strip() + "\n"


def _resolve_ml_intern_command(args: argparse.Namespace) -> list[str]:
    if args.ml_intern_cmd:
        return [args.ml_intern_cmd]

    installed = shutil.which("ml-intern")
    if installed:
        return [installed]

    uv = shutil.which("uv")
    if not uv:
        raise RuntimeError(
            "ml-intern is not installed and uv is not on PATH. "
            "Install ml-intern or pass --ml-intern-cmd."
        )

    ml_intern_dir = Path(args.ml_intern_dir or os.environ.get("ML_INTERN_DIR") or DEFAULT_ML_INTERN_DIR).resolve()
    if not (ml_intern_dir / "pyproject.toml").exists():
        raise RuntimeError(
            f"ml-intern checkout not found at {ml_intern_dir}. "
            "Set ML_INTERN_DIR or pass --ml-intern-dir."
        )

    return [uv, "run", "--project", str(ml_intern_dir), "ml-intern"]


def _resolve_codex_command(args: argparse.Namespace, workspace_dir: Path) -> list[str]:
    codex = args.codex_cmd or shutil.which("codex")
    if not codex:
        raise RuntimeError(
            "codex is not on PATH. Install/authenticate Codex CLI or pass --codex-cmd."
        )

    model = args.model or os.environ.get("CODEX_MODEL") or DEFAULT_CODEX_MODEL
    return [
        codex,
        "exec",
        "--full-auto",
        "-C",
        str(workspace_dir),
        "--skip-git-repo-check",
        "-m",
        model,
        "-c",
        f'model_reasoning_effort="{args.reasoning_effort}"',
        "Read ./task.md and complete the Codex-harnessed ml-intern-style task exactly.",
    ]


def _build_runner_command(args: argparse.Namespace, workspace_dir: Path) -> list[str]:
    if args.runner == "codex":
        return _resolve_codex_command(args, workspace_dir)

    command = _resolve_ml_intern_command(args)
    command.extend(["--max-iterations", str(args.max_iterations), "--no-stream"])
    model = args.model or os.environ.get("ML_INTERN_MODEL")
    if model:
        command.extend(["--model", model])
    command.append("Read ./task.md and complete the Codex-harnessed task exactly.")
    return command


def _build_env(args: argparse.Namespace) -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault("PYTHONIOENCODING", "utf-8")
    env.setdefault("PYTHONUTF8", "1")
    if not args.allow_hf_token:
        for key in STRIPPED_HF_ENV:
            env.pop(key, None)
    if not args.allow_github_token:
        env.pop("GITHUB_TOKEN", None)
    if args.runner == "codex" and not args.allow_api_key_env:
        for key in STRIPPED_CODEX_API_ENV:
            env.pop(key, None)
    return env


def _write_run_files(args: argparse.Namespace) -> tuple[Path, Path, str]:
    run_id = args.run_id or _timestamp()
    run_dir = (Path(args.log_root) if args.log_root else DEFAULT_LOG_ROOT) / run_id
    workspace_dir = run_dir / "workspace"
    workspace_dir.mkdir(parents=True, exist_ok=True)

    user_prompt = args.prompt
    if args.prompt_file:
        user_prompt = Path(args.prompt_file).read_text(encoding="utf-8")
    if not user_prompt or not user_prompt.strip():
        raise RuntimeError("provide a prompt argument or --prompt-file")

    task = _build_task(user_prompt, args.context, args.profile, args.runner)
    task_path = workspace_dir / "task.md"
    task_path.write_text(task, encoding="utf-8")
    (workspace_dir / "approval_requests.md").write_text(
        APPROVAL_REQUEST_TEMPLATE, encoding="utf-8"
    )
    return run_dir, workspace_dir, task


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run ml-intern or Codex through a Codex safety harness.")
    parser.add_argument("prompt", nargs="?", help="Task for the Codex-harnessed ml-intern lane.")
    parser.add_argument("--prompt-file", help="Read task text from a file.")
    parser.add_argument("--context", action="append", type=_safe_repo_path, default=[], help="Repo file to embed as read-only context.")
    parser.add_argument("--profile", choices=["top-level", "read-only"], default="top-level", help="Guardrail profile for the ml-intern lane.")
    parser.add_argument("--runner", choices=["ml-intern", "codex"], default="ml-intern", help="Execution backend. Use codex for ChatGPT/Codex subscription-backed GPT-5.5.")
    parser.add_argument("--model", help="Model id. For --runner ml-intern, falls back to ML_INTERN_MODEL. For --runner codex, defaults to gpt-5.5.")
    parser.add_argument("--max-iterations", type=int, default=8, help="Headless ml-intern iteration cap.")
    parser.add_argument("--ml-intern-dir", help="Path to an external ml-intern checkout.")
    parser.add_argument("--ml-intern-cmd", help="Path/name of an installed ml-intern executable.")
    parser.add_argument("--codex-cmd", help="Path/name of an installed Codex CLI executable.")
    parser.add_argument("--reasoning-effort", default="high", choices=["low", "medium", "high", "xhigh"], help="Codex runner reasoning effort.")
    parser.add_argument("--log-root", help="Directory for harness run logs.")
    parser.add_argument("--run-id", help="Stable run id for reproducible logs.")
    parser.add_argument("--allow-hf-token", action="store_true", help="Pass HF_TOKEN/HUGGINGFACE_HUB_TOKEN through to ml-intern.")
    parser.add_argument("--allow-github-token", action="store_true", help="Pass GITHUB_TOKEN through to the runner.")
    parser.add_argument("--allow-api-key-env", action="store_true", help="For --runner codex only, allow OPENAI_* API env vars instead of forcing ChatGPT-session auth.")
    parser.add_argument("--execute", action="store_true", help="Actually run the selected runner. Omit for dry-run.")
    args = parser.parse_args(argv)

    try:
        run_dir, workspace_dir, _task = _write_run_files(args)
        command = _build_runner_command(args, workspace_dir)

        command_path = run_dir / "command.txt"
        stdout_path = run_dir / "stdout.log"
        stderr_path = run_dir / "stderr.log"
        command_path.write_text(subprocess.list2cmdline(command) + "\n", encoding="utf-8")

        print(f"Run dir: {run_dir}")
        print(f"Task file: {workspace_dir / 'task.md'}")
        print(f"Command file: {command_path}")

        if not args.execute:
            print(f"Dry-run only. Re-run with --execute to launch {args.runner}.")
            return 0

        with stdout_path.open("w", encoding="utf-8", errors="replace") as stdout, stderr_path.open(
            "w", encoding="utf-8", errors="replace"
        ) as stderr:
            result = subprocess.run(
                command,
                cwd=workspace_dir,
                env=_build_env(args),
                text=True,
                stdout=stdout,
                stderr=stderr,
                check=False,
            )

        print(f"stdout: {stdout_path}")
        print(f"stderr: {stderr_path}")
        return result.returncode
    except Exception as exc:
        print(f"ml-intern harness error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
