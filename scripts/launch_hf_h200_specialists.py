"""Launch fire/flood/gas specialist quality runs on HF GPU Jobs.

The launcher reads tokens from `.env` by default:

- `HFALT1_TOKEN` -> fire
- `HFALT2_TOKEN` -> flood
- `HFALT3_TOKEN` -> gas

Tokens are passed to Jobs as encrypted secrets and are not printed.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
import time
from pathlib import Path

from huggingface_hub import HfApi

DEFAULT_ASSIGNMENTS = {
    "fire": "HFALT1_TOKEN",
    "flood": "HFALT2_TOKEN",
    "gas": "HFALT3_TOKEN",
}
DEFAULT_NAMESPACES = {
    "fire": "hfnasjdjas",
    "flood": "skjdfndajksndkjs",
    "gas": "werfasfs",
}


def _load_env(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for raw_line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def _job_command(repo_ref: str) -> list[str]:
    del repo_ref
    return [
        "bash",
        "-lc",
        (
            "python -m pip install -q 'huggingface_hub>=1.7.2' && "
            "mkdir -p /workspace/source /workspace/EvacOS2_boot && "
            "python -c \"import os; from huggingface_hub import hf_hub_download; "
            "p=hf_hub_download(repo_id=os.environ['HF_SOURCE_REPO'], "
            "filename=os.environ['HF_SOURCE_FILENAME'], repo_type='model', "
            "token=os.environ['HF_TOKEN'], local_dir='/workspace/source'); print(p)\" && "
            "tar -xzf \"/workspace/source/$HF_SOURCE_FILENAME\" -C /workspace/EvacOS2_boot && "
            "cd /workspace/EvacOS2_boot && "
            "EVACOS_USE_EXISTING_SOURCE=1 bash scripts/hf_h200_specialist_job.sh"
        ),
    ]


def _git_sha() -> str:
    return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()


def _create_source_archive(target: Path) -> None:
    subprocess.run(
        ["git", "archive", "--format=tar.gz", "-o", str(target), "HEAD"],
        check=True,
    )


def _inspect_job_stage(api: HfApi, job_id: str, namespace: str, token: str) -> str:
    try:
        job = api.inspect_job(job_id=job_id, namespace=namespace, token=token)
    except Exception:
        return "UNKNOWN"
    status = getattr(job, "status", None)
    return str(getattr(status, "stage", status))


def _job_log_tail(job_id: str, namespace: str, token: str, tail: int = 160) -> str:
    env = {**os.environ, "HF_TOKEN": token, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}
    try:
        completed = subprocess.run(
            ["hf", "jobs", "logs", job_id, "--namespace", namespace, "--tail", str(tail)],
            check=False,
            capture_output=True,
            encoding="utf-8",
            errors="replace",
            env=env,
            text=True,
            timeout=60,
        )
    except Exception as exc:
        return f"LOG_FETCH_ERROR {type(exc).__name__}: {exc}"
    return (completed.stdout or "") + ("\n" + completed.stderr if completed.stderr else "")


def _is_cuda_802_failure(log_text: str) -> bool:
    return "Error 802" in log_text or "system not yet initialized" in log_text


def _has_training_progress(log_text: str) -> bool:
    return any(marker in log_text for marker in ("TRAIN_START", "TRAIN_PROGRESS", "ARTIFACT_REPO="))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--repo-url", default="https://github.com/sai-shashankN/EvacOS2.git")
    parser.add_argument("--repo-ref", default="main")
    parser.add_argument("--image", default="pytorch/pytorch:2.7.1-cuda12.6-cudnn9-devel")
    parser.add_argument("--flavor", default="h200")
    parser.add_argument("--timeout", default="4h")
    parser.add_argument("--steps", type=int, help="Override specialist steps for small paid canaries.")
    parser.add_argument("--run-label", default="", help="Optional label used in run/output names.")
    parser.add_argument(
        "--retry-cuda802-attempts",
        type=int,
        default=1,
        help="Relaunch on H100/H200 host CUDA error 802 up to this many total attempts.",
    )
    parser.add_argument(
        "--retry-watch-seconds",
        type=int,
        default=360,
        help="Seconds to watch each attempt for TRAIN_START/progress or CUDA 802 failure.",
    )
    parser.add_argument("--only", choices=["fire", "flood", "gas"], action="append")
    parser.add_argument("--source-tgz", type=Path)
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=10,
        help="Training checkpoint cadence passed to the HF job. Defaults to 10.",
    )
    parser.add_argument(
        "--eval-every",
        type=int,
        default=10,
        help="Eval cadence passed to the HF job. Defaults to 10.",
    )
    parser.add_argument(
        "--upload-every",
        type=int,
        default=10,
        help="Periodic HF artifact upload cadence. Defaults to 10.",
    )
    parser.add_argument(
        "--resume-checkpoint-repo",
        default="",
        help="Optional model repo containing a checkpoint tree to seed/resume from.",
    )
    parser.add_argument(
        "--resume-checkpoint-path-template",
        default="",
        help=(
            "Optional checkpoint path template inside --resume-checkpoint-repo. "
            "Use {family}; e.g. floor-specialists/{family}/vast-canary50/checkpoints"
        ),
    )
    parser.add_argument(
        "--resume-from-public-vast50",
        action="store_true",
        help=(
            "Shortcut for --resume-checkpoint-repo shashankN777/evacos2-7b-orchestrator-artifacts "
            "and floor-specialists/{family}/vast-canary50/checkpoints."
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    env_values = {**os.environ, **_load_env(args.env_file)}
    families = args.only or ["fire", "flood", "gas"]
    launched: list[dict[str, str]] = []
    sha = _git_sha()

    temp_dir_obj: tempfile.TemporaryDirectory[str] | None = None
    if args.source_tgz is None:
        temp_dir_obj = tempfile.TemporaryDirectory()
        source_tgz = Path(temp_dir_obj.name) / f"evacos2_source_{sha}.tgz"
        _create_source_archive(source_tgz)
    else:
        source_tgz = args.source_tgz
    source_filename = f"source/{source_tgz.name}"

    try:
        for family in families:
            token_key = DEFAULT_ASSIGNMENTS[family]
            token = env_values.get(token_key, "").strip()
            if not token:
                raise SystemExit(f"Missing {token_key} in environment or {args.env_file}")

            api = HfApi(token=token)
            namespace = (
                env_values.get(f"{token_key}_NAMESPACE", "").strip()
                or env_values.get(f"HFALT{['fire', 'flood', 'gas'].index(family) + 1}_NAMESPACE", "").strip()
                or DEFAULT_NAMESPACES[family]
            )
            artifact_repo = f"{namespace}/evacos2-h200-specialist-artifacts"
            labels = {
                "project": "evacos2",
                "run": f"{family}-quality",
                "family": family,
            }
            job_env = {
                "DISASTER_FAMILY": family,
                "EVACOS_REPO_URL": args.repo_url,
                "EVACOS_REPO_REF": args.repo_ref,
                "HF_ARTIFACT_REPO": artifact_repo,
                "HF_SOURCE_REPO": artifact_repo,
                "HF_SOURCE_FILENAME": source_filename,
                "HF_SPECIALIST_CHECKPOINT_EVERY": str(args.checkpoint_every),
                "HF_SPECIALIST_EVAL_EVERY": str(args.eval_every),
                "HF_SPECIALIST_UPLOAD_EVERY": str(args.upload_every),
            }
            resume_repo = args.resume_checkpoint_repo
            resume_path_template = args.resume_checkpoint_path_template
            if args.resume_from_public_vast50:
                resume_repo = "shashankN777/evacos2-7b-orchestrator-artifacts"
                resume_path_template = "floor-specialists/{family}/vast-canary50/checkpoints"
            if resume_repo and resume_path_template:
                job_env["HF_RESUME_CHECKPOINT_REPO"] = resume_repo
                job_env["HF_RESUME_CHECKPOINT_PATH"] = resume_path_template.format(family=family)
            if args.steps is not None:
                job_env["HF_SPECIALIST_STEPS"] = str(args.steps)
            if args.run_label:
                job_env["HF_SPECIALIST_RUN_LABEL"] = args.run_label
            print(
                json.dumps(
                    {
                        "family": family,
                        "token_key": token_key,
                        "namespace": namespace,
                        "flavor": args.flavor,
                        "timeout": args.timeout,
                        "steps": args.steps,
                        "run_label": args.run_label,
                        "repo_ref": args.repo_ref,
                        "source": f"{artifact_repo}/{source_filename}",
                        "artifact_repo": artifact_repo,
                        "dry_run": args.dry_run,
                    },
                    indent=2,
                )
            )
            if args.dry_run:
                continue

            api.create_repo(
                repo_id=artifact_repo,
                repo_type="model",
                private=True,
                exist_ok=True,
                token=token,
            )
            api.upload_file(
                repo_id=artifact_repo,
                repo_type="model",
                path_or_fileobj=str(source_tgz),
                path_in_repo=source_filename,
                commit_message=f"Upload EvacOS2 source {sha}",
                token=token,
            )

            total_attempts = max(1, args.retry_cuda802_attempts)
            for attempt in range(1, total_attempts + 1):
                attempt_env = {**job_env, "HF_SPECIALIST_ATTEMPT": str(attempt)}
                job = api.run_job(
                    image=args.image,
                    command=_job_command(args.repo_ref),
                    env=attempt_env,
                    secrets={"HF_TOKEN": token},
                    flavor=args.flavor,  # type: ignore[arg-type]
                    timeout=args.timeout,
                    labels={**labels, "attempt": str(attempt)},
                    namespace=namespace,
                    token=token,
                )
                record = {
                    "family": family,
                    "job_id": job.id,
                    "namespace": namespace,
                    "artifact_repo": artifact_repo,
                    "attempt": str(attempt),
                }
                launched.append(record)
                print(json.dumps(record, indent=2))
                if total_attempts == 1:
                    break

                deadline = time.time() + args.retry_watch_seconds
                retry_needed = False
                while time.time() < deadline:
                    time.sleep(30)
                    log_tail = _job_log_tail(job.id, namespace, token)
                    stage = _inspect_job_stage(api, job.id, namespace, token)
                    print(
                        json.dumps(
                            {
                                "family": family,
                                "job_id": job.id,
                                "attempt": attempt,
                                "stage": stage,
                                "has_progress": _has_training_progress(log_tail),
                                "cuda802": _is_cuda_802_failure(log_tail),
                            }
                        )
                    )
                    if _has_training_progress(log_tail) or stage in {"COMPLETED", "RUNNING"} and "TRAIN_START" in log_tail:
                        retry_needed = False
                        break
                    if stage in {"ERROR", "CANCELED"}:
                        retry_needed = _is_cuda_802_failure(log_tail)
                        break
                else:
                    log_tail = _job_log_tail(job.id, namespace, token)
                    retry_needed = _is_cuda_802_failure(log_tail)

                if not retry_needed:
                    break
                if attempt < total_attempts:
                    print(
                        json.dumps(
                            {
                                "family": family,
                                "job_id": job.id,
                                "attempt": attempt,
                                "action": "retry_fresh_hf_allocation_after_cuda802",
                            }
                        )
                    )

        print("LAUNCHED", json.dumps(launched, indent=2))
        return 0
    finally:
        if temp_dir_obj is not None:
            temp_dir_obj.cleanup()


if __name__ == "__main__":
    raise SystemExit(main())
