"""Launch the main-account HF 7B orchestrator job.

This is intentionally separate from the 3B specialist launcher. It assumes the
selected fire/flood/gas floor-agent adapters have already been uploaded to Hub
model repos and points the 7B job at those frozen adapter directories.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
from pathlib import Path

from huggingface_hub import HfApi


DEFAULT_NAMESPACE = "shashankN777"
DEFAULT_ARTIFACT_REPO = f"{DEFAULT_NAMESPACE}/evacos2-7b-orchestrator-artifacts"
DEFAULT_SOURCE_PATH_PREFIX = "source"


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


def _git_sha() -> str:
    return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()


def _create_source_archive(target: Path) -> None:
    subprocess.run(
        ["git", "archive", "--format=tar.gz", "-o", str(target), "HEAD"],
        check=True,
    )


def _job_command() -> list[str]:
    return [
        "bash",
        "-lc",
        (
            "python -m pip install -q 'huggingface_hub>=0.34.0,<1.0' && "
            "mkdir -p /workspace/source /workspace/EvacOS2 && "
            "python -c \"import os, tarfile; from huggingface_hub import hf_hub_download; "
            "p=hf_hub_download(repo_id=os.environ['HF_SOURCE_REPO'], "
            "filename=os.environ['HF_SOURCE_FILENAME'], repo_type='model', "
            "token=os.environ['HF_TOKEN'], local_dir='/workspace/source'); "
            "tarfile.open(p, 'r:gz').extractall('/workspace/EvacOS2')\" && "
            "bash -lc 'EVACOS_SOURCE_READY=1 EVACOS_WORKDIR=/workspace/EvacOS2 "
            "bash /workspace/EvacOS2/scripts/hf_7b_orchestrator_job.sh'"
        ),
    ]


def _env_or_arg(env: dict[str, str], arg: str | None, key: str) -> str:
    value = (arg or env.get(key, "")).strip()
    if not value:
        raise SystemExit(f"Missing {key}; pass the matching CLI flag or set it in .env")
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--namespace", default=None)
    parser.add_argument("--artifact-repo", default=None)
    parser.add_argument("--image", default="pytorch/pytorch:2.7.1-cuda12.6-cudnn9-devel")
    parser.add_argument("--flavor", default="h100")
    parser.add_argument("--timeout", default="4h")
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--run-name", default="remote-unsloth-7b-orchestrator-frozen-specialists-main")
    parser.add_argument("--fire-adapter-repo")
    parser.add_argument("--fire-adapter-path")
    parser.add_argument("--flood-adapter-repo")
    parser.add_argument("--flood-adapter-path")
    parser.add_argument("--gas-adapter-repo")
    parser.add_argument("--gas-adapter-path")
    parser.add_argument("--source-tgz", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    env_values = {**os.environ, **_load_env(args.env_file)}
    token = env_values.get("HFMAIN_TOKEN") or env_values.get("HF_TOKEN")
    if not token:
        raise SystemExit("Missing HFMAIN_TOKEN or HF_TOKEN")

    namespace = args.namespace or env_values.get("HFMAIN_NAMESPACE") or DEFAULT_NAMESPACE
    artifact_repo = args.artifact_repo or env_values.get("HF_7B_ARTIFACT_REPO") or DEFAULT_ARTIFACT_REPO
    api = HfApi(token=token)
    sha = _git_sha()

    temp_dir_obj: tempfile.TemporaryDirectory[str] | None = None
    if args.source_tgz is None:
        temp_dir_obj = tempfile.TemporaryDirectory()
        source_tgz = Path(temp_dir_obj.name) / f"evacos2_source_{sha}.tgz"
        _create_source_archive(source_tgz)
    else:
        source_tgz = args.source_tgz
    source_filename = f"{DEFAULT_SOURCE_PATH_PREFIX}/{source_tgz.name}"

    job_env = {
        "RUN_NAME": args.run_name,
        "STEPS": str(args.steps),
        "HF_7B_ARTIFACT_REPO": artifact_repo,
        "HF_SOURCE_REPO": artifact_repo,
        "HF_SOURCE_FILENAME": source_filename,
        "HF_FIRE_ADAPTER_REPO": _env_or_arg(env_values, args.fire_adapter_repo, "HF_FIRE_ADAPTER_REPO"),
        "HF_FIRE_ADAPTER_PATH": _env_or_arg(env_values, args.fire_adapter_path, "HF_FIRE_ADAPTER_PATH"),
        "HF_FLOOD_ADAPTER_REPO": _env_or_arg(env_values, args.flood_adapter_repo, "HF_FLOOD_ADAPTER_REPO"),
        "HF_FLOOD_ADAPTER_PATH": _env_or_arg(env_values, args.flood_adapter_path, "HF_FLOOD_ADAPTER_PATH"),
        "HF_GAS_ADAPTER_REPO": _env_or_arg(env_values, args.gas_adapter_repo, "HF_GAS_ADAPTER_REPO"),
        "HF_GAS_ADAPTER_PATH": _env_or_arg(env_values, args.gas_adapter_path, "HF_GAS_ADAPTER_PATH"),
    }
    plan = {
        "namespace": namespace,
        "artifact_repo": artifact_repo,
        "flavor": args.flavor,
        "timeout": args.timeout,
        "steps": args.steps,
        "run_name": args.run_name,
        "source": f"{artifact_repo}/{source_filename}",
        "dry_run": args.dry_run,
    }
    print(json.dumps(plan, indent=2))

    try:
        if args.dry_run:
            return 0

        api.create_repo(repo_id=artifact_repo, repo_type="model", private=False, exist_ok=True, token=token)
        api.upload_file(
            repo_id=artifact_repo,
            repo_type="model",
            path_or_fileobj=str(source_tgz),
            path_in_repo=source_filename,
            commit_message=f"Upload EvacOS2 7B source {sha}",
            token=token,
        )
        job = api.run_job(
            image=args.image,
            command=_job_command(),
            env=job_env,
            secrets={"HF_TOKEN": token},
            flavor=args.flavor,  # type: ignore[arg-type]
            timeout=args.timeout,
            labels={"project": "evacos2", "run": "7b-orchestrator", "sha": sha},
            namespace=namespace,
            token=token,
        )
        print(json.dumps({"job_id": job.id, "namespace": namespace, "artifact_repo": artifact_repo}, indent=2))
        return 0
    finally:
        if temp_dir_obj is not None:
            temp_dir_obj.cleanup()


if __name__ == "__main__":
    raise SystemExit(main())
