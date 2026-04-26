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
    parser.add_argument("--only", choices=["fire", "flood", "gas"], action="append")
    parser.add_argument("--source-tgz", type=Path)
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
            }
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

            job = api.run_job(
                image=args.image,
                command=_job_command(args.repo_ref),
                env=job_env,
                secrets={"HF_TOKEN": token},
                flavor=args.flavor,  # type: ignore[arg-type]
                timeout=args.timeout,
                labels=labels,
                namespace=namespace,
                token=token,
            )
            launched.append(
                {
                    "family": family,
                    "job_id": job.id,
                    "namespace": namespace,
                    "artifact_repo": artifact_repo,
                }
            )
            print(json.dumps(launched[-1], indent=2))

        print("LAUNCHED", json.dumps(launched, indent=2))
        return 0
    finally:
        if temp_dir_obj is not None:
            temp_dir_obj.cleanup()


if __name__ == "__main__":
    raise SystemExit(main())
