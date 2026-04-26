"""Launch fire/flood/gas specialist quality runs on HF H200 Jobs.

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
from pathlib import Path

from huggingface_hub import HfApi

DEFAULT_ASSIGNMENTS = {
    "fire": "HFALT1_TOKEN",
    "flood": "HFALT2_TOKEN",
    "gas": "HFALT3_TOKEN",
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
            "apt-get update && "
            "apt-get install -y --no-install-recommends git ca-certificates && "
            "git clone --depth 1 --branch \"$EVACOS_REPO_REF\" "
            "\"$EVACOS_REPO_URL\" /workspace/EvacOS2_boot && "
            "cd /workspace/EvacOS2_boot && "
            "bash scripts/hf_h200_specialist_job.sh"
        ),
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--repo-url", default="https://github.com/sai-shashankN/EvacOS2.git")
    parser.add_argument("--repo-ref", default="main")
    parser.add_argument("--image", default="pytorch/pytorch:2.7.1-cuda12.6-cudnn9-devel")
    parser.add_argument("--flavor", default="h200")
    parser.add_argument("--timeout", default="4h")
    parser.add_argument("--only", choices=["fire", "flood", "gas"], action="append")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    env_values = {**os.environ, **_load_env(args.env_file)}
    families = args.only or ["fire", "flood", "gas"]
    launched: list[dict[str, str]] = []

    for family in families:
        token_key = DEFAULT_ASSIGNMENTS[family]
        token = env_values.get(token_key, "").strip()
        if not token:
            raise SystemExit(f"Missing {token_key} in environment or {args.env_file}")

        api = HfApi(token=token)
        who = api.whoami()
        namespace = who["name"]
        labels = {
            "project": "evacos2",
            "run": f"{family}-quality",
            "family": family,
        }
        job_env = {
            "DISASTER_FAMILY": family,
            "EVACOS_REPO_URL": args.repo_url,
            "EVACOS_REPO_REF": args.repo_ref,
            "HF_ARTIFACT_REPO": f"{namespace}/evacos2-h200-specialist-artifacts",
        }
        print(
            json.dumps(
                {
                    "family": family,
                    "token_key": token_key,
                    "namespace": namespace,
                    "flavor": args.flavor,
                    "timeout": args.timeout,
                    "repo_ref": args.repo_ref,
                    "artifact_repo": job_env["HF_ARTIFACT_REPO"],
                    "dry_run": args.dry_run,
                },
                indent=2,
            )
        )
        if args.dry_run:
            continue

        job = api.run_job(
            image=args.image,
            command=_job_command(args.repo_ref),
            env=job_env,
            secrets={"HF_TOKEN": token},
            flavor=args.flavor,  # type: ignore[arg-type]
            timeout=args.timeout,
            labels=labels,
            token=token,
        )
        launched.append(
            {
                "family": family,
                "job_id": job.id,
                "namespace": namespace,
                "artifact_repo": job_env["HF_ARTIFACT_REPO"],
            }
        )
        print(json.dumps(launched[-1], indent=2))

    print("LAUNCHED", json.dumps(launched, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
