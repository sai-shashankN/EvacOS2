"""Launch a small HF H200 held-out sweep for 3B specialist checkpoints.

The default use is the gas recovery loop: evaluate candidate checkpoints under
the current fixed-suite evaluator, then select the checkpoint that avoids the
missing-route-target failure.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
from pathlib import Path

from huggingface_hub import HfApi


DEFAULT_CANDIDATES = [
    {
        "name": "gas-vast-ckpt49",
        "repo": "shashankN777/evacos2-7b-orchestrator-artifacts",
        "path": "floor-specialists/gas/vast-canary50/checkpoints/ckpt_49",
    },
    {
        "name": "gas-h200-ckpt89",
        "repo": "shashankN777/evacos2-7b-orchestrator-artifacts",
        "path": "floor-specialists/gas/h200-resume200-from-vast50/checkpoints/ckpt_89",
    },
    {
        "name": "gas-h200-ckpt199",
        "repo": "shashankN777/evacos2-7b-orchestrator-artifacts",
        "path": "floor-specialists/gas/h200-resume200-ckpt199/checkpoints/ckpt_199",
    },
    {
        "name": "gas-parsefix-ckpt249",
        "repo": "hfnasjdjas/evacos2-h200-specialist-artifacts",
        "path": "runs/remote-unsloth-3b-gas-floor-specialist-parsefix250c-alt1-250/checkpoints/ckpt_249",
    },
]


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
            "mkdir -p /workspace/source /workspace/EvacOS2_boot && "
            "python -c \"import os; from huggingface_hub import hf_hub_download; "
            "p=hf_hub_download(repo_id=os.environ['HF_SOURCE_REPO'], "
            "filename=os.environ['HF_SOURCE_FILENAME'], repo_type='model', "
            "token=os.environ['HF_TOKEN'], local_dir='/workspace/source'); print(p)\" && "
            "tar -xzf \"/workspace/source/$HF_SOURCE_FILENAME\" -C /workspace/EvacOS2_boot && "
            "cd /workspace/EvacOS2_boot && "
            "bash scripts/hf_3b_heldout_sweep_job.sh"
        ),
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--token-env", default="HFALT1_TOKEN")
    parser.add_argument("--namespace", default="hfnasjdjas")
    parser.add_argument("--artifact-repo", default="")
    parser.add_argument("--family", default="gas")
    parser.add_argument("--seeds", default="9101,9103,9107")
    parser.add_argument("--max-rounds", type=int, default=50)
    parser.add_argument("--flavor", default="h200")
    parser.add_argument("--timeout", default="3h")
    parser.add_argument("--image", default="pytorch/pytorch:2.7.1-cuda12.6-cudnn9-devel")
    parser.add_argument("--run-label", default="")
    parser.add_argument("--source-tgz", type=Path)
    parser.add_argument("--candidates-json", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    env_values = {**os.environ, **_load_env(args.env_file)}
    token = env_values.get(args.token_env, "").strip()
    if not token:
        raise SystemExit(f"Missing {args.token_env} in environment or {args.env_file}")

    namespace = args.namespace
    artifact_repo = args.artifact_repo or f"{namespace}/evacos2-h200-specialist-artifacts"
    sha = _git_sha()
    run_label = args.run_label or f"{args.family}-candidate-sweep-{sha}"
    temp_dir_obj: tempfile.TemporaryDirectory[str] | None = None
    if args.source_tgz is None:
        temp_dir_obj = tempfile.TemporaryDirectory()
        source_tgz = Path(temp_dir_obj.name) / f"evacos2_source_{sha}.tgz"
        _create_source_archive(source_tgz)
    else:
        source_tgz = args.source_tgz
    source_filename = f"source/{source_tgz.name}"
    candidates = (
        json.loads(args.candidates_json.read_text(encoding="utf-8"))
        if args.candidates_json
        else DEFAULT_CANDIDATES
    )

    payload = {
        "namespace": namespace,
        "artifact_repo": artifact_repo,
        "family": args.family,
        "seeds": args.seeds,
        "max_rounds": args.max_rounds,
        "flavor": args.flavor,
        "timeout": args.timeout,
        "run_label": run_label,
        "source": f"{artifact_repo}/{source_filename}",
        "candidate_count": len(candidates),
        "dry_run": args.dry_run,
    }
    print(json.dumps(payload, indent=2))
    if args.dry_run:
        return 0

    try:
        api = HfApi(token=token)
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
            command=_job_command(),
            env={
                "ARTIFACT_REPO": artifact_repo,
                "HF_ARTIFACT_REPO": artifact_repo,
                "HF_SOURCE_REPO": artifact_repo,
                "HF_SOURCE_FILENAME": source_filename,
                "HF_SWEEP_FAMILY": args.family,
                "HF_SWEEP_SEEDS": args.seeds,
                "HF_SWEEP_MAX_ROUNDS": str(args.max_rounds),
                "HF_SWEEP_RUN_NAME": run_label,
                "HF_SWEEP_CANDIDATES_JSON": json.dumps(candidates),
            },
            secrets={"HF_TOKEN": token},
            flavor=args.flavor,  # type: ignore[arg-type]
            timeout=args.timeout,
            labels={"project": "evacos2", "run": "heldout-sweep", "family": args.family},
            namespace=namespace,
            token=token,
        )
        print(
            json.dumps(
                {
                    "job_id": job.id,
                    "namespace": namespace,
                    "artifact_repo": artifact_repo,
                    "run_label": run_label,
                },
                indent=2,
            )
        )
    finally:
        if temp_dir_obj is not None:
            temp_dir_obj.cleanup()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
