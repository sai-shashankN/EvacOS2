#!/usr/bin/env bash
set -euo pipefail

export DEBIAN_FRONTEND=noninteractive
export HF_HOME="${HF_HOME:-/workspace/hf_cache}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME}"
export WANDB_DISABLED="${WANDB_DISABLED:-true}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
export NVIDIA_VISIBLE_DEVICES="${NVIDIA_VISIBLE_DEVICES:-all}"
export LD_LIBRARY_PATH="/usr/local/nvidia/lib64:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"
if [[ "${CUDA_VISIBLE_DEVICES:-}" == "-1" || "${CUDA_VISIBLE_DEVICES:-}" == "none" ]]; then
  unset CUDA_VISIBLE_DEVICES
fi

FAMILY="${HF_SWEEP_FAMILY:-gas}"
SEEDS="${HF_SWEEP_SEEDS:-9101,9103,9107}"
MAX_ROUNDS="${HF_SWEEP_MAX_ROUNDS:-50}"
RUN_NAME="${HF_SWEEP_RUN_NAME:-gas-checkpoint-current-eval-sweep}"
ARTIFACT_REPO="${HF_ARTIFACT_REPO:?HF_ARTIFACT_REPO is required}"
SOURCE_REPO="${HF_SOURCE_REPO:-$ARTIFACT_REPO}"
SOURCE_FILENAME="${HF_SOURCE_FILENAME:?HF_SOURCE_FILENAME is required}"
OUTPUT_ROOT="/workspace/heldout_sweep/${RUN_NAME}"
export FAMILY SEEDS MAX_ROUNDS RUN_NAME ARTIFACT_REPO SOURCE_REPO SOURCE_FILENAME OUTPUT_ROOT

echo "HELDOUT_SWEEP_START $(date -Is) family=$FAMILY seeds=$SEEDS run=$RUN_NAME"

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "HF_TOKEN secret is required." >&2
  exit 2
fi

mkdir -p /workspace/source /workspace/EvacOS2_boot "$OUTPUT_ROOT" "$HF_HOME"

apt-get update
apt-get install -y --no-install-recommends git curl ca-certificates build-essential
python -m pip install --upgrade pip setuptools wheel
python -m pip install -q "huggingface_hub>=0.34.0,<1.0"

python - <<'PY'
import os
from huggingface_hub import hf_hub_download

p = hf_hub_download(
    repo_id=os.environ["SOURCE_REPO"],
    filename=os.environ["SOURCE_FILENAME"],
    repo_type="model",
    token=os.environ["HF_TOKEN"],
    local_dir="/workspace/source",
)
print(f"SOURCE_DOWNLOADED {p}", flush=True)
PY

tar -xzf "/workspace/source/$SOURCE_FILENAME" -C /workspace/EvacOS2_boot
cd /workspace/EvacOS2_boot

python -m pip install \
  "transformers==4.56.2" "trl==0.24.0" "peft==0.19.1" \
  accelerate bitsandbytes datasets "fsspec==2025.9.0" \
  "pydantic>=2,<3" "fastapi>=0.115" "uvicorn>=0.30" \
  "numpy>=1.26" pyyaml nbformat "wandb>=0.19" matplotlib pytest \
  "huggingface_hub>=0.34.0,<1.0"
python -m pip install --ignore-requires-python -e .

python - <<'PY'
import torch
print("torch_preflight", {
    "torch": torch.__version__,
    "cuda": torch.version.cuda,
    "available": torch.cuda.is_available(),
    "count": torch.cuda.device_count(),
}, flush=True)
if not torch.cuda.is_available():
    raise SystemExit("CUDA unavailable; cannot run held-out model sweep.")
print("gpu", torch.cuda.get_device_name(0), flush=True)
PY

python - <<'PY'
import json
import os
import subprocess
import sys
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

token = os.environ["HF_TOKEN"]
artifact_repo = os.environ["ARTIFACT_REPO"]
family = os.environ.get("FAMILY", "gas")
seeds = os.environ.get("SEEDS", "9101,9103,9107")
max_rounds = os.environ.get("MAX_ROUNDS", "50")
run_name = os.environ["RUN_NAME"]
output_root = Path(os.environ["OUTPUT_ROOT"])
candidates = json.loads(os.environ["HF_SWEEP_CANDIDATES_JSON"])
api = HfApi(token=token)
api.create_repo(repo_id=artifact_repo, repo_type="model", private=True, exist_ok=True)

summary = []
for candidate in candidates:
    name = candidate["name"]
    repo = candidate["repo"]
    path = candidate["path"].strip("/")
    print(f"CANDIDATE_START name={name} repo={repo} path={path}", flush=True)
    local_root = Path("/workspace/candidate_checkpoints") / name
    local_root.mkdir(parents=True, exist_ok=True)
    downloaded = Path(
        snapshot_download(
            repo_id=repo,
            repo_type="model",
            token=token,
            allow_patterns=[f"{path}/**"],
            local_dir=str(local_root),
        )
    )
    checkpoint = downloaded / path
    if not checkpoint.exists():
        raise SystemExit(f"Candidate checkpoint missing after download: {checkpoint}")
    out_dir = output_root / name
    cmd = [
        sys.executable,
        "-m",
        "evaluation.demo_bundle",
        "--trained-checkpoint",
        str(checkpoint),
        "--output-dir",
        str(out_dir),
        "--baseline-policy",
        "base_model",
        "--max-rounds",
        max_rounds,
    ]
    # demo_bundle CLI defaults all families, so use run_comparison directly in a tiny script.
    eval_script = f"""
from pathlib import Path
from evacos_ma.models import DisasterType
from evaluation.demo_bundle import build_demo_bundle
build_demo_bundle(
    trained_checkpoint=Path({str(checkpoint)!r}),
    output_dir=Path({str(out_dir)!r}),
    baseline_policy='base_model',
    seeds=[int(x) for x in {seeds!r}.split(',') if x],
    disaster_families=[DisasterType({family!r})],
    max_rounds=int({max_rounds!r}),
)
"""
    subprocess.run([sys.executable, "-c", eval_script], check=True)
    trained_json = out_dir / "fixed_suite_trained_linear_capped.json"
    baseline_json = out_dir / "fixed_suite_baseline_linear_capped.json"
    trained = json.loads(trained_json.read_text(encoding="utf-8"))
    baseline = json.loads(baseline_json.read_text(encoding="utf-8"))
    row = {
        "name": name,
        "repo": repo,
        "path": path,
        "baseline_score": baseline["aggregate"]["eval_score_pct"]["mean"],
        "trained_score": trained["aggregate"]["eval_score_pct"]["mean"],
        "delta_score": trained["aggregate"]["eval_score_pct"]["mean"] - baseline["aggregate"]["eval_score_pct"]["mean"],
        "baseline_invalid_rate": baseline["aggregate"]["invalid_action_rate"]["mean"],
        "trained_invalid_rate": trained["aggregate"]["invalid_action_rate"]["mean"],
        "trained_save_rate": trained["aggregate"]["save_rate"]["mean"],
        "trained_floor_route_missing_target_rate": sum(
            float(ep.get("floor_route_missing_target_rate", 0.0)) for ep in trained.get("episodes", [])
        ) / max(len(trained.get("episodes", [])), 1),
    }
    summary.append(row)
    print("CANDIDATE_RESULT " + json.dumps(row, sort_keys=True), flush=True)
    api.upload_folder(
        repo_id=artifact_repo,
        repo_type="model",
        folder_path=str(out_dir),
        path_in_repo=f"heldout/{run_name}/{name}",
        commit_message=f"Upload {run_name} {name} heldout eval",
    )

summary_path = output_root / "sweep_summary.json"
summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
passing = [
    row for row in summary
    if row["trained_floor_route_missing_target_rate"] <= 0.01
    and row["trained_invalid_rate"] <= 0.05
]
winner = max(passing or summary, key=lambda row: (row["delta_score"], row["trained_score"]))
winner_path = output_root / "winner.json"
winner_path.write_text(json.dumps(winner, indent=2, sort_keys=True), encoding="utf-8")
print("SWEEP_WINNER " + json.dumps(winner, sort_keys=True), flush=True)
api.upload_file(
    repo_id=artifact_repo,
    repo_type="model",
    path_or_fileobj=str(summary_path),
    path_in_repo=f"heldout/{run_name}/sweep_summary.json",
    commit_message=f"Upload {run_name} summary",
)
api.upload_file(
    repo_id=artifact_repo,
    repo_type="model",
    path_or_fileobj=str(winner_path),
    path_in_repo=f"heldout/{run_name}/winner.json",
    commit_message=f"Upload {run_name} winner",
)
PY

echo "HELDOUT_SWEEP_EXIT=0 $(date -Is)"
