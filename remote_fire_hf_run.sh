set -u
export HF_HOME=/workspace/hf_cache
export TRANSFORMERS_CACHE=/workspace/hf_cache
export WANDB_DISABLED=true
cd /workspace/EvacOS2
source .venv/bin/activate

cat > training/config.fire-hour-hf.yaml <<'YAML'
backend: "hf"
max_steps: 20
unsloth_max_seq_length: 2048
load_in_4bit: false
model:
  base: "Qwen/Qwen2.5-3B-Instruct"
  dtype: "bfloat16"
  max_prompt_tokens: 1024
  max_completion_tokens: 128
roles:
  trainable: ["floor_agent"]
  orchestrator_policy: "stub"
lora:
  rank: 16
  alpha: 32
  dropout: 0.05
  target_modules: ["q_proj", "k_proj", "v_proj", "o_proj"]
rollout:
  episodes_per_step: 1
  max_rounds_per_episode: 4
  seed_retry_limit: 200
  use_vllm: false
  disaster_families: ["fire"]
grpo:
  learning_rate: 5.0e-6
  kl_coef: 0.04
  clip_range: 0.2
  num_train_epochs_per_step: 1
reward:
  rationale_scaling: "linear_capped"
  alpha: 0.01
  beta: 0.25
  cap: 1.0
  eligible_token_ceiling: 96
  clip_normalized_to: 1.0
eval:
  every_steps: 10
  tiers: ["easy"]
  seeds: [42, 123]
checkpoint:
  every_steps: 5
  keep_last_n: 5
  root_dir: "outputs/training/fire-3b-hour-hf"
metrics:
  csv_path: "outputs/training/fire-3b-hour-hf-metrics.csv"
  jsonl_dir: "outputs/logs/fire-3b-hour-hf"
seed:
  training_rng: 12345
YAML

rm -rf outputs/training/fire-3b-hour-hf \
  outputs/training/fire-3b-hour-hf-metrics.csv \
  outputs/logs/fire-3b-hour-hf \
  /root/evacos2_fire_3b_artifacts.tgz

python - <<'PY'
from training.compat import patch_transformers_cache_exports

patch_transformers_cache_exports()
from peft import LoraConfig
from trl import GRPOTrainer
import torch

print(
    "stack_ok",
    torch.__version__,
    torch.cuda.is_available(),
    torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    flush=True,
)
PY

START_TS=$(date -Is)
echo "TRAIN_START=$START_TS"
timeout 3300s python -m training.train training/config.fire-hour-hf.yaml
TRAIN_EXIT=$?
echo "TRAIN_EXIT=$TRAIN_EXIT"

python - <<'PY'
from pathlib import Path
import csv
import json

metrics = Path("outputs/training/fire-3b-hour-hf-metrics.csv")
report = Path("outputs/training/fire_3b_hour_report.json")
rows = []
if metrics.exists():
    with metrics.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

summary = {"metrics_path": str(metrics), "rows": len(rows)}
if rows:
    first = rows[0]
    last = rows[-1]
    keys = [
        "step",
        "invalid_action_rate",
        "mean_norm_reward_floor",
        "mean_raw_reward_floor",
        "raw_reward_std_floor",
        "norm_reward_std_floor",
        "wait_rate",
        "active_empty_args_rate",
        "floor_agent_valid_but_hollow_action_rate",
    ]
    summary["first"] = {k: first.get(k) for k in keys if k in first}
    summary["last"] = {k: last.get(k) for k in keys if k in last}
    for key in ["mean_norm_reward_floor", "mean_raw_reward_floor", "invalid_action_rate"]:
        vals = []
        for row in rows[-10:]:
            try:
                vals.append(float(row[key]))
            except Exception:
                pass
        if vals:
            summary[f"last10_avg_{key}"] = round(sum(vals) / len(vals), 6)

ckpt = Path("outputs/training/fire-3b-hour-hf/latest")
summary["latest_checkpoint_exists"] = ckpt.exists()
summary["latest_checkpoint_files"] = (
    [str(p.relative_to(ckpt)) for p in ckpt.rglob("*")][:100] if ckpt.exists() else []
)
report.parent.mkdir(parents=True, exist_ok=True)
report.write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(json.dumps(summary, indent=2), flush=True)
PY

cd /workspace
tar -czf /root/evacos2_fire_3b_artifacts.tgz \
  EvacOS2/outputs/training/fire-3b-hour-hf \
  EvacOS2/outputs/training/fire-3b-hour-hf-metrics.csv \
  EvacOS2/outputs/training/fire_3b_hour_report.json \
  EvacOS2/training/config.fire-hour-hf.yaml
