#!/usr/bin/env bash
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
export HF_HOME=/workspace/hf_cache
export TRANSFORMERS_CACHE=/workspace/hf_cache
export WANDB_DISABLED=true
mkdir -p /workspace /workspace/hf_cache
cd /workspace
rm -rf EvacOS2
mkdir -p EvacOS2
python3 - <<'PY'
import zipfile
zipfile.ZipFile('/root/evacos2_remote_upload.zip').extractall('/workspace/EvacOS2')
PY
cd /workspace/EvacOS2
python3 -m venv .venv || (apt-get update && apt-get install -y python3-venv && python3 -m venv .venv)
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install "torch==2.10.0" "torchvision==0.25.0"
python -m pip install "transformers==4.56.2" "trl==0.24.0" "peft==0.19.1" accelerate bitsandbytes datasets "fsspec==2025.9.0" "pydantic>=2,<3" "fastapi>=0.115" "uvicorn>=0.30" "numpy>=1.26" pyyaml nbformat "wandb>=0.19" matplotlib
python -m pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
python -m pip install --no-deps "trl==0.24.0" "peft==0.19.1" accelerate bitsandbytes
python -m pip install -e .
python - <<'PY'
import torch
print('torch', torch.__version__, torch.cuda.is_available(), torch.version.cuda, 'has_int1', hasattr(torch, 'int1'))
print('gpu', torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
try:
    import unsloth
    print('unsloth OK', getattr(unsloth, '__version__', 'no_version'))
except Exception as e:
    print('unsloth FAIL', repr(e))
    raise
PY
cp training/config.remote-unsloth-3b-fire-floor-specialist.yaml training/config.fire-hour.yaml
python - <<'PY'
from pathlib import Path
p = Path('training/config.fire-hour.yaml')
text = p.read_text()
text = text.replace('max_steps: 100', 'max_steps: 30')
p.write_text(text)
print(p.read_text())
PY
mkdir -p /workspace/EvacOS2/outputs/training
START_TS=$(date -Is)
echo "TRAIN_START=$START_TS"
set +e
timeout 3600s python -m training.train training/config.fire-hour.yaml
TRAIN_EXIT=$?
set -e
echo "TRAIN_EXIT=$TRAIN_EXIT"
python - <<'PY'
from pathlib import Path
import csv, json, statistics
metrics = Path('outputs/training/remote-unsloth-3b-fire-floor-specialist-metrics.csv')
report = Path('outputs/training/fire_3b_hour_report.json')
rows=[]
if metrics.exists():
    with metrics.open(newline='', encoding='utf-8') as f:
        rows=list(csv.DictReader(f))
summary={'metrics_path': str(metrics), 'rows': len(rows)}
if rows:
    first=rows[0]; last=rows[-1]
    keys=['step','invalid_action_rate','mean_norm_reward_floor','mean_raw_reward_floor','raw_reward_std_floor','wait_rate','active_empty_args_rate','floor_agent_valid_but_hollow_action_rate']
    summary['first']={k:first.get(k) for k in keys if k in first}
    summary['last']={k:last.get(k) for k in keys if k in last}
    for k in ['mean_norm_reward_floor','mean_raw_reward_floor','invalid_action_rate']:
        vals=[]
        for r in rows[-10:]:
            try: vals.append(float(r[k]))
            except Exception: pass
        if vals:
            summary[f'last10_avg_{k}']=sum(vals)/len(vals)
ckpt=Path('outputs/training/remote-unsloth-3b-fire-floor-specialist/latest')
summary['latest_checkpoint_exists']=ckpt.exists()
summary['latest_checkpoint_files']=[str(p.relative_to(ckpt)) for p in ckpt.rglob('*')][:50] if ckpt.exists() else []
report.write_text(json.dumps(summary, indent=2), encoding='utf-8')
print(json.dumps(summary, indent=2))
PY
cd /workspace
tar -czf /root/evacos2_fire_3b_artifacts.tgz \
  EvacOS2/outputs/training/remote-unsloth-3b-fire-floor-specialist \
  EvacOS2/outputs/training/remote-unsloth-3b-fire-floor-specialist-metrics.csv \
  EvacOS2/outputs/training/fire_3b_hour_report.json \
  EvacOS2/training/config.fire-hour.yaml || true
exit "$TRAIN_EXIT"
