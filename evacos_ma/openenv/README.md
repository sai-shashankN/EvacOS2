# EvacOS-MA OpenEnv Package

Provides the OpenEnv-compatible surface for the EvacOS-MA multi-agent evacuation benchmark.

## Surface

| Endpoint | Method | Request | Response |
|---|---|---|---|
| `/openenv/health` | GET | — | `{status, version}` |
| `/openenv/schema` | GET | — | JSON Schema for ActionBundleMA, observations, StepResultMA |
| `/openenv/metadata` | GET | — | `{name, description, version, manifest}` |
| `/openenv/reset` | POST | `{task_id, seed?, tier}` | `{episode_id, step_result}` |
| `/openenv/step` | POST | `ActionBundleMA` | `{step_result}` |
| `/openenv/state` | GET | `?episode_id=` | metadata only, or full state if `EVACOS_DEBUG_STATE=true` |

## Usage

```python
from evacos_ma.openenv.client import OpenEnvClient

client = OpenEnvClient("http://localhost:8000")
resp = client.reset(task_id="task_1_fire_easy", seed=42)
print(resp["episode_id"])

# Later:
# from evacos_ma.schemas.multi_agent import ActionBundleMA
# bundle = ActionBundleMA(...)
# result = client.step(bundle)
```

## Debug State

Set the environment variable `EVACOS_DEBUG_STATE=true` to enable full-state
payloads on `/openenv/state`. Without this flag, only metadata is returned.
