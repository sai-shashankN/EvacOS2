# EvacOS-MA OpenEnv Package

Provides the OpenEnv-compatible surface for the live EvacOS-MA multi-agent evacuation benchmark.

## Surface

| Endpoint | Method | Request | Response |
|---|---|---|---|
| `/openenv/health` | GET | — | `{status, version}` |
| `/openenv/schema` | GET | — | JSON Schema for ActionBundleMA, observations, StepResultMA |
| `/openenv/metadata` | GET | — | `{name, description, version, manifest}` |
| `/openenv/reset` | POST | `{task_id, seed?, disaster_family?, max_steps?}` | `{episode_id, step_result}` |
| `/openenv/step` | POST | `ActionBundleMA` | `{step_result}` |
| `/openenv/state` | GET | `?episode_id=` | metadata only, or full state if `EVACOS_DEBUG_STATE=true` |

## Usage

```python
from evacos_ma.openenv.client import OpenEnvClient

client = OpenEnvClient("http://localhost:8000")
resp = client.reset(seed=42)
print(resp["episode_id"])

# Later:
# from evacos_ma.schemas.multi_agent import ActionBundleMA
# bundle = ActionBundleMA(...)
# result = client.step(bundle)
```

## Local Server

Run the API locally with:

```bash
uvicorn evacos_ma.api:app --host 0.0.0.0 --port 8000
```

The current implementation is backed by the real simulator, not a canned stub.

## Procedural Scenario Reset

The public scenario IDs are:

- `openenv_fire_response`
- `openenv_flood_response`
- `openenv_gas_response`

The client example above uses the default fixed proof task. You can also drive a
procedural reset by posting a `disaster_family` directly. The server keeps the
internal generator contract stable; public callers choose only the scenario
family.

Example:

```json
{
  "task_id": "openenv_fire_response",
  "seed": 42,
  "disaster_family": "fire",
  "max_steps": 80
}
```

## Debug State

Set the environment variable `EVACOS_DEBUG_STATE=true` to enable full-state
payloads on `/openenv/state`. Without this flag, only metadata is returned.
