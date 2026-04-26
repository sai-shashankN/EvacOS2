# EvacOS2 Dashboard

Run locally:

```bash
uvicorn dashboard.server:app --port 8765
```

The dashboard is read-only and consumes `outputs/logs/*.jsonl` by default.
Set `EVACOS_DASHBOARD_LOG_ROOT` to inspect another log directory.
