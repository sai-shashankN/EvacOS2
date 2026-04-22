# Demo Artifacts

The files in this folder are presentation assets, not the only demo surface.

- `hf_blog.md` is a mini-blog scaffold meant to be filled with generated evaluation outputs.
- `storyboard.md` is a shot list for presenting those generated artifacts.
- The functional offline renderer bridge lives in [`renderer/unity_bridge.py`](../renderer/unity_bridge.py) and is covered by [`tests/test_unity_bridge.py`](../tests/test_unity_bridge.py).
- The live API-facing demo surface now includes the OpenEnv routes in [`evacos_ma/openenv/server_shell.py`](../evacos_ma/openenv/server_shell.py).
- The local interactive inspection surface lives under `dashboard/`.
