from __future__ import annotations

import sys
from types import ModuleType

from scripts import upload_adapter


def test_upload_adapter_uses_hf_api_upload_folder(tmp_path, monkeypatch, capsys):
    adapter_path = tmp_path / "latest" / "lora_adapter"
    adapter_path.mkdir(parents=True)
    (adapter_path / "adapter_config.json").write_text("{}", encoding="utf-8")
    calls: list[tuple[str, dict[str, object]]] = []

    class FakeHfApi:
        def __init__(self, token=None):
            calls.append(("init", {"token": token}))

        def create_repo(self, **kwargs):
            calls.append(("create_repo", kwargs))

        def upload_folder(self, **kwargs):
            calls.append(("upload_folder", kwargs))

    fake_module = ModuleType("huggingface_hub")
    fake_module.HfApi = FakeHfApi
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_module)
    monkeypatch.setenv("HF_TOKEN", "hf_test_token")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "upload_adapter.py",
            str(adapter_path),
            "user/evacos2-adapter",
            "--path-in-repo",
            "fire/latest",
            "--commit-message",
            "test upload",
        ],
    )

    upload_adapter.main()

    assert calls[0] == ("init", {"token": "hf_test_token"})
    assert calls[1] == (
        "create_repo",
        {
            "repo_id": "user/evacos2-adapter",
            "repo_type": "model",
            "private": False,
            "exist_ok": True,
        },
    )
    assert calls[2][0] == "upload_folder"
    assert calls[2][1]["folder_path"] == str(adapter_path.resolve())
    assert calls[2][1]["repo_id"] == "user/evacos2-adapter"
    assert calls[2][1]["path_in_repo"] == "fire/latest"
    assert "https://huggingface.co/user/evacos2-adapter" in capsys.readouterr().out
