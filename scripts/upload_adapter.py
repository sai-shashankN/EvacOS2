"""Upload a trained LoRA adapter folder to the Hugging Face Hub.

This intentionally uploads the adapter artifact only, not raw training logs or
large demo videos. Authentication is handled by the standard HF_TOKEN env var
or an existing `hf auth login` session.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "adapter_path",
        type=Path,
        help="Path to a PEFT/LoRA adapter folder, usually latest/lora_adapter.",
    )
    parser.add_argument(
        "repo_id",
        help="Destination Hub repo, e.g. username/evacos2-lora-adapter.",
    )
    parser.add_argument(
        "--repo-type",
        default="model",
        choices=("model", "dataset", "space"),
        help="Hub repo type. Defaults to model.",
    )
    parser.add_argument(
        "--path-in-repo",
        default=".",
        help="Destination path inside the repo. Defaults to repo root.",
    )
    parser.add_argument(
        "--commit-message",
        default="Upload EvacOS2 LoRA adapter",
        help="Hub commit message.",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create the repo as private if it does not already exist.",
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    adapter_path = args.adapter_path.expanduser().resolve()
    if not adapter_path.exists() or not adapter_path.is_dir():
        raise SystemExit(f"adapter_path does not exist or is not a directory: {adapter_path}")

    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise SystemExit(
            "huggingface_hub is required. Install training deps or run `pip install huggingface_hub`."
        ) from exc

    token = os.environ.get("HF_TOKEN") or None
    api = HfApi(token=token)
    api.create_repo(
        repo_id=args.repo_id,
        repo_type=args.repo_type,
        private=args.private,
        exist_ok=True,
    )
    api.upload_folder(
        folder_path=str(adapter_path),
        repo_id=args.repo_id,
        repo_type=args.repo_type,
        path_in_repo=args.path_in_repo,
        commit_message=args.commit_message,
    )
    print(f"Uploaded {adapter_path} to https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
