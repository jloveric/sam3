#!/usr/bin/env python3
"""Download SAM3 model weights from Hugging Face."""

from __future__ import annotations

import argparse

from huggingface_hub import hf_hub_download


def download(version: str = "sam3") -> str:
    if version == "sam3.1":
        repo_id = "facebook/sam3.1"
        ckpt_name = "sam3.1_multiplex.pt"
    else:
        repo_id = "facebook/sam3"
        ckpt_name = "sam3.pt"

    hf_hub_download(repo_id=repo_id, filename="config.json")
    return hf_hub_download(repo_id=repo_id, filename=ckpt_name)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download SAM3 weights from Hugging Face")
    parser.add_argument(
        "--version",
        choices=["sam3", "sam3.1"],
        default="sam3",
        help="Checkpoint version to download (default: sam3)",
    )
    args = parser.parse_args()
    path = download(args.version)
    print(path)


if __name__ == "__main__":
    main()
