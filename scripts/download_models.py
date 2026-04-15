"""Download (and optionally ONNX-quantize) the local models used by the server.

Run on first deploy:
    python scripts/download_models.py --models-dir ./models

Models pulled:
  - yaya36095/xlm-roberta-text-detector              (multilingual AI detector)
  - ai-forever/rugpt3small_based_on_gpt2             (perplexity, RU)
  - distilgpt2                                       (perplexity, EN)
  - intfloat/multilingual-e5-small                   (local embedding fallback)
"""

from __future__ import annotations

import argparse
from pathlib import Path

MODELS = [
    "yaya36095/xlm-roberta-text-detector",
    "ai-forever/rugpt3small_based_on_gpt2",
    "distilgpt2",
    "intfloat/multilingual-e5-small",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models-dir", type=Path, default=Path("./models"))
    args = parser.parse_args()
    args.models_dir.mkdir(parents=True, exist_ok=True)

    from huggingface_hub import snapshot_download

    for repo in MODELS:
        target = args.models_dir / repo.replace("/", "__")
        print(f"→ {repo}")
        snapshot_download(repo_id=repo, local_dir=target, local_dir_use_symlinks=False)
    print("Done.")


if __name__ == "__main__":
    main()
