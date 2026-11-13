#!/usr/bin/env python3
"""Download only the MiniMax-H3 files required by the selected recipe."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

TOTAL_BUDGET_GB = float(os.environ.get("TRAINER_TOTAL_BUDGET_GB", "100"))
MODEL_BUDGET_GB = float(os.environ.get("TRAINER_MODEL_BUDGET_GB", "80"))


def tree_bytes(paths: list[Path]) -> int:
    total = 0
    seen = set()
    for root in paths:
        if not root.exists():
            continue
        for current, _, files in os.walk(root):
            for filename in files:
                try:
                    stat = (Path(current) / filename).stat()
                except OSError:
                    continue
                identity = (stat.st_dev, stat.st_ino)
                if identity not in seen:
                    seen.add(identity)
                    total += stat.st_size
    return total


PROFILES = {
    "quantized": {
        "DiffSynth-Studio/MiniMax-H3-Image-EntroPack": ["models/dit_7bit.safetensors", "models/dit_7bit.json", "models/vae_int8.safetensors"],
        "DiffSynth-Studio/EntroPackPreQuants": ["models/MiniMax-H3/text_encoder_4bpp.safetensors", "models/MiniMax-H3/text_encoder_4bpp.json"],
        "MiniMax/MiniMax-H3-Image": ["pdd_fused_heads.safetensors", "processor/**"],
    },
    "bf16": {
        "MiniMax/MiniMax-H3": ["FL2VA/text_encoder/**"],
        "MiniMax/MiniMax-H3-Image": ["transformer/**", "pdd_fused_heads.safetensors", "vae/**"],
    },
}

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=PROFILES, default="quantized")
    parser.add_argument("--cache-dir", required=True)
    args = parser.parse_args()
    cache_dir = Path(args.cache_dir).expanduser().resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    print(f"[download] python={sys.executable}", flush=True)
    print(f"[download] prefix={sys.prefix} base_prefix={sys.base_prefix}", flush=True)
    from modelscope import snapshot_download
    print(f"[download] modelscope={Path(sys.modules['modelscope'].__file__).resolve()}", flush=True)
    print(f"[download] profile={args.profile} cache={cache_dir}", flush=True)
    home = cache_dir.parent
    model_paths = [cache_dir, home / "models"]
    model_gb = tree_bytes(model_paths) / 2**30
    total_gb = tree_bytes([home]) / 2**30
    print(f"[storage] model={model_gb:.1f}/{MODEL_BUDGET_GB:.0f}GB total={total_gb:.1f}/{TOTAL_BUDGET_GB:.0f}GB", flush=True)
    if model_gb >= MODEL_BUDGET_GB or total_gb >= TOTAL_BUDGET_GB:
        raise SystemExit("Storage budget reached; model download is blocked to preserve generated-data space.")
    for model_id, patterns in PROFILES[args.profile].items():
        print(f"[download] checking {model_id}: {', '.join(patterns)}", flush=True)
        path = snapshot_download(model_id, revision="master", cache_dir=str(cache_dir), allow_file_pattern=patterns)
        print(f"[download] ready: {path}", flush=True)
        model_gb = tree_bytes(model_paths) / 2**30
        total_gb = tree_bytes([home]) / 2**30
        print(f"[storage] model={model_gb:.1f}/{MODEL_BUDGET_GB:.0f}GB total={total_gb:.1f}/{TOTAL_BUDGET_GB:.0f}GB", flush=True)
        if model_gb > MODEL_BUDGET_GB or total_gb > TOTAL_BUDGET_GB:
            raise SystemExit("Storage budget exceeded; stop and remove unneeded model files before continuing.")
    print("[download] all required H3 weights are ready", flush=True)

if __name__ == "__main__":
    main()
