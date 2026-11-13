#!/usr/bin/env python3
"""Download only the MiniMax-H3 files required by the selected recipe."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

TOTAL_BUDGET_GB = float(os.environ.get("TRAINER_TOTAL_BUDGET_GB", "100"))
MODEL_BUDGET_GB = float(os.environ.get("TRAINER_MODEL_BUDGET_GB", "80"))
HOME = Path(os.environ.get("TRAINER_HOME", "/mnt/workspace/pose-model-trainer-data")).expanduser().resolve()
DEFAULT_MODELS_DIR = HOME / "models"


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
        "MiniMax/MiniMax-H3-Image": ["transformer/**", "pdd_fused_heads.safetensors", "vae/**", "processor/**"],
    },
}

QUANTIZED_REQUIRED = {
    "DiffSynth-Studio/MiniMax-H3-Image-EntroPack": [
        "models/dit_7bit.safetensors", "models/dit_7bit.json", "models/vae_int8.safetensors",
    ],
    "DiffSynth-Studio/EntroPackPreQuants": [
        "models/MiniMax-H3/text_encoder_4bpp.safetensors", "models/MiniMax-H3/text_encoder_4bpp.json",
    ],
    "MiniMax/MiniMax-H3-Image": [
        "pdd_fused_heads.safetensors",
        "processor/chat_template.json", "processor/merges.txt", "processor/preprocessor_config.json",
        "processor/tokenizer.json", "processor/tokenizer_config.json",
        "processor/video_preprocessor_config.json", "processor/vocab.json",
    ],
}
REQUIRED_FILES = {"quantized": QUANTIZED_REQUIRED, "bf16": {}}
RECEIPT_NAME = ".h3-download-receipts.json"


def model_files(model_root: Path) -> dict[str, int]:
    if not model_root.is_dir():
        return {}
    files = {}
    for path in sorted(model_root.rglob("*")):
        if not path.is_file():
            continue
        try:
            size = path.stat().st_size
        except OSError:
            continue
        files[path.relative_to(model_root).as_posix()] = size
    return files


def required_files_present(model_root: Path, required: list[str]) -> bool:
    return all((model_root / relative).is_file() and (model_root / relative).stat().st_size > 0 for relative in required)


def load_receipts(path: Path) -> dict:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return {}
    return data if isinstance(data, dict) else {}


def save_receipts(path: Path, receipts: dict) -> None:
    temp_path = path.with_name(f"{path.name}.tmp")
    try:
        temp_path.write_text(json.dumps(receipts, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(temp_path, path)
    except OSError:
        pass


def receipt_matches(entry: dict, model_root: Path) -> bool:
    files = entry.get("files") if isinstance(entry, dict) else None
    if not isinstance(files, dict) or not files:
        return False
    for relative, expected_size in files.items():
        path = model_root / str(relative)
        try:
            if not path.is_file() or path.stat().st_size != int(expected_size):
                return False
        except (OSError, TypeError, ValueError):
            return False
    return True


def remember_model(receipts: dict, profile: str, model_id: str, model_root: Path) -> None:
    receipts.setdefault("version", 1)
    receipts.setdefault("profiles", {}).setdefault(profile, {})[model_id] = {
        "files": model_files(model_root),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=PROFILES, default="quantized")
    parser.add_argument("--models-dir", default=None)
    parser.add_argument("--cache-dir", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--refresh", action="store_true", help="Ignore local receipts and check ModelScope again.")
    args = parser.parse_args()
    if args.models_dir:
        models_dir = Path(args.models_dir).expanduser().resolve()
    elif args.cache_dir:
        models_dir = Path(args.cache_dir).expanduser().resolve().parent / "models"
    else:
        models_dir = Path(os.environ.get("DIFFSYNTH_MODEL_BASE_PATH", DEFAULT_MODELS_DIR)).expanduser().resolve()
    models_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = models_dir / ".modelscope-cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MODELSCOPE_CACHE"] = str(cache_dir)
    os.environ["DIFFSYNTH_MODEL_BASE_PATH"] = str(models_dir)
    print(f"[download] python={sys.executable}", flush=True)
    print(f"[download] prefix={sys.prefix} base_prefix={sys.base_prefix}", flush=True)
    print(f"[download] profile={args.profile} models={models_dir}", flush=True)
    model_paths = [models_dir]
    model_gb = tree_bytes(model_paths) / 2**30
    total_gb = tree_bytes([HOME]) / 2**30
    print(f"[storage] model={model_gb:.1f}/{MODEL_BUDGET_GB:.0f}GB total={total_gb:.1f}/{TOTAL_BUDGET_GB:.0f}GB", flush=True)

    receipt_path = models_dir / RECEIPT_NAME
    receipts = load_receipts(receipt_path)
    profile_receipts = receipts.get("profiles", {}).get(args.profile, {})
    pending = []
    for model_id, patterns in PROFILES[args.profile].items():
        model_root = models_dir / model_id
        required = REQUIRED_FILES[args.profile].get(model_id, [])
        if (
            not args.refresh
            and receipt_matches(profile_receipts.get(model_id, {}), model_root)
            and (not required or required_files_present(model_root, required))
        ):
            print(f"[download] cache hit: {model_id}", flush=True)
            continue
        if not args.refresh and required and required_files_present(model_root, required):
            remember_model(receipts, args.profile, model_id, model_root)
            print(f"[download] local cache validated: {model_id}", flush=True)
            continue
        pending.append((model_id, patterns))

    save_receipts(receipt_path, receipts)
    if not pending:
        print("[download] all required H3 weights are ready (local cache)", flush=True)
        return
    if model_gb >= MODEL_BUDGET_GB or total_gb >= TOTAL_BUDGET_GB:
        raise SystemExit("Storage budget reached; model download is blocked to preserve generated-data space.")

    from modelscope import snapshot_download
    print(f"[download] modelscope={Path(sys.modules['modelscope'].__file__).resolve()}", flush=True)
    for model_id, patterns in pending:
        print(f"[download] checking {model_id}: {', '.join(patterns)}", flush=True)
        path = snapshot_download(
            model_id,
            revision="master",
            local_dir=str(models_dir / model_id),
            cache_dir=str(cache_dir),
            allow_file_pattern=patterns,
        )
        print(f"[download] ready: {path}", flush=True)
        remember_model(receipts, args.profile, model_id, models_dir / model_id)
        save_receipts(receipt_path, receipts)
        model_gb = tree_bytes(model_paths) / 2**30
        total_gb = tree_bytes([HOME]) / 2**30
        print(f"[storage] model={model_gb:.1f}/{MODEL_BUDGET_GB:.0f}GB total={total_gb:.1f}/{TOTAL_BUDGET_GB:.0f}GB", flush=True)
        if model_gb > MODEL_BUDGET_GB or total_gb > TOTAL_BUDGET_GB:
            raise SystemExit("Storage budget exceeded; stop and remove unneeded model files before continuing.")
    print("[download] all required H3 weights are ready", flush=True)

if __name__ == "__main__":
    main()
