#!/usr/bin/env python3
"""Generate supervised H3 targets from identity, outfit and pose references."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

HOME = Path(os.environ.get("TRAINER_HOME", "/mnt/workspace/pose-model-trainer-data")).expanduser().resolve()
MODELS_DIR = Path(os.environ.get("DIFFSYNTH_MODEL_BASE_PATH", HOME / "models")).expanduser().resolve()
CACHE_DIR = MODELS_DIR / ".modelscope-cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ["DIFFSYNTH_MODEL_BASE_PATH"] = str(MODELS_DIR)
os.environ["MODELSCOPE_CACHE"] = str(CACHE_DIR)

import torch
from PIL import Image
from diffsynth.configs import MODEL_CONFIGS
from diffsynth.pipelines.minimax_h3_image import MiniMaxH3ImagePipeline, ModelConfig


def register(model_id: str, pattern: str, vram: dict) -> ModelConfig:
    package = ModelConfig(model_id=model_id, origin_file_pattern=pattern, **vram)
    package.download_if_necessary()
    sidecar = ModelConfig(model_id=model_id, origin_file_pattern=pattern.replace(".safetensors", ".json"))
    sidecar.download_if_necessary()
    with open(sidecar.path, encoding="utf-8") as handle:
        entry = json.load(handle)
    if not any(item.get("model_hash") == entry["model_hash"] for item in MODEL_CONFIGS):
        MODEL_CONFIGS.append(entry)
    return package


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", required=True)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    job = Path(args.job).resolve()
    manifest = json.loads((job / "generation.json").read_text(encoding="utf-8"))
    vram = {
        "offload_dtype": "disk", "offload_device": "disk",
        "onload_dtype": "disk", "onload_device": "disk",
        "preparing_dtype": torch.bfloat16, "preparing_device": "cuda",
        "computation_dtype": torch.bfloat16, "computation_device": "cuda",
    }
    pipe = MiniMaxH3ImagePipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device="cuda",
        model_configs=[
            register("DiffSynth-Studio/MiniMax-H3-Image-EntroPack", "models/dit_7bit.safetensors", vram),
            register("DiffSynth-Studio/EntroPackPreQuants", "models/MiniMax-H3/text_encoder_4bpp.safetensors", vram),
            ModelConfig(model_id="MiniMax/MiniMax-H3-Image", origin_file_pattern="pdd_fused_heads.safetensors", **vram),
            ModelConfig(model_id="DiffSynth-Studio/MiniMax-H3-Image-EntroPack", origin_file_pattern="models/vae_int8.safetensors", **vram),
        ],
        processor_config=ModelConfig(model_id="MiniMax/MiniMax-H3-Image", origin_file_pattern="processor/"),
        vram_limit=max(torch.cuda.mem_get_info("cuda")[1] / 2**30 - 0.5, 1),
    )
    targets = job / "targets"
    targets.mkdir(exist_ok=True)
    for index, item in enumerate(manifest, 1):
        output = targets / item["output"]
        if output.is_file():
            print(f"[generate] skip existing {output.name}", flush=True)
            continue
        refs = [Image.open(job / path).convert("RGB") for path in item["edit_image"]]
        print(f"[generate] {index}/{len(manifest)} {item['pose_name']}", flush=True)
        image = pipe(prompt=item["prompt"], edit_image=refs, width=args.width, height=args.height, seed=args.seed + index - 1)
        image.save(output)
    print("[generate] batch complete", flush=True)


if __name__ == "__main__":
    main()
