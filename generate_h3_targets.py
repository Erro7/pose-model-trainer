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
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from PIL import Image
from diffsynth.configs import MODEL_CONFIGS
from diffsynth.pipelines.minimax_h3_image import MiniMaxH3ImagePipeline, ModelConfig


def valid_image(path: Path) -> bool:
    try:
        with Image.open(path) as image:
            image.verify()
        return True
    except (OSError, ValueError, Image.DecompressionBombError):
        return False


def pending_manifest_items(job: Path, manifest: list, targets: Path) -> list[dict]:
    if not isinstance(manifest, list) or not manifest:
        raise SystemExit("[generate] generation.json must contain a non-empty list")
    pending = []
    seen_outputs = set()
    errors = []
    for index, item in enumerate(manifest, 1):
        if not isinstance(item, dict):
            errors.append(f"item {index}: not an object")
            continue
        output_name = item.get("output")
        if (
            not isinstance(output_name, str)
            or not output_name
            or Path(output_name).name != output_name
            or Path(output_name).suffix.lower() != ".png"
        ):
            errors.append(f"item {index}: invalid output name {output_name!r}")
            continue
        if output_name in seen_outputs:
            errors.append(f"item {index}: duplicate output {output_name}")
            continue
        seen_outputs.add(output_name)
        prompt = item.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            errors.append(f"item {index}: missing prompt")
            continue
        refs = item.get("edit_image")
        if isinstance(refs, str):
            refs = [refs]
            item["edit_image"] = refs
        if not isinstance(refs, list) or not refs:
            errors.append(f"item {index}: missing edit_image references")
            continue
        output = targets / output_name
        if output.is_file() and valid_image(output):
            continue
        ref_paths = []
        for ref in refs:
            if not isinstance(ref, str) or not ref.strip():
                errors.append(f"item {index}: invalid edit_image reference")
                break
            ref_path = (job / ref).resolve()
            if job.resolve() not in ref_path.parents:
                errors.append(f"item {index}: reference escapes job directory: {ref}")
                break
            ref_paths.append(ref_path)
        else:
            missing = [str(path) for path in ref_paths if not path.is_file()]
            if missing:
                errors.append(f"item {index}: missing reference(s): {', '.join(missing)}")
                continue
            invalid = [str(path) for path in ref_paths if not valid_image(path)]
            if invalid:
                errors.append(f"item {index}: invalid reference image(s): {', '.join(invalid)}")
                continue
            pending.append(item)
    if errors:
        raise SystemExit("[generate] invalid generation manifest:\n" + "\n".join(errors))
    return pending


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
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--height", type=int, default=768)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    job = Path(args.job).resolve()
    try:
        manifest = json.loads((job / "generation.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"[generate] cannot read generation.json: {exc}") from exc
    targets = job / "targets"
    targets.mkdir(exist_ok=True)
    pending = pending_manifest_items(job, manifest, targets)
    if not pending:
        print("[generate] all target images already exist", flush=True)
        return
    vram = {
        "offload_dtype": "disk", "offload_device": "disk",
        "onload_dtype": "disk", "onload_device": "disk",
        "preparing_dtype": torch.bfloat16, "preparing_device": "cuda",
        "computation_dtype": torch.bfloat16, "computation_device": "cuda",
    }
    total_gib = torch.cuda.mem_get_info("cuda")[1] / 2**30
    vram_limit = max(total_gib - 2.5, 1.0)
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
        vram_limit=vram_limit,
    )
    torch.cuda.empty_cache()
    print(f"[generate] vram_limit={vram_limit:.1f}GiB", flush=True)
    for index, item in enumerate(manifest, 1):
        output = targets / item["output"]
        if output.is_file():
            if valid_image(output):
                print(f"[generate] skip existing {output.name}", flush=True)
                continue
            print(f"[generate] replacing invalid image {output.name}", flush=True)
            output.unlink(missing_ok=True)
        ref_paths = [job / path for path in item.get("edit_image", [])]
        refs = []
        temp = None
        print(f"[generate] {index}/{len(manifest)} {item.get('pose_name', '')}", flush=True)
        torch.cuda.empty_cache()
        try:
            for path in ref_paths:
                with Image.open(path) as source:
                    refs.append(source.convert("RGB"))
            image = pipe(
                prompt=item["prompt"], edit_image=refs,
                width=args.width, height=args.height, seed=args.seed + index - 1,
            )
            temp = output.with_name(f".{output.name}.tmp")
            image.save(temp, format="PNG")
            os.replace(temp, output)
        finally:
            for ref in refs:
                ref.close()
            if temp is not None:
                temp.unlink(missing_ok=True)
    print("[generate] batch complete", flush=True)


if __name__ == "__main__":
    main()
