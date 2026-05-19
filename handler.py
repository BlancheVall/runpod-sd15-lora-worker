import base64
import io
import os
import random
import time
from pathlib import Path
from threading import Lock

import runpod
from PIL import Image


DEFAULT_NEGATIVE_PROMPT = (
    "low quality, worst quality, blurry, anti-aliased, smooth shading, realistic, "
    "3d render, background, shadow, glow, text, watermark, multiple characters, sprite sheet"
)

PIPELINE = None
PIPELINE_LOCK = Lock()


def env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)

    if value is None:
        return default

    return value.lower() in {"1", "true", "yes", "on"}


def find_first_model_file(path: Path) -> Path | None:
    if path.is_file():
        return path

    for pattern in ("*.safetensors", "*.ckpt", "*.bin"):
        matches = sorted(path.glob(pattern))

        if matches:
            return matches[0]

    return None


def resolve_existing_file(configured_path: Path, fallback_dir: Path, fallback_names: tuple[str, ...]) -> Path:
    if configured_path.exists():
        return configured_path

    for name in fallback_names:
        candidate = fallback_dir / name

        if candidate.exists():
            return candidate

    discovered_file = find_first_model_file(fallback_dir)

    if discovered_file:
        return discovered_file

    return configured_path


def list_directory(path: Path) -> list[str]:
    if not path.exists() or not path.is_dir():
        return []

    return sorted(item.name for item in path.iterdir())


def load_pipeline():
    global PIPELINE

    if PIPELINE is not None:
        return PIPELINE

    with PIPELINE_LOCK:
        if PIPELINE is not None:
            return PIPELINE

        import torch
        from diffusers import DPMSolverMultistepScheduler, EulerAncestralDiscreteScheduler, StableDiffusionPipeline

        model_path = resolve_existing_file(
            Path(os.getenv("SD_MODEL_PATH", "/runpod-volume/models/v1-5/sd1-5.safetensors")),
            Path("/runpod-volume/models/v1-5"),
            ("sd1-5.safetensors", "sd1-5.ckpt", "model.safetensors", "model.ckpt"),
        )
        lora_path = resolve_existing_file(
            Path(os.getenv("LORA_PATH", "/runpod-volume/loras/pixel_f2.safetensors")),
            Path("/runpod-volume/loras"),
            ("pixel_f2.safetensors", "pixel_f2.ckpt"),
        )
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        if not model_path.exists():
            raise FileNotFoundError(f"SD model path does not exist: {model_path}")

        model_file = find_first_model_file(model_path)

        if model_file and model_file.suffix.lower() in {".safetensors", ".ckpt"}:
            pipe = StableDiffusionPipeline.from_single_file(
                str(model_file),
                torch_dtype=dtype,
                safety_checker=None,
                requires_safety_checker=False,
                local_files_only=True,
            )
        else:
            pipe = StableDiffusionPipeline.from_pretrained(
                str(model_path),
                torch_dtype=dtype,
                safety_checker=None,
                requires_safety_checker=False,
                local_files_only=True,
            )

        scheduler = os.getenv("SCHEDULER", "dpm").lower()
        if scheduler in {"euler_a", "euler-ancestral", "euler_ancestral"}:
            pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
        else:
            pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

        pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")

        if env_bool("ENABLE_ATTENTION_SLICING", False):
            pipe.enable_attention_slicing()

        if lora_path.exists():
            lora_file = find_first_model_file(lora_path)
            if lora_file:
                pipe.load_lora_weights(str(lora_file.parent), weight_name=lora_file.name)
            else:
                pipe.load_lora_weights(str(lora_path))
        else:
            print(f"Lora path does not exist, continuing without LoRA: {lora_path}")

        PIPELINE = pipe
        return PIPELINE


def image_to_base64(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def remove_corner_background(image: Image.Image, tolerance: int = 20) -> Image.Image:
    rgba = image.convert("RGBA")
    pixels = rgba.load()
    width, height = rgba.size
    corner_colors = [
        pixels[0, 0][:3],
        pixels[width - 1, 0][:3],
        pixels[0, height - 1][:3],
        pixels[width - 1, height - 1][:3],
    ]

    def similar_to_corner(pixel):
        return any(
            all(abs(pixel[channel] - corner[channel]) <= tolerance for channel in range(3))
            for corner in corner_colors
        )

    visited = set()
    stack = [(0, 0), (width - 1, 0), (0, height - 1), (width - 1, height - 1)]

    while stack:
        x, y = stack.pop()

        if x < 0 or y < 0 or x >= width or y >= height or (x, y) in visited:
            continue

        visited.add((x, y))
        pixel = pixels[x, y]

        if not similar_to_corner(pixel):
            continue

        pixels[x, y] = (pixel[0], pixel[1], pixel[2], 0)
        stack.extend(((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)))

    return rgba


def get_seed(input_data: dict) -> int:
    seed = int(input_data.get("seed", -1))

    if seed < 0:
        return random.randint(0, 2**31 - 1)

    return seed


def handler(job):
    started_at = time.time()
    input_data = job.get("input", {}) or {}

    if input_data.get("health_check") or not input_data:
        return {
            "ok": True,
            "message": "Worker is reachable. Model loading is tested when a real generation request runs.",
        }

    if input_data.get("check_paths"):
        configured_model_path = Path(os.getenv("SD_MODEL_PATH", "/runpod-volume/models/v1-5/sd1-5.safetensors"))
        configured_lora_path = Path(os.getenv("LORA_PATH", "/runpod-volume/loras/pixel_f2.safetensors"))
        model_path = resolve_existing_file(
            configured_model_path,
            Path("/runpod-volume/models/v1-5"),
            ("sd1-5.safetensors", "sd1-5.ckpt", "model.safetensors", "model.ckpt"),
        )
        lora_path = resolve_existing_file(
            configured_lora_path,
            Path("/runpod-volume/loras"),
            ("pixel_f2.safetensors", "pixel_f2.ckpt"),
        )

        return {
            "configured_sd_model_path": str(configured_model_path),
            "sd_model_path": str(model_path),
            "sd_model_exists": model_path.exists(),
            "sd_model_size_bytes": model_path.stat().st_size if model_path.exists() else None,
            "configured_lora_path": str(configured_lora_path),
            "lora_path": str(lora_path),
            "lora_exists": lora_path.exists(),
            "lora_size_bytes": lora_path.stat().st_size if lora_path.exists() else None,
            "runpod_volume_root_exists": Path("/runpod-volume").exists(),
            "model_dir_files": list_directory(Path("/runpod-volume/models/v1-5")),
            "lora_dir_files": list_directory(Path("/runpod-volume/loras")),
            "workspace_root_exists": Path("/workspace").exists(),
        }

    prompt = str(input_data.get("prompt", "")).strip()

    if not prompt:
        return {"error": "prompt is required"}

    pipe = load_pipeline()
    import torch

    seed = get_seed(input_data)
    generator = torch.Generator(device=pipe.device).manual_seed(seed)
    lora_weight = float(input_data.get("lora_weight", os.getenv("LORA_WEIGHT", "0.8")))

    width = int(input_data.get("width", os.getenv("WIDTH", "512")))
    height = int(input_data.get("height", os.getenv("HEIGHT", "512")))
    steps = int(input_data.get("num_inference_steps", input_data.get("steps", os.getenv("STEPS", "28"))))
    guidance_scale = float(input_data.get("guidance_scale", os.getenv("GUIDANCE_SCALE", "7")))
    negative_prompt = str(input_data.get("negative_prompt", os.getenv("NEGATIVE_PROMPT", DEFAULT_NEGATIVE_PROMPT)))
    clip_skip = int(input_data.get("clip_skip", os.getenv("CLIP_SKIP", "2")))

    with torch.inference_mode():
        result = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            generator=generator,
            clip_skip=clip_skip,
            cross_attention_kwargs={"scale": lora_weight},
        )

    image = result.images[0]

    if bool(input_data.get("remove_background", env_bool("REMOVE_BACKGROUND", False))):
        tolerance = int(input_data.get("background_tolerance", os.getenv("BACKGROUND_TOLERANCE", "20")))
        image = remove_corner_background(image, tolerance=tolerance)

    return {
        "images": [
            {
                "image": image_to_base64(image),
                "seed": seed,
                "width": image.width,
                "height": image.height,
            }
        ],
        "seed": seed,
        "elapsed_seconds": round(time.time() - started_at, 3),
    }


runpod.serverless.start({"handler": handler})
