# RunPod SD1.5 + LoRA Serverless Worker

[![Runpod](https://api.runpod.io/badge/BlancheVall/runpod-sd15-lora-worker)](https://console.runpod.io/hub/BlancheVall/runpod-sd15-lora-worker)

This repo is a RunPod Serverless worker for Stable Diffusion 1.5 plus one pixel-art LoRA. It accepts a prompt and returns a PNG image as base64.

## Files

- `handler.py` - RunPod serverless handler.
- `Dockerfile` - GPU runtime image.
- `requirements.txt` - Python dependencies.
- `test_input.json` - Example request body.

## Expected API Input

```json
{
  "input": {
    "prompt": "(masterpiece, top quality, best quality), pixel, pixel art, 1girl, full body, pink long-haired female mage, RPG pixel sprite",
    "negative_prompt": "(worst quality, low quality:2), blurry, bad anatomy, text, watermark",
    "width": 512,
    "height": 768,
    "num_inference_steps": 30,
    "guidance_scale": 7,
    "clip_skip": 2,
    "lora_weight": 0.5,
    "seed": -1,
    "remove_background": false
  }
}
```

## Output

```json
{
  "images": [
    {
      "image": "base64_png_here",
      "seed": 123,
      "width": 512,
      "height": 768
    }
  ],
  "seed": 123,
  "elapsed_seconds": 4.2
}
```

Your website code already accepts this shape through `output.images[0].image`.

## Model Paths

Do not commit SD1.5 or LoRA files into git. They are too large.

For local testing on your PC, set:

```powershell
$env:SD_MODEL_PATH="D:\Download\v1-5\sd1-5.ckpt"
$env:LORA_PATH="D:\Download\pixel_f2\pixel_f2.safetensors"
$env:LORA_WEIGHT="0.5"
```

I could not find these two folders from the current shell, so check whether your actual paths are `D:\Downloads\...` or another drive.

For RunPod production, put the model files on a Network Volume or bake them into a private image, then set:

```env
SD_MODEL_PATH=/workspace/models/v1-5/sd1-5.ckpt
LORA_PATH=/workspace/loras/pixel_f2.safetensors
LORA_WEIGHT=0.5
WIDTH=512
HEIGHT=768
STEPS=30
GUIDANCE_SCALE=7
CLIP_SKIP=2
SCHEDULER=dpm
REMOVE_BACKGROUND=false
```

`SD_MODEL_PATH` can be either:

- A Diffusers folder with `model_index.json`
- A folder containing one `.safetensors` or `.ckpt`
- A direct `.safetensors` or `.ckpt` file

`LORA_PATH` can be either:

- A direct LoRA `.safetensors` file
- A folder containing one LoRA `.safetensors`

## RunPod Hub Setup

1. Push this repo to GitHub.
2. In RunPod, choose `Add your repo`.
3. Select this repo.
4. Use the included `Dockerfile`.
5. Attach the Network Volume that contains your SD1.5 and LoRA files.
6. Add the environment variables above.
7. Deploy as a Serverless Endpoint.
8. Copy the endpoint id into your website as `RUNPOD_ENDPOINT_ID`.

## Local Docker Test

```powershell
docker build -t sd15-lora-worker .
docker run --gpus all --rm `
  -e SD_MODEL_PATH=/models/v1-5/sd1-5.ckpt `
  -e LORA_PATH=/loras/pixel_f2.safetensors `
  -v D:\Download\v1-5:/models/v1-5 `
  -v D:\Download\pixel_f2:/loras/pixel_f2 `
  sd15-lora-worker
```

Then test through RunPod once deployed with `/runsync`.

## Notes

- SD1.5 does not naturally output real transparency. `remove_background=true` enables a simple corner flood-fill transparency pass, but for production quality it is better to train/generate on a flat background and remove it intentionally.
- The worker loads the model once and reuses it for later jobs in the same warm container.
