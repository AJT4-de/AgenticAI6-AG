"""
tools/image_tool.py
──────────────────────────────────────────────────────────────────────
CrewAI-compatible async tool that generates a HIGH-QUALITY infographic
image via the Hugging Face Inference API (FLUX.1-dev by default)
and saves it to the output folder.

Includes:
  - Automatic prompt enhancement for better quality
  - 1024x1024 resolution with 30 inference steps
  - Retry logic via tenacity with exponential backoff
  - Graceful fallback to a styled placeholder image
──────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import datetime
import textwrap
import asyncio
from pathlib import Path
from typing import Any, Type

import requests
from crewai.tools import BaseTool
from PIL import Image, ImageDraw, ImageFont
from pydantic import BaseModel, Field
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type

from config.settings import HF_API_TOKEN, HF_IMAGE_MODEL


class ImageToolException(Exception):
    """Custom exception for retriable image generation errors."""
    pass


# ── Prompt enhancement ────────────────────────────────────────────────

def _enhance_prompt(raw_prompt: str) -> str:
    """Append quality-boosting keywords to a raw prompt."""
    prompt = raw_prompt.strip().strip("'\"")

    quality_tags = (
        ", high quality, sharp details, professional design, "
        "clean layout, modern flat design, vector style, "
        "white background, 4k resolution, detailed infographic"
    )
    negative_tags = (
        ", no text, no watermark, no blurry elements, no low resolution"
    )

    if "high quality" not in prompt.lower():
        prompt += quality_tags
    if "no watermark" not in prompt.lower():
        prompt += negative_tags

    return prompt


# ── Placeholder ──────────────────────────────────────────────────────

def _create_placeholder(
    filename: str,
    output_folder: str,
    prompt: str = "",
    message: str = "Image generation failed",
) -> str:
    """Create a styled 1024×1024 placeholder image."""
    img = Image.new("RGB", (1024, 1024), color=(240, 240, 245))
    draw = ImageDraw.Draw(img)

    # Border
    draw.rectangle([20, 20, 1003, 1003], outline=(100, 100, 180), width=3)
    # Title bar
    draw.rectangle([20, 20, 1003, 80], fill=(100, 100, 180))

    try:
        font_title = ImageFont.truetype("arial.ttf", 22)
        font_body = ImageFont.truetype("arial.ttf", 16)
        font_small = ImageFont.truetype("arial.ttf", 12)
    except OSError:
        font_title = ImageFont.load_default()
        font_body = font_title
        font_small = font_title

    draw.text((40, 35), "INFOGRAPHIC PLACEHOLDER", fill="white", font=font_title)
    draw.text(
        (40, 100), f"[ {message} ]", fill=(120, 120, 120), font=font_body
    )

    if prompt:
        wrapped = textwrap.fill(prompt[:300], width=70)
        draw.text((40, 140), f"Prompt:\n{wrapped}", fill=(80, 80, 80), font=font_small)

    out_dir = Path(output_folder)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / filename
    img.save(str(path), "PNG")
    print(f"  [Image Gen] Placeholder saved: {path}")
    return str(path)


# ── HF Inference API caller ─────────────────────────────────────────

@retry(
    wait=wait_exponential(multiplier=2, min=10, max=60),
    stop=stop_after_attempt(3),
    retry=retry_if_exception_type(ImageToolException),
    reraise=True
)
async def _generate_huggingface(prompt: str, filename: str, output_folder: str) -> str:
    """Call the Hugging Face Inference API with quality enhancements."""
    api_url = f"https://router.huggingface.co/hf-inference/models/{HF_IMAGE_MODEL}"
    headers = {
        "Authorization": f"Bearer {HF_API_TOKEN}",
        "Content-Type": "application/json",
    }

    enhanced = _enhance_prompt(prompt)
    payload = {
        "inputs": enhanced,
        "parameters": {
            "width": 1024,
            "height": 1024,
            "num_inference_steps": 30,
            "guidance_scale": 7.5,
        },
    }

    print(f"  [Image Gen] Requesting model={HF_IMAGE_MODEL}")
    loop = asyncio.get_running_loop()
    
    def _do_post():
        return requests.post(api_url, headers=headers, json=payload, timeout=120)

    try:
        resp = await loop.run_in_executor(None, _do_post)
    except Exception as e:
        raise ImageToolException(f"Connection error: {e}")

    if resp.status_code == 200:
        content_type = resp.headers.get("content-type", "")
        if "image" in content_type:
            import io
            
            def _save_image():
                img = Image.open(io.BytesIO(resp.content))
                out_dir = Path(output_folder)
                out_dir.mkdir(parents=True, exist_ok=True)
                path = out_dir / filename
                img.save(str(path), "PNG")
                return str(path)
                
            return await loop.run_in_executor(None, _save_image)
        else:
            raise ValueError(f"Unexpected content-type from API: {content_type}")
            
    elif resp.status_code == 503:
        raise ImageToolException("HTTP 503: Model loading")
    elif resp.status_code == 429:
        raise ImageToolException("HTTP 429: Rate limited")
    else:
        raise Exception(f"HTTP {resp.status_code}: {resp.text[:300]}")


# ── CrewAI BaseTool ───────────────────────────────────────────────────

class GenerateImageInput(BaseModel):
    """Strict input schema."""
    prompt: str = Field(..., description="A detailed text description of the image to generate.")
    output_folder: str = Field(..., description="Absolute path to the session output folder.")


class GenerateImageTool(BaseTool):
    name: str = "generate_image"
    description: str = "Generate a high-quality infographic image from a text prompt."
    args_schema: Type[BaseModel] = GenerateImageInput

    def _run(self, prompt: str, output_folder: str) -> str:
        raise NotImplementedError("This tool strictly uses async execution.")

    async def _arun(self, prompt: str, output_folder: str) -> str:
        prompt = prompt.strip()
        if not prompt:
            raise ValueError("Prompt cannot be empty.")

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"illustration_{timestamp}.png"

        if HF_API_TOKEN:
            try:
                return await _generate_huggingface(prompt, filename, output_folder)
            except Exception as e:
                print(f"  [Image Gen] Failed after retries: {e}")
                return _create_placeholder(filename, output_folder, prompt, f"Failed: {e}")
        else:
            print("  [Image Gen] No HF_API_TOKEN configured — using placeholder.")
            return _create_placeholder(filename, output_folder, prompt, "No HF API token configured")


# Module-level instance for easy import
generate_image_tool = GenerateImageTool()
