"""
tools/image_tool.py
──────────────────────────────────────────────────────────────────────
CrewAI-compatible tool that generates a HIGH-QUALITY infographic
image via the Hugging Face Inference API (FLUX.1-dev by default)
and saves it to the output folder.

Includes:
  - Automatic prompt enhancement for better quality
  - 1024x1024 resolution with 30 inference steps
  - Retry logic (max 3 attempts) with longer waits for large models
  - Graceful fallback to a styled placeholder image

Uses BaseTool (not @tool decorator) so we can accept flexible input
— Llama 3 sometimes passes prompt as a dict instead of a string.
──────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import datetime
import textwrap
import time
from typing import Any, Type

import requests
from crewai.tools import BaseTool
from PIL import Image, ImageDraw, ImageFont
from pydantic import BaseModel, Field

from config.settings import HF_API_TOKEN, HF_IMAGE_MODEL, OUTPUT_DIR

MAX_RETRIES = 3
RETRY_DELAY_SECONDS = 10  # FLUX models need more warm-up time


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

    # Decorative section bars
    for i in range(5):
        y = 350 + i * 120
        draw.rounded_rectangle(
            [60, y, 960, y + 90],
            radius=10,
            fill=(220, 225, 235),
            outline=(180, 185, 200),
        )
        draw.text(
            (80, y + 30), f"Section {i + 1}", fill=(100, 100, 120), font=font_body
        )

    path = OUTPUT_DIR / filename
    img.save(str(path), "PNG")
    print(f"  [Image Gen] Placeholder saved: {path}")
    return str(path)


# ── HF Inference API caller ─────────────────────────────────────────

def _generate_huggingface(prompt: str, filename: str) -> str:
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

    last_error = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            print(f"  [Image Gen] Attempt {attempt}/{MAX_RETRIES}  model={HF_IMAGE_MODEL}")
            print(f"  [Image Gen] Prompt: {enhanced[:120]}...")
            resp = requests.post(
                api_url, headers=headers, json=payload, timeout=120
            )

            if resp.status_code == 200:
                content_type = resp.headers.get("content-type", "")
                if "image" in content_type:
                    import io
                    img = Image.open(io.BytesIO(resp.content))
                    path = OUTPUT_DIR / filename
                    img.save(str(path), "PNG")
                    print(
                        f"  [Image Gen] Success — {path} "
                        f"({path.stat().st_size:,} bytes, {img.size[0]}x{img.size[1]})"
                    )
                    return str(path)
                else:
                    last_error = f"Unexpected content-type: {content_type}"

            elif resp.status_code == 503:
                # Model loading — parse estimated_time if available
                try:
                    body = resp.json()
                    est = body.get("estimated_time", RETRY_DELAY_SECONDS * attempt)
                except Exception:
                    est = RETRY_DELAY_SECONDS * attempt
                wait = max(est, RETRY_DELAY_SECONDS)
                last_error = f"HTTP 503: Model loading"
                print(f"  [Image Gen] Model loading, waiting {wait:.0f}s...")
                time.sleep(wait)
                continue

            elif resp.status_code == 429:
                wait = RETRY_DELAY_SECONDS * attempt
                last_error = "HTTP 429: Rate limited"
                print(f"  [Image Gen] Rate limited, waiting {wait}s...")
                time.sleep(wait)
                continue

            else:
                last_error = f"HTTP {resp.status_code}: {resp.text[:300]}"

        except requests.exceptions.Timeout:
            last_error = "Request timed out"
        except requests.exceptions.ConnectionError as e:
            last_error = f"Connection error: {e}"
        except Exception as e:
            last_error = f"Unexpected error: {e}"

        if attempt < MAX_RETRIES:
            print(
                f"  [Image Gen] Failed: {last_error}. "
                f"Retrying in {RETRY_DELAY_SECONDS}s..."
            )
            time.sleep(RETRY_DELAY_SECONDS)

    # All retries exhausted
    print(f"  [Image Gen] All {MAX_RETRIES} attempts failed: {last_error}")
    print("  [Image Gen] Falling back to placeholder image.")
    return _create_placeholder(filename, prompt, f"Failed: {last_error}")


# ── CrewAI BaseTool ───────────────────────────────────────────────────

class GenerateImageInput(BaseModel):
    """Flexible input schema — accepts prompt as str or dict."""
    prompt: Any = Field(
        ..., description="A detailed text description of the image to generate."
    )


class GenerateImageTool(BaseTool):
    name: str = "generate_image"
    description: str = (
        "Generate a high-quality infographic image from a text prompt. "
        "Pass the prompt argument as a plain string describing the image. "
        "Returns the absolute file path of the saved PNG."
    )
    args_schema: Type[BaseModel] = GenerateImageInput

    def _run(self, prompt: Any) -> str:
        # Coerce dict → string (Llama 3 quirk)
        if isinstance(prompt, dict):
            prompt = prompt.get("description", "") or str(prompt)
        prompt = str(prompt).strip()
        if not prompt:
            prompt = "abstract colorful infographic illustration"

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"illustration_{timestamp}.png"

        if HF_API_TOKEN:
            return _generate_huggingface(prompt, filename)
        else:
            print("  [Image Gen] No HF_API_TOKEN configured — using placeholder.")
            return _create_placeholder(filename, prompt, "No HF API token configured")


# Module-level instance for easy import
generate_image_tool = GenerateImageTool()
