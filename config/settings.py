"""
config/settings.py
Centralised configuration loaded from .env and sensible defaults.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# ── Load .env from project root ──────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_PROJECT_ROOT / ".env")

# ── LLM Settings (Ollama — local Llama 3) ────────────────────────────────────
OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "llama3")

# ── Image-Generation Settings (Hugging Face) ─────────────────────────────────
HF_API_TOKEN: str = os.getenv("HF_API_TOKEN", "")
HF_IMAGE_MODEL: str = os.getenv(
    "HF_IMAGE_MODEL", "black-forest-labs/FLUX.1-schnell"
)

# ── Pipeline Defaults ────────────────────────────────────────────────────────
DEFAULT_OUTPUT_LANGUAGE: str = os.getenv("DEFAULT_OUTPUT_LANGUAGE", "English")
DEFAULT_TARGET_GROUP: str = os.getenv("DEFAULT_TARGET_GROUP", "University student")

# ── Output directory ─────────────────────────────────────────────────────────
OUTPUT_DIR: Path = _PROJECT_ROOT / "output"
OUTPUT_DIR.mkdir(exist_ok=True)


# ── Readability grade-level targets per audience ─────────────────────────────
TARGET_GRADE_LEVELS: dict[str, float] = {
    "5-year-old child": 1.0,
    "10-year-old child": 5.0,
    "High-school student": 10.0,
    "University student": 13.0,
    "Graduate researcher": 16.0,
    "General public": 8.0,
}


def get_target_grade(target_group: str) -> float:
    """Return the Flesch-Kincaid grade level for a target audience string."""
    # Exact match first, then case-insensitive partial match
    if target_group in TARGET_GRADE_LEVELS:
        return TARGET_GRADE_LEVELS[target_group]
    lower = target_group.lower()
    for key, value in TARGET_GRADE_LEVELS.items():
        if key.lower() in lower or lower in key.lower():
            return value
    # Fallback
    return 8.0
