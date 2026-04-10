"""
tools/readability_tool.py
──────────────────────────────────────────────────────────────────────
Utility functions to compute readability metrics using textstat.
Not a CrewAI tool itself — called directly by the orchestrator to
compute the Accessibility Score before passing it to The Publisher.
──────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import textstat


def flesch_kincaid_grade(text: str) -> float:
    """Return the Flesch-Kincaid Grade Level for *text*."""
    return textstat.flesch_kincaid_grade(text)


def flesch_reading_ease(text: str) -> float:
    """Return the Flesch Reading Ease score (0-100+)."""
    return textstat.flesch_reading_ease(text)


def full_readability_report(text: str) -> dict:
    """Return a dict of several readability metrics."""
    return {
        "flesch_kincaid_grade": textstat.flesch_kincaid_grade(text),
        "flesch_reading_ease": textstat.flesch_reading_ease(text),
        "gunning_fog": textstat.gunning_fog(text),
        "smog_index": textstat.smog_index(text),
        "coleman_liau_index": textstat.coleman_liau_index(text),
        "automated_readability_index": textstat.automated_readability_index(text),
        "dale_chall_readability_score": textstat.dale_chall_readability_score(text),
        "text_standard": textstat.text_standard(text, float_output=False),
    }
