"""
main.py  —  Multi-Modal Agentic Pipeline Orchestrator
══════════════════════════════════════════════════════════════════════
Lead Orchestrator for the CrewAI system.
Transforms raw input into a high-quality, target-group-specific
PDF report by coordinating four specialised agents in sequence:

    1. The Interpreter  → Strategic Blueprint
    2. The Writer       → Educational Text
    3. The Illustrator  → Image Generation
    4. The Publisher    → PDF Assembly

Usage:
    python main.py
    python main.py --input "Your raw text here" --language French --audience "10-year-old child"
══════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import argparse
import json
import re
import sys

import requests
from crewai import Crew, LLM, Process

from config.settings import (
    DEFAULT_OUTPUT_LANGUAGE,
    DEFAULT_TARGET_GROUP,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
)
from agents.interpreter_agent import create_interpreter_agent, create_interpreter_task
from agents.writer_agent import create_writer_agent, create_writer_task
from agents.illustrator_agent import create_illustrator_agent, create_illustrator_task
from tools.pdf_tool import build_pdf
from tools.readability_tool import flesch_kincaid_grade, flesch_reading_ease


# ── Helpers: strip Llama / CrewAI artefacts ──────────────────────────────────
_REACT_NOISE = re.compile(
    r"^\s*"
    r"(Thought:\s*.+?\n|Final Answer:\s*|I now can give a great answer\s*)"
    r"+",
    re.IGNORECASE | re.DOTALL,
)


def _strip_react_prefix(text: str) -> str:
    """Remove common CrewAI ReAct / Llama preamble from agent output."""
    text = text.strip()
    # Remove leading "Thought: ... Final Answer:" block
    text = _REACT_NOISE.sub("", text).strip()
    # Also remove if it appears on a line by itself
    for noise in ("I now can give a great answer",):
        if text.lower().startswith(noise.lower()):
            text = text[len(noise):].strip()
    return text


def _extract_json(text: str) -> dict | None:
    """Try hard to extract a JSON object from agent output.

    Strategy:
    1. Check for backtick-fenced JSON blocks.
    2. Find the outermost { … } and try to parse it.
    3. Return None on failure.
    """
    text = _strip_react_prefix(text)

    # Strategy 1: backtick fences
    if "```" in text:
        parts = text.split("```")
        for part in parts[1::2]:            # odd-indexed pieces are inside fences
            inner = part.strip()
            if inner.startswith("json"):
                inner = inner[4:].strip()
            try:
                return json.loads(inner)
            except json.JSONDecodeError:
                continue

    # Strategy 2: find outermost braces
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            pass

    return None


def _extract_educational_text(text: str) -> str:
    """Clean the Writer's output, stripping ReAct noise."""
    text = _strip_react_prefix(text)
    # Remove any trailing meta-notes the model may add
    for marker in ("Note:", "note:"):
        idx = text.rfind(marker)
        if idx > len(text) * 0.8:  # only if in the last 20 %
            text = text[:idx].strip()
    return text


# ── CLI ───────────────────────────────────────────────────────────────────────
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Multi-Modal Agentic PDF Report Pipeline"
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Raw input text (or omit to enter interactively).",
    )
    parser.add_argument(
        "--language",
        type=str,
        default=DEFAULT_OUTPUT_LANGUAGE,
        help=f"Output language (default: {DEFAULT_OUTPUT_LANGUAGE}).",
    )
    parser.add_argument(
        "--audience",
        type=str,
        default=DEFAULT_TARGET_GROUP,
        help=f"Target audience profile (default: {DEFAULT_TARGET_GROUP}).",
    )
    return parser.parse_args()


# ── LLM factory ──────────────────────────────────────────────────────────────
def _get_llm():
    """Return a CrewAI LLM instance connected to the local
    Ollama server running Llama 3."""
    return LLM(
        model=f"ollama/{OLLAMA_MODEL}",
        base_url=OLLAMA_BASE_URL,
        temperature=0.4,
    )


# ── Direct Ollama writer (bypasses CrewAI ReAct loop) ────────────────────────
def _call_ollama_writer(raw_input: str, blueprint_json: str) -> str:
    """Call Ollama directly to generate the educational article.

    CrewAI wraps every task in a ReAct-style prompt which causes Llama 3
    to short-circuit with 'Thought: I now can give a great answer' instead
    of producing the actual article.  Calling the REST API directly avoids
    this entirely.
    """
    prompt = (
        "You are an award-winning educational writer fluent in many languages. "
        "You specialise in making complex topics accessible without sacrificing accuracy.\n\n"
        "Write a COMPLETE, DETAILED educational article about the topic below.\n\n"
        f"**Topic**: {raw_input}\n\n"
        f"**Blueprint** (use this to guide tone, audience, and structure):\n"
        f"{blueprint_json}\n\n"
        "### Writing Instructions\n"
        "1. Write a COMPLETE article of at least 800 words.\n"
        "2. Structure your article with these 6 sections (use ## for each):\n"
        "   - ## Introduction\n"
        "   - ## Background & Context\n"
        "   - ## Key Analysis\n"
        "   - ## Current Situation & Developments\n"
        "   - ## Impact & Implications\n"
        "   - ## Conclusion\n"
        "3. Use Markdown formatting throughout:\n"
        "   - **bold** for key terms and important concepts\n"
        "   - *italic* for emphasis and definitions\n"
        "   - > blockquotes for notable quotes or key takeaways\n"
        "   - Numbered lists (1. 2. 3.) for sequential steps or rankings\n"
        "   - Bullet lists (- item) for non-sequential points\n"
        "4. Match the tone, register, and grade level from the Blueprint.\n"
        "5. Define key terms from the glossary naturally in the text.\n"
        "6. End with a bullet-point summary of the 5 most important takeaways:\n"
        "   ### Key Takeaways\n"
        "   - First takeaway\n"
        "   - Second takeaway\n"
        "   - (etc.)\n\n"
        "IMPORTANT: Output ONLY the article text. "
        "Do NOT include any preamble, commentary, or meta-text. "
        "Begin writing the article NOW.\n\n"
        "## Introduction\n"
    )

    resp = requests.post(
        f"{OLLAMA_BASE_URL}/api/generate",
        json={
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.4, "num_predict": 4096},
        },
        timeout=300,
    )
    resp.raise_for_status()
    body = resp.json().get("response", "").strip()
    return "## Introduction\n" + body


# ── Main pipeline ────────────────────────────────────────────────────────────
def run_pipeline(
    raw_input: str,
    output_language: str = DEFAULT_OUTPUT_LANGUAGE,
    target_group: str = DEFAULT_TARGET_GROUP,
) -> dict:
    """Execute the full four-agent pipeline and return result paths."""

    llm = _get_llm()

    # Accumulate results -- partial output even if a stage fails
    blueprint = {}
    educational_text = ""
    fk_grade = 0.0
    fre_score = 0.0
    image_path = ""
    pdf_path = ""
    errors: list[str] = []

    # ────────────────────────────────────────────────────────────────
    # Stage 1 — The Interpreter
    # ────────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  STAGE 1 : The Interpreter (Input Analysis)")
    print("=" * 60)

    try:
        interpreter = create_interpreter_agent(llm)
        interpreter_task = create_interpreter_task(
            agent=interpreter,
            raw_input=raw_input,
            target_group=target_group,
            output_language=output_language,
        )

        interpreter_crew = Crew(
            agents=[interpreter],
            tasks=[interpreter_task],
            process=Process.sequential,
            verbose=True,
        )
        blueprint_raw = interpreter_crew.kickoff()
        blueprint_text = str(blueprint_raw).strip()

        # Robust JSON extraction (handles ReAct prefixes & missing fences)
        parsed = _extract_json(blueprint_text)
        if parsed and isinstance(parsed, dict):
            blueprint = parsed
        else:
            blueprint = {"raw_blueprint": _strip_react_prefix(blueprint_text)}

        print(f"\n✔  Blueprint received ({len(blueprint)} keys)")

    except Exception as e:
        err_msg = f"Stage 1 (Interpreter) failed: {e}"
        errors.append(err_msg)
        print(f"\n⚠  {err_msg}")
        print("   Continuing with empty blueprint...")
        blueprint = {"error": str(e), "raw_input": raw_input}

    # ────────────────────────────────────────────────────────────────
    # Stage 2 — The Writer
    # ────────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  STAGE 2 : The Writer (Content Generation)")
    print("=" * 60)

    try:
        blueprint_json_str = json.dumps(blueprint, ensure_ascii=False, indent=2)

        # Call Ollama directly — bypasses CrewAI ReAct loop which
        # consistently makes Llama 3 output meta-commentary
        # ("I now can give a great answer") instead of content.
        MAX_WRITER_ATTEMPTS = 3
        for attempt in range(1, MAX_WRITER_ATTEMPTS + 1):
            print(f"  [Writer] Attempt {attempt}/{MAX_WRITER_ATTEMPTS} "
                  f"(direct Ollama API)...")
            raw_writer_output = _call_ollama_writer(
                raw_input=raw_input,
                blueprint_json=blueprint_json_str,
            )
            educational_text = _extract_educational_text(raw_writer_output)

            if len(educational_text) >= 200:
                break  # good output

            print(f"\n\u26a0  Writer attempt {attempt}/{MAX_WRITER_ATTEMPTS} "
                  f"produced only {len(educational_text)} chars — "
                  f"{'retrying...' if attempt < MAX_WRITER_ATTEMPTS else 'giving up.'}")

        # Final quality gate after all retries
        if len(educational_text) < 200:
            print(f"\n\u26a0  Writer output too short ({len(educational_text)} chars) "
                  f"after {MAX_WRITER_ATTEMPTS} attempts, falling back to raw input.")
            errors.append(
                f"Writer produced only {len(educational_text)} chars — used raw input."
            )
            educational_text = raw_input

        fk_grade = flesch_kincaid_grade(educational_text)
        fre_score = flesch_reading_ease(educational_text)
        print(f"\n\u2714  Educational text generated ({len(educational_text)} chars)")
        print(f"   Flesch-Kincaid grade level: {fk_grade:.1f}")
        print(f"   Flesch Reading Ease: {fre_score:.1f}")

    except Exception as e:
        err_msg = f"Stage 2 (Writer) failed: {e}"
        errors.append(err_msg)
        print(f"\n⚠  {err_msg}")
        print("   Continuing with raw input as educational text...")
        educational_text = raw_input

    # ────────────────────────────────────────────────────────────────
    # Stage 3 — The Illustrator
    # ────────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  STAGE 3 : The Illustrator (Image Generation)")
    print("=" * 60)

    try:
        illustrator = create_illustrator_agent(llm)
        illustrator_task = create_illustrator_task(
            agent=illustrator,
            educational_text=educational_text,
            raw_topic=raw_input,
            target_group=target_group,
            output_language=output_language,
        )

        illustrator_crew = Crew(
            agents=[illustrator],
            tasks=[illustrator_task],
            process=Process.sequential,
            verbose=True,
        )
        illustrator_result = str(illustrator_crew.kickoff()).strip()

        # Extract image path from result
        parsed_ill = _extract_json(illustrator_result)
        if parsed_ill and isinstance(parsed_ill, dict):
            image_path = parsed_ill.get("image_path", "")
        else:
            # Fallback: scan for a file path
            for token in illustrator_result.replace("\\\\", "\\").split():
                token_clean = token.strip('",{}[]')
                if token_clean.endswith((".png", ".jpg")):
                    image_path = token_clean
                    break

        print(f"\n✔  Illustration saved → {image_path or '(no path extracted)'}")

    except Exception as e:
        err_msg = f"Stage 3 (Illustrator) failed: {e}"
        errors.append(err_msg)
        print(f"\n⚠  {err_msg}")
        print("   Continuing without illustration...")
        image_path = ""

    # ────────────────────────────────────────────────────────────────
    # Stage 4 — The Publisher
    # ────────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  STAGE 4 : The Publisher (PDF Assembly)")
    print("=" * 60)

    readability_info = (
        f"FK Grade: {fk_grade:.1f} | "
        f"Reading Ease: {fre_score:.1f} | "
        f"Words: {len(educational_text.split())}"
    )

    try:
        # Call build_pdf directly instead of going through the LLM-driven
        # publisher agent.  Llama 3 consistently truncates or replaces the
        # body text when forwarding it to the tool, so direct invocation
        # guarantees the full educational text is embedded in the PDF.
        pdf_path = build_pdf(
            title=raw_input,
            body_text=educational_text,
            image_path=image_path,
            target_group=target_group,
            language=output_language,
            accessibility_score=fk_grade,
            readability_info=readability_info,
        )
        print(f"\n✔  PDF assembled → {pdf_path}")

    except Exception as e:
        err_msg = f"Stage 4 (Publisher) failed: {e}"
        errors.append(err_msg)
        print(f"\n⚠  {err_msg}")
        print("   PDF generation skipped.")

    # ────────────────────────────────────────────────────────────────
    # Summary
    # ────────────────────────────────────────────────────────────────
    status = "COMPLETE" if not errors else f"PARTIAL ({len(errors)} error(s))"
    print("\n" + "=" * 60)
    print(f"  PIPELINE {status}")
    print("=" * 60)
    print(f"  Language        : {output_language}")
    print(f"  Target audience : {target_group}")
    print(f"  FK Grade Level  : {fk_grade:.1f}")
    print(f"  Image path      : {image_path or 'N/A'}")
    print(f"  PDF path        : {pdf_path or 'N/A'}")
    if errors:
        print("  Errors:")
        for err in errors:
            print(f"    ⚠  {err}")
    print("=" * 60 + "\n")

    return {
        "blueprint": blueprint,
        "educational_text": educational_text,
        "fk_grade": fk_grade,
        "image_path": image_path,
        "pdf_path": pdf_path,
        "errors": errors,
    }


# ── Entry-point ──────────────────────────────────────────────────────────────
def main():
    args = _parse_args()

    raw_input = args.input
    if not raw_input:
        print("Enter/paste your raw input text (press Ctrl-Z then Enter to finish):\n")
        try:
            raw_input = sys.stdin.read().strip()
        except EOFError:
            pass

    if not raw_input:
        print("Error: No input text provided.")
        sys.exit(1)

    run_pipeline(
        raw_input=raw_input,
        output_language=args.language,
        target_group=args.audience,
    )


if __name__ == "__main__":
    main()
