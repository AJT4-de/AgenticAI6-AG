"""
main.py  —  Multi-Modal Agentic Pipeline Orchestrator
══════════════════════════════════════════════════════════════════════
Lead Orchestrator for the CrewAI system.
Transforms raw input into a high-quality, target-group-specific
PDF report by coordinating five specialised agents in sequence:

    1. The Interpreter  → Strategic Blueprint
    2. The Writer       → Educational Text
    3. The Reviewer     → Validation
    4. The Illustrator  → Image Generation
    5. The Publisher    → PDF Assembly

Usage:
    python main.py
    python main.py --input "Your raw text here" --language French --audience "10-year-old child"
══════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import argparse
import sys
import uuid
import os
from pathlib import Path

# Override stdout encoding for Windows Emojis
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

from crewai import Crew, LLM, Process

from config.settings import (
    DEFAULT_OUTPUT_LANGUAGE,
    DEFAULT_TARGET_GROUP,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
)
from agents.interpreter_agent import create_interpreter_agent, create_interpreter_task
from agents.writer_agent import create_writer_agent, create_writer_task
from agents.reviewer_agent import create_reviewer_agent, create_reviewer_task
from agents.illustrator_agent import create_illustrator_agent, create_illustrator_task
from agents.publisher_agent import create_publisher_agent, create_publisher_task


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


def _get_llm():
    """Return a CrewAI LLM instance connected to the local Ollama server."""
    return LLM(
        model=f"ollama/{OLLAMA_MODEL}",
        base_url=OLLAMA_BASE_URL,
        temperature=0.4,
    )


def run_pipeline(
    raw_input: str,
    output_language: str = DEFAULT_OUTPUT_LANGUAGE,
    target_group: str = DEFAULT_TARGET_GROUP,
) -> dict:
    """Execute the full five-agent pipeline and return result paths."""
    print("\n" + "=" * 60)
    print("  INITIALIZING AGENT PIPELINE")
    print("=" * 60)

    llm = _get_llm()

    # Agents
    interpreter = create_interpreter_agent(llm)
    writer = create_writer_agent(llm)
    reviewer = create_reviewer_agent(llm)
    illustrator = create_illustrator_agent(llm)
    publisher = create_publisher_agent(llm)

    # Tasks
    interpreter_task = create_interpreter_task(
        agent=interpreter,
        raw_input=raw_input,
        target_group=target_group,
        output_language=output_language,
    )

    writer_task = create_writer_task(
        agent=writer,
        raw_input=raw_input,
        context=[interpreter_task],
    )

    reviewer_task = create_reviewer_task(
        agent=reviewer,
        context=[writer_task],
    )

    illustrator_task = create_illustrator_task(
        agent=illustrator,
        context=[reviewer_task, interpreter_task],
    )

    publisher_task = create_publisher_task(
        agent=publisher,
        context=[reviewer_task, illustrator_task, interpreter_task],
    )

    crew = Crew(
        agents=[interpreter, writer, reviewer, illustrator, publisher],
        tasks=[
            interpreter_task,
            writer_task,
            reviewer_task,
            illustrator_task,
            publisher_task,
        ],
        process=Process.sequential,
        verbose=False,
    )

    project_root = Path(__file__).resolve().parent
    session_id = str(uuid.uuid4())[:8]
    output_folder = str(project_root / "output" / f"session_{session_id}")

    inputs = {
        "output_folder": output_folder,
    }

    try:
        result = crew.kickoff(inputs=inputs)
        print("\n" + "=" * 60)
        print("  PIPELINE COMPLETE")
        print("=" * 60)
        print(f"  Session ID      : {session_id}")
        print(f"  Output folder   : {output_folder}")
        print(f"  Final Result    : {result}")
        print("=" * 60 + "\n")
        return {"result": result, "output_folder": output_folder}

    except Exception as e:
        print(f"\n[WARNING] Pipeline failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}


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
