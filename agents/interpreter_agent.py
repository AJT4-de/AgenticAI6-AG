"""
agents/interpreter_agent.py
──────────────────────────────────────────────────────────────────────
The Interpreter  —  Input Analysis Agent
Role   : Content Strategist & Linguist
Task   : Analyse raw user input and produce a "Strategic Blueprint"
         containing tone, target language, audience profile, core
         concepts, and readability target.
──────────────────────────────────────────────────────────────────────
"""

from crewai import Agent, Task
from config.settings import (
    DEFAULT_OUTPUT_LANGUAGE,
    DEFAULT_TARGET_GROUP,
    get_target_grade,
)
from config.schemas import BlueprintOutput


def create_interpreter_agent(llm) -> Agent:
    """Return a configured Interpreter agent."""
    return Agent(
        role="Content Strategist & Linguist",
        goal=(
            "Analyse the raw input text and produce a Strategic Blueprint "
            "that captures: (1) the core concepts, (2) the target audience "
            "profile, (3) the desired output language, (4) the appropriate "
            "tone and register, and (5) the Flesch-Kincaid grade-level target."
        ),
        backstory=(
            "You are a senior content strategist with deep expertise in "
            "linguistics, readability science, and audience profiling. "
            "You can instantly identify the key ideas in any domain and "
            "map them to the vocabulary and structure appropriate for "
            "any audience, from a 5-year-old child to a PhD researcher."
        ),
        verbose=True,
        allow_delegation=False,
        llm=llm,
    )


# Keep old name as alias for backward compatibility
build_interpreter_agent = create_interpreter_agent


def create_interpreter_task(
    agent: Agent,
    raw_input: str,
    target_group: str | None = None,
    output_language: str | None = None,
) -> Task:
    """Return the interpretation task for the given raw input."""
    target_group = target_group or DEFAULT_TARGET_GROUP
    output_language = output_language or DEFAULT_OUTPUT_LANGUAGE
    grade_level = get_target_grade(target_group)

    return Task(
        description=(
            f"You have received the following raw input text:\n\n"
            f"---\n{raw_input}\n---\n\n"
            f"Target Audience : {target_group}\n"
            f"Output Language : {output_language}\n"
            f"Readability Target (Flesch-Kincaid grade): {grade_level}\n\n"
            "Your job is to create a Strategic Blueprint for an educational article.\n"
            "Analyse the text and structure the core ideas appropriately for the target audience.\n"
        ),
        expected_output="A structured Blueprint detailing core concepts, audience, tone, and necessary glossary terms.",
        output_pydantic=BlueprintOutput,
        agent=agent,
    )


# Keep old name as alias for backward compatibility
build_interpreter_task = create_interpreter_task
