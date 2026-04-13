"""
agents/writer_agent.py
----------------------------------------------------------------------
The Writer  --  Content Generation Agent
Role   : Multi-Lingual Educational Writer
Task   : Rewrite the input into the selected language at the correct
         complexity / readability level, guided by the Strategic
         Blueprint from The Interpreter.

Produces long-form (800+ word) articles with 6 structured sections,
Markdown formatting (headings, bold, italic, blockquotes, numbered
lists, bullet lists) for rich PDF rendering.
----------------------------------------------------------------------
"""

from crewai import Agent, Task
from config.schemas import WriterOutput


def create_writer_agent(llm) -> Agent:
    """Return a configured Writer agent."""
    return Agent(
        role="Multi-Lingual Educational Writer",
        goal=(
            "Transform raw or complex input text into clear, engaging, "
            "in-depth educational content that matches the target "
            "audience's reading level and is written in the specified "
            "output language. Produce comprehensive, well-structured "
            "articles of at least 800 words."
        ),
        backstory=(
            "You are an award-winning educational writer fluent in dozens "
            "of languages. You specialise in making complex topics "
            "accessible without sacrificing accuracy. You pay close "
            "attention to readability metrics and can write at any "
            "Flesch-Kincaid grade level on demand. Your articles are "
            "known for being thorough, well-researched, and beautifully "
            "structured with clear sections, examples, and summaries."
        ),
        verbose=True,
        allow_delegation=False,
        llm=llm,
    )


# Keep old name as alias for backward compatibility
build_writer_agent = create_writer_agent


def create_writer_task(
    agent: Agent,
    raw_input: str,
    context: list[Task] = None,
) -> Task:
    """Return the writing / rewriting task."""
    return Task(
        description=(
            "You are an educational writer. Your job is to WRITE a full, "
            "detailed educational article about the topic below.\n\n"
            f"**Topic / Raw Input**:\n---\n{raw_input}\n---\n\n"
            "Use the Strategic Blueprint provided in the context to guide "
            "your tone, audience, and structure.\n\n"
            "### Writing Guidelines\n"
            "1. Write a comprehensive article of at least 800 words.\n"
            "2. Ensure you use Markdown formatting:\n"
            "   - `## ` for the 6 main sections (Introduction, Background & Context, "
            "Key Analysis, Current Situation & Developments, Impact & Implications, Conclusion)\n"
            "   - `**bold**` and `*italic*` for emphasis\n"
            "   - `> blockquotes` for key takeaways\n"
            "   - `- ` or `1. ` for lists\n"
            "3. End with a `### Key Takeaways` bulleted summary.\n"
            "4. Define key terms from the blueprint naturally in the text.\n\n"
            "### Format Example\n"
            "## Introduction\n"
            "Here is the beginning of the article...\n"
            "## Background & Context\n"
            "Here is the context...\n"
            "### Key Takeaways\n"
            "- First takeaway\n"
            "- Second takeaway"
        ),
        expected_output="A complete, well-structured educational article in Markdown format matching the blueprint requirements.",
        output_pydantic=WriterOutput,
        context=context,
        agent=agent,
    )


# Keep old name as alias for backward compatibility
build_writer_task = create_writer_task
