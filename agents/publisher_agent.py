"""
agents/publisher_agent.py
----------------------------------------------------------------------
The Publisher  --  Document Assembly Agent
Role   : Technical Document Engineer
Task   : Combine the educational text and generated image into a
         polished, professional PDF report.
----------------------------------------------------------------------
"""

from crewai import Agent, Task
from tools.pdf_tool import generate_pdf_tool


def create_publisher_agent(llm) -> Agent:
    """Return a configured Publisher agent."""
    return Agent(
        role="Technical Document Engineer",
        goal=(
            "Assemble the educational text and illustration into a "
            "professional, well-formatted PDF report with correct "
            "UTF-8 encoding for multi-language support."
        ),
        backstory=(
            "You are a document-engineering specialist experienced in "
            "automated report generation. You understand typography, "
            "page layout, and multi-script rendering. Your PDFs are "
            "always clean, accessible, and print-ready."
        ),
        verbose=True,
        allow_delegation=False,
        tools=[generate_pdf_tool],
        llm=llm,
    )


# Keep old name as alias for backward compatibility
build_publisher_agent = create_publisher_agent


def create_publisher_task(
    agent: Agent,
    educational_text: str,
    image_path: str,
    target_group: str,
    output_language: str,
    accessibility_score: float,
    topic_title: str = "Report",
    readability_info: str = "",
) -> Task:
    """Return the PDF-assembly task."""
    return Task(
        description=(
            "Create a PDF report using the generate_pdf tool.\n\n"
            "You MUST call the generate_pdf tool with these EXACT arguments:\n"
            f"- title: \"{topic_title}\"\n"
            f"- body_text: the COMPLETE educational text between the "
            f"---START--- and ---END--- markers below\n"
            f"- image_path: \"{image_path}\"\n"
            f"- target_group: \"{target_group}\"\n"
            f"- language: \"{output_language}\"\n"
            f"- accessibility_score: {accessibility_score}\n\n"
            "---START---\n"
            f"{educational_text}\n"
            "---END---\n\n"
            "After the tool returns the PDF path, respond with ONLY:\n"
            '{"pdf_path": "<path returned by tool>"}\n\n'
            "CRITICAL RULES:\n"
            "1. Copy the ENTIRE text between ---START--- and ---END--- "
            "into the body_text argument.\n"
            "2. Do NOT summarise, shorten, or paraphrase the text.\n"
            "3. Do NOT use placeholders like '(full text above)'.\n"
            "4. The body_text must contain ALL headings, paragraphs, "
            "and formatting from the original."
        ),
        expected_output='JSON with key "pdf_path".',
        agent=agent,
    )


# Keep old name as alias for backward compatibility
build_publisher_task = create_publisher_task
