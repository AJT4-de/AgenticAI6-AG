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
from config.schemas import PublisherOutput


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
    context: list[Task] = None,
) -> Task:
    """Return the PDF-assembly task."""
    return Task(
        description=(
            "Create a PDF report using the generate_pdf tool.\n\n"
            "You MUST call the generate_pdf tool by pulling the exact information "
            "from your context (which contains the article markdown, image path, "
            "and blueprint data).\n"
            "Pass the `{output_folder}` variable as the output_folder parameter.\n"
            "After the tool returns the PDF path, respond with the pdf_path."
        ),
        expected_output="A PublisherOutput containing the final pdf_path.",
        output_pydantic=PublisherOutput,
        context=context,
        agent=agent,
    )


# Keep old name as alias for backward compatibility
build_publisher_task = create_publisher_task
