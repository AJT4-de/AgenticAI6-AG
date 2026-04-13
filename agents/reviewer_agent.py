"""
agents/reviewer_agent.py
----------------------------------------------------------------------
The Reviewer  --  Content Quality Agent
Role   : Quality Assurance Editor
Task   : Review the Writer's output against the Blueprint requirements.
         Allows delegation back to Writer if corrections are needed.
----------------------------------------------------------------------
"""

from crewai import Agent, Task
from config.schemas import WriterOutput

def create_reviewer_agent(llm) -> Agent:
    """Return a configured Reviewer agent."""
    return Agent(
        role="Quality Assurance Editor",
        goal="Review drafted articles to ensure they meet all formatting, structural, and length requirements.",
        backstory=(
            "You are a strict QA editor. You verify that writers follow their guidelines perfectly. "
            "If an article is missing sections, doesn't use Markdown correctly, or is too short, "
            "you immediately delegate it back to the writer for revision."
        ),
        verbose=True,
        allow_delegation=True,
        max_iter=3,
        llm=llm,
    )

def create_reviewer_task(
    agent: Agent,
    context: list[Task] = None,
) -> Task:
    """Return the review task."""
    return Task(
        description=(
            "Review the article produced by the writer.\n"
            "Ensure the article meets the following checklist:\n"
            "1. It is at least 800 words long.\n"
            "2. It uses Markdown formatting correctly.\n"
            "3. It has the 6 required sections (Introduction, Background & Context, Key Analysis, Current Situation & Developments, Impact & Implications, Conclusion) with `## ` headings.\n"
            "4. It ends with a `### Key Takeaways` bulleted summary.\n"
            "If it fails ANY of these checks, you MUST use your delegation tool to send it back to the writer to fix the specific issues. "
            "If it passes, output the final approved article exactly as written."
        ),
        expected_output="The finalized, verified educational article in Markdown format.",
        output_pydantic=WriterOutput,
        context=context,
        agent=agent,
    )
