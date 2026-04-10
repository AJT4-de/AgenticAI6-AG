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
    blueprint_json: str,
) -> Task:
    """Return the writing / rewriting task."""
    return Task(
        description=(
            "You are an educational writer. Your job is to WRITE a full, "
            "detailed educational article about the topic below.\n\n"
            f"**Topic / Raw Input**:\n---\n{raw_input}\n---\n\n"
            f"**Blueprint** (use this to guide tone, audience, and structure):\n"
            f"{blueprint_json}\n\n"
            "### Writing Instructions\n"
            "1. Write a COMPLETE article of **at least 800 words**.\n"
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
            "6. End with a bullet-point summary of the 5 most important "
            "takeaways, formatted as:\n"
            "   ### Key Takeaways\n"
            "   - First takeaway\n"
            "   - Second takeaway\n"
            "   - (etc.)\n\n"
            "CRITICAL RULES (you MUST follow these):\n"
            "- Your response must be ONLY the educational article text.\n"
            "- Do NOT say 'I will provide', 'I'll provide', 'Here is', "
            "'I now can give a great answer' or ANY similar preamble.\n"
            "- Do NOT include meta-commentary about what you are doing.\n"
            "- Your FIRST line must be: ## Introduction\n"
            "- Immediately after ## Introduction, begin writing the actual content.\n"
            "- The article MUST be at least 800 words. Short responses are REJECTED."
        ),
        expected_output=(
            "A complete educational article of at least 800 words starting with "
            "'## Introduction' followed by actual content. Must have 6 sections "
            "(Introduction, Background, Key Analysis, Current Situation, Impact, "
            "Conclusion), rich Markdown formatting, and a Key Takeaways summary. "
            "No meta-commentary, no preambles."
        ),
        agent=agent,
    )


# Keep old name as alias for backward compatibility
build_writer_task = create_writer_task
