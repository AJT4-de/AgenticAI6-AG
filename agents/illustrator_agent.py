"""
agents/illustrator_agent.py
──────────────────────────────────────────────────────────────────────
The Illustrator  —  Graphic Design Agent
Role   : Visual Specialist
Task   : Generate an image prompt from the simplified text and call
         the Hugging Face Inference API for image generation.
──────────────────────────────────────────────────────────────────────
"""

from crewai import Agent, Task
from tools.image_tool import generate_image_tool


def create_illustrator_agent(llm) -> Agent:
    """Return a configured Illustrator agent."""
    return Agent(
        role="Visual Specialist",
        goal=(
            "Create a vivid, accurate image prompt that captures the "
            "essence of the educational content, then generate the image "
            "using the available API and save it to the output folder."
        ),
        backstory=(
            "You are a senior graphic designer who specialises in "
            "educational and scientific illustration. You know how to "
            "translate abstract concepts into clear, visually appealing "
            "images that enhance understanding for any age group."
        ),
        verbose=True,
        allow_delegation=False,
        tools=[generate_image_tool],
        llm=llm,
    )


# Keep old name as alias for backward compatibility
build_illustrator_agent = create_illustrator_agent


def create_illustrator_task(
    agent: Agent,
    educational_text: str,
    target_group: str,
    output_language: str,
    raw_topic: str = "",
) -> Task:
    """Return the illustration task."""
    topic_hint = (
        f"\n**Original Topic**: {raw_topic}\n"
        if raw_topic else ""
    )
    return Task(
        description=(
            "Your job is to generate ONE illustration for the educational "
            "content below.\n\n"
            f"**Educational Text**:\n---\n{educational_text}\n---\n"
            f"{topic_hint}\n"
            f"**Target Audience**: {target_group}\n"
            f"**Language Context**: {output_language}\n\n"
            "### Step 1 - Write an image prompt\n"
            "Write a single descriptive sentence (in English) for a Stable "
            "Diffusion image generator. The prompt MUST:\n"
            "- Directly relate to the topic of the educational text above.\n"
            "- Be age-appropriate.\n"
            "- Use an infographic art style: clean, modern, data-driven visual "
            "with icons, charts, or diagrams on a plain background.\n"
            "- NOT contain any text or letters in the image.\n\n"
            "### Step 2 - Call the tool\n"
            "Call the generate_image tool. Pass the prompt as a plain string.\n"
            'Example: generate_image(prompt="a clean modern infographic illustration of ...")\n\n'
            "### Step 3 - Return the result\n"
            "After the tool returns the file path, respond with ONLY:\n"
            '{"image_prompt": "<your prompt>", "image_path": "<path from tool>"}'
        ),
        expected_output='JSON with keys "image_prompt" and "image_path".',
        agent=agent,
    )


# Keep old name as alias for backward compatibility
build_illustrator_task = create_illustrator_task
