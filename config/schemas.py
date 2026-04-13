from pydantic import BaseModel, Field

class BlueprintOutput(BaseModel):
    core_concepts: list[str] = Field(..., description="List of 3-6 key concepts from the input")
    target_audience: str = Field(..., description="The audience label")
    grade_level: float = Field(..., description="The target FK grade as a number")
    output_language: str = Field(..., description="The output language")
    tone: str = Field(..., description="The tone of the article (e.g. friendly, academic, playful)")
    tone_register: str = Field(..., description="The register of the article (e.g. informal, formal, semi-formal)")
    key_terms_glossary: dict[str, str] = Field(..., description="A mapping of important terms to simple definitions")
    structural_notes: str = Field(..., description="Notes on how the text should be organised")

class WriterOutput(BaseModel):
    article_markdown: str = Field(..., description="The complete, finalized educational article in Markdown format. Minimum 800 words.")

class IllustratorOutput(BaseModel):
    image_prompt: str = Field(..., description="The textual prompt that was used to generate the illustration.")
    image_path: str = Field(..., description="The absolute file path where the generated image was saved.")

class PublisherOutput(BaseModel):
    pdf_path: str = Field(..., description="The absolute file path where the generated PDF was saved.")
