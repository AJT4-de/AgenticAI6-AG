# Multi-Modal Agentic PDF Report Pipeline

A **CrewAI**-powered pipeline that transforms raw input text into a polished, audience-specific PDF report with auto-generated illustrations.

## Architecture

The system coordinates **four specialised agents** in sequence:

| # | Agent | Role | Output |
|---|-------|------|--------|
| 1 | **The Interpreter** | Content Strategist & Linguist | Strategic Blueprint (JSON) |
| 2 | **The Writer** | Multi-Lingual Educational Writer | Educational text at target readability |
| 3 | **The Illustrator** | Visual Specialist | AI-generated illustration |
| 4 | **The Publisher** | Technical Document Engineer | Final PDF report |

## Requirements

- **Python 3.10 – 3.12** (strict CrewAI requirement)
- **Ollama** running locally with Llama 3 pulled
- **Hugging Face** API token (for image generation)

## Quick Start

```bash
# 1. Clone and enter the project
cd AgenticAI6

# 2. Create & activate virtual environment
python -m venv venv
# Windows PowerShell:
.\venv\Scripts\Activate.ps1
# macOS / Linux:
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
cp .env.example .env
# Edit .env and add your HF_API_TOKEN

# 5. Ensure Ollama is running with Llama 3
ollama pull llama3
ollama serve

# 6. Run the pipeline
python main.py --input "Quantum computing explained" --language English --audience "10-year-old child"
```

## Project Structure

```
AgenticAI6/
├── main.py                  # Orchestrator / entry-point
├── requirements.txt         # Python dependencies
├── .env.example             # Template for environment variables
├── .env                     # Your actual API keys (git-ignored)
├── .gitignore
├── config/
│   ├── __init__.py
│   └── settings.py          # Centralised config & defaults
├── agents/
│   ├── __init__.py
│   ├── interpreter_agent.py # The Interpreter
│   ├── writer_agent.py      # The Writer
│   ├── illustrator_agent.py # The Illustrator
│   └── publisher_agent.py   # The Publisher
├── tools/
│   ├── __init__.py
│   ├── image_tool.py        # Hugging Face image generation
│   ├── pdf_tool.py          # PDF assembly with fpdf2
│   └── readability_tool.py  # Flesch-Kincaid & other metrics
├── output/                  # Generated PDFs and images
└── venv/                    # Virtual environment (git-ignored)
```

## Configuration

Edit `.env` to set:

| Variable | Purpose |
|----------|---------|
| `OLLAMA_BASE_URL` | Ollama server URL (default: `http://localhost:11434`) |
| `OLLAMA_MODEL` | Model name (default: `llama3`) |
| `HF_API_TOKEN` | Hugging Face Inference API token |
| `HF_IMAGE_MODEL` | HF model for image generation |
| `DEFAULT_OUTPUT_LANGUAGE` | Pipeline default language |
| `DEFAULT_TARGET_GROUP` | Pipeline default audience |

## Supported Target Audiences

| Audience | FK Grade Level |
|----------|---------------|
| 5-year-old child | 1.0 |
| 10-year-old child | 5.0 |
| High-school student | 10.0 |
| University student | 13.0 |
| Graduate researcher | 16.0 |
| General public | 8.0 |

## Usage Examples

```bash
# Interactive mode (paste text, Ctrl-Z to submit on Windows)
python main.py

# Direct input with defaults
python main.py --input "The water cycle and its stages"

# Full options
python main.py \
  --input "Machine learning fundamentals" \
  --language Spanish \
  --audience "High-school student"
```

## Dependencies

| Category | Package | Purpose |
|----------|---------|---------|
| Orchestration | `crewai`, `crewai-tools` | Agent management & sequential processes |
| LLM | `langchain`, `langchain-ollama` | Backend for local Ollama Llama 3 |
| Image Gen | `requests` | Hugging Face Inference API calls |
| PDF | `fpdf2` | PDF assembly with UTF-8 support |
| Readability | `textstat` | Flesch-Kincaid accessibility scoring |
| Environment | `python-dotenv` | Secure API key management |
| Utilities | `Pillow` | Image handling (placeholder fallback) |
| Environment | `python-dotenv` | Secure API key management |
| Utilities | `Pillow`, `requests` | Image handling & HTTP |