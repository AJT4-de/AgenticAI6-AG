"""
test_pipeline.py -- Comprehensive unit tests for all pipeline components.
Run with:  python test_pipeline.py
"""

import sys
import os
import pathlib

sys.path.insert(0, os.path.dirname(__file__))
os.chdir(os.path.dirname(__file__))

passed = 0
failed = 0


def section(name):
    print(f"\n{'=' * 60}")
    print(f"  {name}")
    print("=" * 60)


# -- TEST 1: Config & Settings ------------------------------------------------
section("TEST 1: Config & Settings")
try:
    from config.settings import (
        OLLAMA_BASE_URL, OLLAMA_MODEL, HF_API_TOKEN, HF_IMAGE_MODEL,
        DEFAULT_OUTPUT_LANGUAGE, DEFAULT_TARGET_GROUP,
        OUTPUT_DIR, TARGET_GRADE_LEVELS, get_target_grade,
    )
    assert OLLAMA_MODEL, "OLLAMA_MODEL is empty"
    assert OLLAMA_BASE_URL, "OLLAMA_BASE_URL is empty"
    assert DEFAULT_OUTPUT_LANGUAGE == "English"
    assert OUTPUT_DIR.exists(), "output/ directory not created"
    assert get_target_grade("10-year-old child") == 5.0
    assert get_target_grade("University student") == 13.0
    assert get_target_grade("alien species") == 8.0  # fallback
    print(f"  OLLAMA_BASE_URL    : {OLLAMA_BASE_URL}")
    print(f"  OLLAMA_MODEL       : {OLLAMA_MODEL}")
    print(f"  HF_IMAGE_MODEL     : {HF_IMAGE_MODEL}")
    print(f"  DEFAULT_LANGUAGE   : {DEFAULT_OUTPUT_LANGUAGE}")
    print(f"  DEFAULT_TARGET     : {DEFAULT_TARGET_GROUP}")
    print(f"  OUTPUT_DIR exists  : {OUTPUT_DIR.exists()}")
    print(f"  Grade levels       : {len(TARGET_GRADE_LEVELS)} entries")
    print("  [PASS]")
    passed += 1
except Exception as e:
    print(f"  [FAIL] {e}")
    failed += 1


# -- TEST 2: Readability Tool -------------------------------------------------
section("TEST 2: Readability Tool")
try:
    from tools.readability_tool import (
        flesch_kincaid_grade, flesch_reading_ease, full_readability_report,
    )
    sample = (
        "The sun gives us light and heat. Plants use sunlight to grow. "
        "We need the sun to live. Water falls from clouds as rain."
    )
    fk = flesch_kincaid_grade(sample)
    fre = flesch_reading_ease(sample)
    report = full_readability_report(sample)
    assert isinstance(fk, (int, float)), "FK grade is not a number"
    assert isinstance(fre, (int, float)), "FRE score is not a number"
    assert "flesch_kincaid_grade" in report
    assert "text_standard" in report
    print(f"  FK grade level  : {fk}")
    print(f"  FRE score       : {fre}")
    print(f"  Report keys     : {list(report.keys())}")
    print("  [PASS]")
    passed += 1
except Exception as e:
    print(f"  [FAIL] {e}")
    failed += 1


# -- TEST 3: Agent Module Imports (new create_* names) -------------------------
section("TEST 3: Agent Module Imports")
try:
    from agents.interpreter_agent import create_interpreter_agent, create_interpreter_task
    from agents.writer_agent import create_writer_agent, create_writer_task
    from agents.illustrator_agent import create_illustrator_agent, create_illustrator_task
    from agents.publisher_agent import create_publisher_agent, create_publisher_task
    assert callable(create_interpreter_agent)
    assert callable(create_writer_task)
    assert callable(create_illustrator_agent)
    assert callable(create_publisher_task)
    # Backward compatibility aliases
    from agents.interpreter_agent import build_interpreter_agent, build_interpreter_task
    from agents.writer_agent import build_writer_agent, build_writer_task
    from agents.illustrator_agent import build_illustrator_agent, build_illustrator_task
    from agents.publisher_agent import build_publisher_agent, build_publisher_task
    assert create_interpreter_agent is build_interpreter_agent
    assert create_writer_task is build_writer_task
    print("  create_interpreter_agent : OK")
    print("  create_writer_agent      : OK")
    print("  create_illustrator_agent : OK")
    print("  create_publisher_agent   : OK")
    print("  build_* aliases          : OK (backward compat)")
    print("  [PASS]")
    passed += 1
except Exception as e:
    print(f"  [FAIL] {e}")
    failed += 1


# -- TEST 4: Tool Module Imports -----------------------------------------------
section("TEST 4: Tool Module Imports")
try:
    from tools.image_tool import generate_image_tool
    from tools.pdf_tool import generate_pdf_tool, build_pdf, ProfessionalPDF
    assert generate_image_tool.name == "generate_image"
    assert generate_pdf_tool.name == "generate_pdf"
    assert callable(build_pdf)
    assert issubclass(ProfessionalPDF, object)
    print(f"  generate_image_tool : name={generate_image_tool.name}")
    print(f"  generate_pdf_tool   : name={generate_pdf_tool.name}")
    print(f"  build_pdf()         : callable OK")
    print(f"  ProfessionalPDF     : class OK")
    print("  [PASS]")
    passed += 1
except Exception as e:
    print(f"  [FAIL] {e}")
    failed += 1


# -- TEST 5: Professional PDF Generation (build_pdf direct) --------------------
section("TEST 5: Professional PDF (build_pdf direct call)")
try:
    from tools.pdf_tool import build_pdf

    test_text = (
        "## Introduction\n"
        "This is a comprehensive test report about the solar system.\n\n"
        "> The solar system is our cosmic neighborhood.\n\n"
        "## Background & Context\n"
        "The solar system has **eight planets**. They orbit around the "
        "*Sun*, which is a medium-sized star.\n\n"
        "## Key Analysis\n"
        "The planets can be divided into two groups:\n"
        "1. Terrestrial planets (rocky inner planets)\n"
        "2. Gas giants (outer planets)\n"
        "3. Ice giants (furthest planets)\n\n"
        "### The Inner Planets\n"
        "- **Mercury** is the closest to the Sun.\n"
        "- **Venus** is the hottest planet.\n"
        "- **Earth** is our home.\n"
        "- **Mars** is called the *Red Planet*.\n\n"
        "## Current Situation & Developments\n"
        "NASA and other agencies are actively exploring Mars with rovers.\n\n"
        "---\n\n"
        "## Impact & Implications\n"
        "Understanding our solar system helps us search for life elsewhere.\n\n"
        "## Conclusion\n"
        "The solar system is a fascinating place with many worlds to explore.\n\n"
        "### Key Takeaways\n"
        "- The Sun is the centre of our solar system\n"
        "- Eight planets orbit the Sun\n"
        "- Mars is a focus of current exploration\n"
        "- Understanding space helps humanity's future\n"
        "- Many discoveries are still to be made"
    )

    pdf_path = build_pdf(
        title="Test Report: The Solar System",
        body_text=test_text,
        image_path="",
        target_group="University student",
        language="English",
        accessibility_score=8.5,
        readability_info="FK Grade: 8.5 | Reading Ease: 62.3 | Words: 150",
        key_findings=[
            "The solar system has eight planets",
            "Mars is the primary target for exploration",
            "Gas giants dominate the outer system",
        ],
    )
    p = pathlib.Path(pdf_path)
    assert p.exists(), f"PDF not found at {pdf_path}"
    assert p.stat().st_size > 1000, "PDF file is suspiciously small"
    print(f"  PDF generated at : {pdf_path}")
    print(f"  PDF file exists  : True")
    print(f"  PDF file size    : {p.stat().st_size:,} bytes")
    print("  [PASS]")
    passed += 1
except Exception as e:
    print(f"  [FAIL] {e}")
    failed += 1


# -- TEST 6: PDF via Tool (generate_pdf_tool.run) -----------------------------
section("TEST 6: PDF via Tool (generate_pdf_tool.run)")
try:
    from tools.pdf_tool import generate_pdf_tool

    long_body = (
        "## Introduction\n"
        "Artificial intelligence represents one of the most significant "
        "technological developments of the 21st century. This report "
        "explores the current state of AI.\n\n"
        "## Background & Context\n"
        "AI research began in the 1950s with pioneers like Alan Turing. "
        "Today, machine learning and deep learning have revolutionized "
        "the field.\n\n"
        "## Key Analysis\n"
        "Modern AI systems can perform tasks that were once thought to be "
        "exclusively human. Natural language processing, computer vision, "
        "and autonomous systems are advancing rapidly.\n\n"
        "## Conclusion\n"
        "AI will continue to shape our world in profound ways."
    )

    pdf_path2 = generate_pdf_tool.run(
        title="AI Report Test",
        body_text=long_body,
        image_path="",
        target_group="University student",
        language="English",
        accessibility_score=12.0,
    )
    p = pathlib.Path(pdf_path2)
    assert p.exists(), f"PDF not found at {pdf_path2}"
    assert p.stat().st_size > 500, "PDF file too small"
    print(f"  PDF via tool at  : {pdf_path2}")
    print(f"  File size        : {p.stat().st_size:,} bytes")
    print("  [PASS]")
    passed += 1
except Exception as e:
    print(f"  [FAIL] {e}")
    failed += 1


# -- TEST 7: Image Placeholder ------------------------------------------------
section("TEST 7: Image Tool -- Placeholder (no HF token)")
try:
    import tools.image_tool as img_mod
    from tools.image_tool import generate_image_tool

    orig_hf = img_mod.HF_API_TOKEN
    img_mod.HF_API_TOKEN = ""
    try:
        img_path = generate_image_tool.run(
            prompt="A bright cartoon sun smiling over green hills"
        )
    finally:
        img_mod.HF_API_TOKEN = orig_hf

    p = pathlib.Path(img_path)
    assert p.exists(), f"Image not found at {img_path}"
    assert p.stat().st_size > 100, "Image file too small"
    print(f"  Placeholder at  : {img_path}")
    print(f"  File exists     : True")
    print(f"  File size       : {p.stat().st_size:,} bytes")
    print("  [PASS]")
    passed += 1
except Exception as e:
    print(f"  [FAIL] {e}")
    failed += 1


# -- TEST 8: PDF with Image ---------------------------------------------------
section("TEST 8: PDF with embedded image")
try:
    from tools.pdf_tool import build_pdf

    pdf_path3 = build_pdf(
        title="Test Report with Illustration",
        body_text=(
            "## Solar Energy\n"
            "The sun is the main source of energy for our planet. "
            "Solar power is becoming increasingly affordable and efficient. "
            "Many countries are investing heavily in solar infrastructure.\n\n"
            "## How It Works\n"
            "Solar panels convert sunlight into electricity using photovoltaic cells. "
            "This clean energy source produces no greenhouse gas emissions.\n\n"
            "## Conclusion\n"
            "Solar energy is a key part of our sustainable future."
        ),
        image_path=img_path,
        target_group="10-year-old child",
        language="English",
        accessibility_score=4.0,
        readability_info="FK Grade: 4.0 | Reading Ease: 85.0 | Words: 80",
    )
    p = pathlib.Path(pdf_path3)
    assert p.exists(), f"PDF not found at {pdf_path3}"
    assert p.stat().st_size > 2000, "PDF with image should be larger"
    print(f"  PDF with image at : {pdf_path3}")
    print(f"  File size         : {p.stat().st_size:,} bytes")
    print("  [PASS]")
    passed += 1
except Exception as e:
    print(f"  [FAIL] {e}")
    failed += 1


# -- TEST 9: Main module import -----------------------------------------------
section("TEST 9: Main orchestrator import")
try:
    from main import run_pipeline, _get_llm, _parse_args
    assert callable(run_pipeline)
    assert callable(_get_llm)
    print("  run_pipeline()  : callable OK")
    print("  _get_llm()      : callable OK")
    print("  [PASS]")
    passed += 1
except Exception as e:
    print(f"  [FAIL] {e}")
    failed += 1


# -- TEST 10: Ollama live connectivity ----------------------------------------
section("TEST 10: Ollama live connectivity (Llama 3)")
try:
    import requests as _req
    from config.settings import OLLAMA_BASE_URL, OLLAMA_MODEL
    r = _req.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
    r.raise_for_status()
    models = [m["name"] for m in r.json().get("models", [])]
    assert any("llama" in m.lower() for m in models), (
        f"No llama model found. Available: {models}"
    )
    print(f"  Ollama reachable : {OLLAMA_BASE_URL}")
    print(f"  Available models : {models}")

    from crewai import LLM
    llm = LLM(model=f"ollama/{OLLAMA_MODEL}", base_url=OLLAMA_BASE_URL, temperature=0.0)
    resp = llm.call(messages=[{"role": "user", "content": "Say only: hello"}])
    reply = resp.strip() if isinstance(resp, str) else str(resp).strip()
    assert len(reply) > 0, "Empty response from Ollama"
    print(f"  LLM response     : {reply[:80]}")
    print("  [PASS]")
    passed += 1
except Exception as e:
    print(f"  [FAIL] {e}")
    failed += 1


# -- Summary -------------------------------------------------------------------
print("\n" + "=" * 60)
print(f"  RESULTS: {passed} passed, {failed} failed, {passed + failed} total")
print("=" * 60)

if failed:
    sys.exit(1)
else:
    print("  ALL TESTS PASSED")
    sys.exit(0)
