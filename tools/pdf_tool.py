"""
tools/pdf_tool.py
----------------------------------------------------------------------
Professional PDF report generator using fpdf2.

Features:
  - Cover page with title, metadata, and date
  - Table of Contents
  - Metadata info box (audience, language, readability)
  - Key Findings / Executive Summary box
  - Styled section headers (3 levels) with blue colour scheme
  - Rich body rendering: bold, italic, underline, blockquotes,
    numbered lists, bullet lists, horizontal rules
  - Full-page illustration placement
  - Anti-overwrite guard and Llama 3 dict-format handling

Uses BaseTool (not @tool decorator) so we can accept flexible input
-- Llama 3 sometimes passes arguments as dicts instead of strings.
----------------------------------------------------------------------
"""

from __future__ import annotations

import datetime
import re
from pathlib import Path
from typing import Any, Type

from crewai.tools import BaseTool
from fpdf import FPDF
from pydantic import BaseModel, Field
import asyncio


# -- Colour palette ----------------------------------------------------
_DARK_BLUE = (41, 65, 122)       # #29417A
_MID_BLUE = (70, 110, 180)       # softer accent
_LIGHT_BLUE_BG = (230, 237, 248)
_GREY_TEXT = (100, 100, 100)
_WHITE = (255, 255, 255)
_BLACK = (30, 30, 30)
_LIGHT_GREY = (220, 220, 220)
_BOX_BORDER = (180, 195, 220)
_FINDINGS_BG = (245, 250, 255)


# -- Unicode -> ASCII mapping for Helvetica compatibility --------------
_UNICODE_MAP = {
    "\u2014": "--",     # em-dash
    "\u2013": "-",      # en-dash
    "\u2018": "'",      # left single quote
    "\u2019": "'",      # right single quote
    "\u201c": '"',      # left double quote
    "\u201d": '"',      # right double quote
    "\u2022": "-",      # bullet
    "\u2026": "...",    # ellipsis
    "\u00a0": " ",      # non-breaking space
    "\u200b": "",       # zero-width space
    "\u2212": "-",      # minus sign
    "\u00b7": "-",      # middle dot
    "\u25cf": "-",      # black circle
    "\u25cb": "-",      # white circle
    "\u2713": "[x]",    # check mark
    "\u2714": "[x]",    # heavy check mark
    "\u2715": "[!]",    # multiplication X
    "\u2716": "[!]",    # heavy multiplication X
    "\u00ab": "<<",     # left guillemet
    "\u00bb": ">>",     # right guillemet
}


def _sanitize(text: str) -> str:
    """Replace Unicode characters that Helvetica cannot render."""
    for char, replacement in _UNICODE_MAP.items():
        text = text.replace(char, replacement)
    text = text.encode("latin-1", errors="replace").decode("latin-1")
    return text





# -- Inline formatting regex ------------------------------------------
_INLINE_RE = re.compile(
    r'(__(.+?)__)'           # __underline__
    r'|(\*\*(.+?)\*\*)'     # **bold**
    r'|(\*(.+?)\*)'         # *italic*
)


def _prescan_headings(body_text: str) -> list[tuple[str, int]]:
    """Pre-scan body text for Markdown headings to build TOC entries.

    Returns a list of (title, level) tuples:
      level 0 = ## heading
      level 1 = # heading
      level 2 = ### heading
    """
    entries = []
    for line in body_text.split("\n"):
        stripped = line.strip()
        if stripped.startswith("###"):
            entries.append((stripped.lstrip("#").strip(), 2))
        elif stripped.startswith("##"):
            entries.append((stripped.lstrip("#").strip(), 0))
        elif stripped.startswith("#"):
            entries.append((stripped.lstrip("#").strip(), 1))
    return entries


# =====================================================================
#   ProfessionalPDF  --  the core PDF class
# =====================================================================

class ProfessionalPDF(FPDF):
    """Full-featured PDF with cover page, TOC, styled sections, and
    rich body text rendering."""

    def __init__(self):
        super().__init__()
        self.report_title = ""
        self.report_language = ""
        self.report_target = ""
        self.report_score = 0.0
        self._toc_entries: list[tuple[str, int, int]] = []

    # -- Header / Footer -----------------------------------------------

    def header(self):
        if self.page_no() == 1:
            return  # cover page has its own layout
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(*_GREY_TEXT)
        self.cell(0, 8, _sanitize(self.report_title), align="L",
                  new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(*_MID_BLUE)
        self.set_line_width(0.4)
        self.line(self.l_margin, self.get_y(),
                  self.w - self.r_margin, self.get_y())
        self.ln(4)

    def footer(self):
        if self.page_no() == 1:
            return
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(*_GREY_TEXT)
        self.cell(0, 10, f"Page {self.page_no() - 1}", align="C")

    # -- Cover Page ----------------------------------------------------

    def add_cover_page(self, title: str, target_group: str,
                       language: str, score: float):
        """Create a professional cover page."""
        self.report_title = title
        self.report_target = target_group
        self.report_language = language
        self.report_score = score
        self.add_page()

        # Blue banner
        self.set_fill_color(*_DARK_BLUE)
        self.rect(0, 0, self.w, 100, "F")

        # Title text (white on blue)
        self.set_y(25)
        self.set_font("Helvetica", "B", 28)
        self.set_text_color(*_WHITE)
        self.multi_cell(0, 14, _sanitize(title), align="C",
                        new_x="LMARGIN", new_y="NEXT")

        # Subtitle line
        self.ln(4)
        self.set_font("Helvetica", "", 13)
        self.cell(0, 8, "Educational Report", align="C",
                  new_x="LMARGIN", new_y="NEXT")

        # Decorative line
        self.ln(10)
        y = self.get_y()
        self.set_draw_color(*_WHITE)
        self.set_line_width(0.8)
        cx = self.w / 2
        self.line(cx - 40, y, cx + 40, y)

        # Metadata block below the banner
        self.set_y(115)
        self.set_text_color(*_BLACK)
        self.set_font("Helvetica", "", 12)

        info_lines = [
            f"Audience:  {target_group}",
            f"Language:  {language}",
            f"Readability (FK Grade):  {score:.1f}",
            f"Generated:  {datetime.datetime.now().strftime('%B %d, %Y  %H:%M')}",
        ]
        for line in info_lines:
            self.cell(0, 10, _sanitize(line), align="C",
                      new_x="LMARGIN", new_y="NEXT")

        # Footer note on cover
        self.set_y(-35)
        self.set_font("Helvetica", "I", 9)
        self.set_text_color(*_GREY_TEXT)
        self.cell(
            0, 8,
            "Generated by Multi-Modal Agentic AI Pipeline  |  CrewAI + Ollama Llama 3",
            align="C",
        )

    # -- Table of Contents ---------------------------------------------

    def add_table_of_contents(self):
        """Insert a Table of Contents page in tabular form with page numbers.

        Page numbers are read directly from self._toc_entries (pre-populated
        by a dry-run render before this method is called).
        """
        self.add_page()
        self.set_font("Helvetica", "B", 20)
        self.set_text_color(*_DARK_BLUE)
        self.cell(0, 14, "Table of Contents", new_x="LMARGIN", new_y="NEXT")
        self.ln(6)

        # ---- Table header ----
        x = self.l_margin
        w = self.w - self.l_margin - self.r_margin
        col_num_w = 14          # "#" column
        col_page_w = 22         # "Page" column
        col_title_w = w - col_num_w - col_page_w  # remaining for title
        row_h = 8

        self.set_fill_color(*_DARK_BLUE)
        self.set_text_color(*_WHITE)
        self.set_font("Helvetica", "B", 11)
        self.set_draw_color(*_BOX_BORDER)
        self.set_line_width(0.3)

        self.cell(col_num_w, row_h, "#", border=1, align="C",
                  fill=True, new_x="END", new_y="TOP")
        self.cell(col_title_w, row_h, "  Section", border=1,
                  fill=True, new_x="END", new_y="TOP")
        self.cell(col_page_w, row_h, "Page", border=1, align="R",
                  fill=True, new_x="LMARGIN", new_y="NEXT")

        # ---- Table rows ----
        self.set_text_color(*_BLACK)
        for idx, (title, level, page_num) in enumerate(self._toc_entries, start=1):
            # Alternate row background
            if idx % 2 == 0:
                self.set_fill_color(*_LIGHT_BLUE_BG)
                fill = True
            else:
                self.set_fill_color(*_WHITE)
                fill = True

            # Style based on heading level
            indent = "    " * level
            if level == 0:
                self.set_font("Helvetica", "B", 11)
            else:
                self.set_font("Helvetica", "", 10)

            self.cell(col_num_w, row_h, str(idx), border=1, align="C",
                      fill=fill, new_x="END", new_y="TOP")
            self.set_font("Helvetica", "B" if level == 0 else "",
                          11 if level == 0 else 10)
            self.cell(col_title_w, row_h,
                      f"  {indent}{_sanitize(title)}", border=1,
                      fill=fill, new_x="END", new_y="TOP")
            self.set_font("Helvetica", "", 10)
            self.cell(col_page_w, row_h, str(page_num), border=1,
                      align="R", fill=fill,
                      new_x="LMARGIN", new_y="NEXT")

        self.ln(6)

    # -- Metadata Info Box ---------------------------------------------

    def add_metadata_box(self, target_group: str, language: str,
                         score: float, readability_info: str = ""):
        """Render a styled information box with report metadata."""
        self.ln(4)
        x = self.l_margin
        w = self.w - self.l_margin - self.r_margin
        y_start = self.get_y()

        box_h = 42 if not readability_info else 52
        self.set_fill_color(*_LIGHT_BLUE_BG)
        self.set_draw_color(*_BOX_BORDER)
        self.set_line_width(0.5)
        self.rect(x, y_start, w, box_h, "DF")

        # Left accent bar
        self.set_fill_color(*_DARK_BLUE)
        self.rect(x, y_start, 4, box_h, "F")

        # Content
        self.set_xy(x + 10, y_start + 4)
        self.set_font("Helvetica", "B", 11)
        self.set_text_color(*_DARK_BLUE)
        self.cell(0, 7, "Report Information", new_x="LMARGIN", new_y="NEXT")

        self.set_x(x + 10)
        self.set_font("Helvetica", "", 10)
        self.set_text_color(*_BLACK)
        self.cell(0, 6,
                  _sanitize(f"Target Audience: {target_group}   |   "
                            f"Language: {language}   |   "
                            f"FK Grade: {score:.1f}"),
                  new_x="LMARGIN", new_y="NEXT")

        self.set_x(x + 10)
        self.cell(0, 6,
                  _sanitize(f"Date: "
                            f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}"),
                  new_x="LMARGIN", new_y="NEXT")

        if readability_info:
            self.set_x(x + 10)
            self.set_font("Helvetica", "I", 9)
            self.set_text_color(*_GREY_TEXT)
            self.cell(0, 6, _sanitize(readability_info),
                      new_x="LMARGIN", new_y="NEXT")

        self.set_y(y_start + box_h + 6)

    # -- Key Findings / Executive Summary Box --------------------------

    def add_key_findings_box(self, findings: list[str]):
        """Render a key findings / executive summary box."""
        if not findings:
            return
        self.ln(2)
        x = self.l_margin
        w = self.w - self.l_margin - self.r_margin
        y_start = self.get_y()

        line_h = 7
        box_h = 14 + len(findings) * line_h + 6

        self.set_fill_color(*_FINDINGS_BG)
        self.set_draw_color(*_BOX_BORDER)
        self.set_line_width(0.4)
        self.rect(x, y_start, w, box_h, "DF")

        # Top accent line
        self.set_fill_color(*_MID_BLUE)
        self.rect(x, y_start, w, 3, "F")

        self.set_xy(x + 8, y_start + 6)
        self.set_font("Helvetica", "B", 11)
        self.set_text_color(*_DARK_BLUE)
        self.cell(0, 7, "Key Findings", new_x="LMARGIN", new_y="NEXT")

        self.set_font("Helvetica", "", 10)
        self.set_text_color(*_BLACK)
        for finding in findings:
            self.set_x(x + 12)
            self.cell(4, line_h, chr(127), new_x="END", new_y="TOP")
            self.cell(0, line_h, _sanitize(f"  {finding}"),
                      new_x="LMARGIN", new_y="NEXT")

        self.set_y(y_start + box_h + 4)

    # -- Section Headers (3 levels) ------------------------------------

    def add_section_header(self, title: str, level: int = 0):
        """Add a styled section heading.

        level 0 = major heading  (## in Markdown)
        level 1 = sub-heading    (# single hash)
        level 2 = minor heading  (### in Markdown)
        """
        self._toc_entries.append((title, level, self.page_no()))
        title = _sanitize(title)

        if level == 0:
            self.ln(6)
            self.set_font("Helvetica", "B", 16)
            self.set_text_color(*_DARK_BLUE)
            self.cell(0, 10, title, new_x="LMARGIN", new_y="NEXT")
            # Underline accent
            self.set_draw_color(*_MID_BLUE)
            self.set_line_width(0.6)
            self.line(self.l_margin, self.get_y(),
                      self.l_margin + min(len(title) * 4.5, 120),
                      self.get_y())
            self.ln(4)

        elif level == 1:
            self.ln(4)
            self.set_font("Helvetica", "B", 13)
            self.set_text_color(*_MID_BLUE)
            self.cell(0, 8, title, new_x="LMARGIN", new_y="NEXT")
            self.ln(2)

        else:
            self.ln(3)
            self.set_font("Helvetica", "BI", 11)
            self.set_text_color(*_DARK_BLUE)
            self.cell(0, 7, title, new_x="LMARGIN", new_y="NEXT")
            self.ln(1)

    # -- Illustration --------------------------------------------------

    def add_illustration(self, image_path: str):
        """Insert the illustration image spanning the usable width."""
        if not image_path or not Path(image_path).exists():
            return
        self.ln(4)
        usable_w = self.w - self.l_margin - self.r_margin
        self.set_draw_color(*_BOX_BORDER)
        self.set_line_width(0.3)
        self.image(image_path, x=self.l_margin, w=usable_w)
        self.ln(6)

    # -- Horizontal Rule -----------------------------------------------

    def add_horizontal_rule(self):
        self.ln(4)
        self.set_draw_color(*_LIGHT_GREY)
        self.set_line_width(0.3)
        self.line(self.l_margin + 20, self.get_y(),
                  self.w - self.r_margin - 20, self.get_y())
        self.ln(4)


# =====================================================================
#  Rich Body Renderer
# =====================================================================

def _write_inline_formatted(pdf: ProfessionalPDF, text: str,
                            base_size: int = 11) -> None:
    """Write a line handling **bold**, *italic*, and __underline__."""
    last_end = 0
    for m in _INLINE_RE.finditer(text):
        before = text[last_end:m.start()]
        if before:
            pdf.set_font("Helvetica", "", base_size)
            pdf.set_text_color(*_BLACK)
            pdf.write(6, _sanitize(before))

        if m.group(2):       # __underline__
            pdf.set_font("Helvetica", "U", base_size)
            pdf.set_text_color(*_BLACK)
            pdf.write(6, _sanitize(m.group(2)))
        elif m.group(4):     # **bold**
            pdf.set_font("Helvetica", "B", base_size)
            pdf.set_text_color(*_BLACK)
            pdf.write(6, _sanitize(m.group(4)))
        elif m.group(6):     # *italic*
            pdf.set_font("Helvetica", "I", base_size)
            pdf.set_text_color(*_GREY_TEXT)
            pdf.write(6, _sanitize(m.group(6)))

        last_end = m.end()

    remainder = text[last_end:]
    if remainder:
        pdf.set_font("Helvetica", "", base_size)
        pdf.set_text_color(*_BLACK)
        pdf.write(6, _sanitize(remainder))


def _render_body(pdf: ProfessionalPDF, body_text: str) -> None:
    """Parse Markdown-like body text and render with rich formatting.

    Supported syntax:
      ## Heading         -> level-0 section header
      # Heading          -> level-1 section header
      ### Heading        -> level-2 section header
      **bold**           -> inline bold
      *italic*           -> inline italic
      __underline__      -> inline underline
      > blockquote       -> indented italic block
      1. / 2. etc.       -> numbered list
      * or -             -> bullet list
      ---                -> horizontal rule
    """
    numbered_re = re.compile(r'^(\d+)\.\s+(.+)')

    for paragraph in body_text.split("\n"):
        stripped = paragraph.strip()
        if not stripped:
            pdf.ln(3)
            continue

        # -- Horizontal rule -------------------------------------------
        if stripped in ("---", "***", "___"):
            pdf.add_horizontal_rule()
            continue

        # -- Level-2 heading (###) -------------------------------------
        if stripped.startswith("###"):
            heading_text = stripped.lstrip("#").strip()
            pdf.add_section_header(heading_text, level=2)
            continue

        # -- Level-0 heading (##) --------------------------------------
        if stripped.startswith("##"):
            heading_text = stripped.lstrip("#").strip()
            pdf.add_section_header(heading_text, level=0)
            continue

        # -- Level-1 heading (#) ---------------------------------------
        if stripped.startswith("#"):
            heading_text = stripped.lstrip("#").strip()
            pdf.add_section_header(heading_text, level=1)
            continue

        # -- Blockquote (>) --------------------------------------------
        if stripped.startswith(">"):
            quote_text = stripped.lstrip(">").strip()
            x_save = pdf.l_margin
            pdf.set_fill_color(*_LIGHT_BLUE_BG)
            y_q = pdf.get_y()
            pdf.set_x(x_save + 10)
            pdf.set_font("Helvetica", "I", 10)
            pdf.set_text_color(*_GREY_TEXT)

            # Left bar
            pdf.set_draw_color(*_MID_BLUE)
            pdf.set_line_width(1.2)
            pdf.line(x_save + 8, y_q, x_save + 8, y_q + 8)

            pdf.multi_cell(pdf.w - pdf.r_margin - x_save - 14, 6,
                           _sanitize(quote_text),
                           new_x="LMARGIN", new_y="NEXT")
            pdf.ln(3)
            continue

        # -- Numbered list (1. 2. 3.) ----------------------------------
        nm = numbered_re.match(stripped)
        if nm:
            num = nm.group(1)
            item_text = nm.group(2).strip()
            pdf.set_font("Helvetica", "B", 11)
            pdf.set_text_color(*_MID_BLUE)
            pdf.cell(10, 6, f"{num}.", new_x="END", new_y="TOP")
            _write_inline_formatted(pdf, item_text, base_size=11)
            pdf.ln(6)
            continue

        # -- Bullet list item (* or -) ---------------------------------
        if stripped.startswith(("* ", "- ")):
            item_text = stripped[2:].strip()
            pdf.set_font("Helvetica", "", 11)
            pdf.set_text_color(*_MID_BLUE)
            pdf.cell(6, 6, chr(127), new_x="END", new_y="TOP")
            _write_inline_formatted(pdf, item_text, base_size=11)
            pdf.ln(6)
            continue

        # -- Normal paragraph ------------------------------------------
        _write_inline_formatted(pdf, stripped, base_size=11)
        pdf.ln(6)


# =====================================================================
#  build_pdf  --  high-level convenience function
# =====================================================================

def build_pdf(
    title: str,
    body_text: str,
    output_folder: str,
    image_path: str = "",
    target_group: str = "General public",
    language: str = "English",
    accessibility_score: float = 0.0,
    readability_info: str = "",
    key_findings: list[str] | None = None,
) -> str:
    """Build a professional multi-page PDF and return the output path.

    This is the primary entry-point used by both the CrewAI tool and
    direct test calls.
    """
    title = _sanitize(title)
    body_text = _sanitize(body_text)
    image_path = image_path
    target_group = _sanitize(target_group)
    language = _sanitize(language)
    readability_info = _sanitize(readability_info) if readability_info else ""

    pdf = ProfessionalPDF()
    pdf.alias_nb_pages()

    # 1. Cover page
    pdf.add_cover_page(title, target_group, language, accessibility_score)

    # 2. Table of Contents (right after cover page)
    #    Dry-run: render the body into a throw-away PDF so we can
    #    discover the real page number where each heading lands.
    tmp = ProfessionalPDF()
    tmp.report_title = title
    tmp.alias_nb_pages()
    tmp.add_page()          # page 1 – stands in for cover
    tmp.add_page()          # page 2 – stands in for TOC
    tmp.add_page()          # page 3 – body start (same as real PDF)
    if key_findings:
        tmp.add_key_findings_box(key_findings)
    tmp.add_illustration(image_path)
    _render_body(tmp, body_text)

    # Footer displays page_no()-1, so TOC numbers must match.
    for h_title, h_level, raw_pg in tmp._toc_entries:
        pdf._toc_entries.append((h_title, h_level, raw_pg - 1))

    pdf.add_table_of_contents()

    # 3. Report Information box (same page as TOC)
    pdf.add_metadata_box(target_group, language, accessibility_score,
                         readability_info)

    # 4. Body content page(s)
    pdf.add_page()

    # 5. Key findings box (if provided)
    if key_findings:
        pdf.add_key_findings_box(key_findings)

    # 6. Illustration (before body text)
    pdf.add_illustration(image_path)

    # 7. Body text with rich formatting
    _render_body(pdf, body_text)

    # 8. Footer readability badge
    pdf.ln(10)
    pdf.add_horizontal_rule()
    pdf.set_font("Helvetica", "I", 9)
    pdf.set_text_color(*_GREY_TEXT)
    pdf.cell(0, 6,
             _sanitize(f"Readability: Flesch-Kincaid Grade "
                       f"{accessibility_score:.1f}  |  "
                       f"Audience: {target_group}"),
             align="C", new_x="LMARGIN", new_y="NEXT")

    # Save
    safe_name = re.sub(r'[<>:"/\\|?*]', '_', title).strip().strip('.')
    safe_name = safe_name[:100] or "report"
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{safe_name}_{timestamp}.pdf"
    out_dir = Path(output_folder)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / filename
    pdf.output(str(out_path))
    return str(out_path)


# =====================================================================
#  CrewAI BaseTool wrapper
# =====================================================================

class GeneratePdfInput(BaseModel):
    """Strict input schema."""
    title: str = Field(..., description="Report title.")
    body_text: str = Field(..., description="The full educational text.")
    output_folder: str = Field(..., description="Absolute path to the session output folder.")
    image_path: str = Field(..., description="Path to the illustration.")
    target_group: str = Field(..., description="Intended audience label.")
    language: str = Field(..., description="Language of the report.")
    accessibility_score: float = Field(..., description="FK readability score.")


class GeneratePdfTool(BaseTool):
    name: str = "generate_pdf"
    description: str = (
        "Assemble text and an image into a professional PDF report. "
        "Returns the absolute file path of the generated PDF."
    )
    args_schema: Type[BaseModel] = GeneratePdfInput

    def _run(self, *args, **kwargs) -> str:
        raise NotImplementedError("This tool strictly uses async execution.")

    async def _arun(
        self,
        title: str,
        body_text: str,
        output_folder: str,
        image_path: str,
        target_group: str,
        language: str,
        accessibility_score: float,
    ) -> str:
        # Guard: reject placeholder / empty body text
        clean_check = body_text.strip().strip('.')
        if len(clean_check) < 200:
            raise ValueError(
                "ERROR: body_text is too short or is a placeholder. "
                "You must pass the FULL educational text! "
                "Do not summarize or use placeholders."
            )

        loop = asyncio.get_running_loop()
        
        def _execute_pdf():
            return build_pdf(
                title=title,
                body_text=body_text,
                output_folder=output_folder,
                image_path=image_path,
                target_group=target_group,
                language=language,
                accessibility_score=accessibility_score,
            )

        return await loop.run_in_executor(None, _execute_pdf)


# Module-level instance for easy import
generate_pdf_tool = GeneratePdfTool()
