"""Render RESEARCH_PAPER.md in an IEEE-like two-column paper format.

The visual target is the reference paper supplied by the user: A4 page,
centered title/author block, dense two-column body, roman-numeral section
headings, compact figures/tables, and numbered references. The renderer does
not add fake conference, IEEE, DOI, or copyright claims.
"""

from __future__ import annotations

import html
import re
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    FrameBreak,
    HRFlowable,
    Image,
    KeepTogether,
    ListFlowable,
    ListItem,
    PageTemplate,
    Paragraph,
    Preformatted,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "RESEARCH_PAPER.md"
OUTPUT = ROOT / "RESEARCH_PAPER.pdf"

PAGE_WIDTH, PAGE_HEIGHT = A4
LEFT_MARGIN = 0.62 * inch
RIGHT_MARGIN = 0.62 * inch
TOP_MARGIN = 0.48 * inch
BOTTOM_MARGIN = 0.58 * inch
GUTTER = 0.22 * inch
FULL_WIDTH = PAGE_WIDTH - LEFT_MARGIN - RIGHT_MARGIN
COLUMN_WIDTH = (FULL_WIDTH - GUTTER) / 2

INK = colors.HexColor("#111111")
MUTED = colors.HexColor("#555555")
RULE = colors.HexColor("#c9c9c9")
LINK = colors.HexColor("#0645ad")


@dataclass
class FrontMatter:
    title: str
    meta: dict[str, str]
    abstract: list[str]
    keywords: str
    main_markdown: str


def make_styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "TopNote": ParagraphStyle(
            "TopNote",
            parent=base["BodyText"],
            fontName="Times-Italic",
            fontSize=7.4,
            leading=8.4,
            textColor=INK,
            alignment=TA_LEFT,
        ),
        "Title": ParagraphStyle(
            "Title",
            parent=base["Title"],
            fontName="Times-Roman",
            fontSize=21.5,
            leading=24.2,
            textColor=INK,
            alignment=TA_CENTER,
            spaceAfter=14,
        ),
        "Author": ParagraphStyle(
            "Author",
            parent=base["BodyText"],
            fontName="Times-Roman",
            fontSize=9.1,
            leading=10.3,
            alignment=TA_CENTER,
            textColor=INK,
        ),
        "Abstract": ParagraphStyle(
            "Abstract",
            parent=base["BodyText"],
            fontName="Times-Roman",
            fontSize=8.2,
            leading=9.2,
            alignment=TA_JUSTIFY,
            textColor=INK,
            firstLineIndent=0,
            spaceAfter=6,
        ),
        "Keywords": ParagraphStyle(
            "Keywords",
            parent=base["BodyText"],
            fontName="Times-Roman",
            fontSize=8.2,
            leading=9.2,
            alignment=TA_JUSTIFY,
            textColor=INK,
            spaceAfter=8,
        ),
        "H2": ParagraphStyle(
            "H2",
            parent=base["Heading2"],
            fontName="Times-Roman",
            fontSize=10.2,
            leading=11.8,
            alignment=TA_CENTER,
            textColor=INK,
            spaceBefore=8,
            spaceAfter=4,
            keepWithNext=True,
        ),
        "H3": ParagraphStyle(
            "H3",
            parent=base["Heading3"],
            fontName="Times-Bold",
            fontSize=8.7,
            leading=10.0,
            alignment=TA_LEFT,
            textColor=INK,
            spaceBefore=5,
            spaceAfter=2,
            keepWithNext=True,
        ),
        "Body": ParagraphStyle(
            "Body",
            parent=base["BodyText"],
            fontName="Times-Roman",
            fontSize=8.35,
            leading=9.65,
            alignment=TA_JUSTIFY,
            textColor=INK,
            spaceAfter=4.0,
            firstLineIndent=0.16 * inch,
        ),
        "BodyNoIndent": ParagraphStyle(
            "BodyNoIndent",
            parent=base["BodyText"],
            fontName="Times-Roman",
            fontSize=8.35,
            leading=9.65,
            alignment=TA_JUSTIFY,
            textColor=INK,
            spaceAfter=4.0,
            firstLineIndent=0,
        ),
        "Bullet": ParagraphStyle(
            "Bullet",
            parent=base["BodyText"],
            fontName="Times-Roman",
            fontSize=8.05,
            leading=9.2,
            alignment=TA_LEFT,
            textColor=INK,
        ),
        "Quote": ParagraphStyle(
            "Quote",
            parent=base["BodyText"],
            fontName="Times-Italic",
            fontSize=8.1,
            leading=9.3,
            alignment=TA_JUSTIFY,
            textColor=INK,
            leftIndent=9,
            rightIndent=5,
            borderColor=RULE,
            borderWidth=0.35,
            borderPadding=4,
            spaceBefore=2,
            spaceAfter=5,
        ),
        "Caption": ParagraphStyle(
            "Caption",
            parent=base["BodyText"],
            fontName="Times-Roman",
            fontSize=7.2,
            leading=8.0,
            alignment=TA_CENTER,
            textColor=INK,
            spaceAfter=5,
        ),
        "Code": ParagraphStyle(
            "Code",
            parent=base["Code"],
            fontName="Courier",
            fontSize=5.2,
            leading=6.0,
            textColor=INK,
            leftIndent=3,
            rightIndent=3,
            spaceBefore=2,
            spaceAfter=4,
        ),
        "TableCell": ParagraphStyle(
            "TableCell",
            parent=base["BodyText"],
            fontName="Times-Roman",
            fontSize=5.1,
            leading=5.8,
            alignment=TA_LEFT,
            textColor=INK,
            wordWrap="CJK",
        ),
        "TableHeader": ParagraphStyle(
            "TableHeader",
            parent=base["BodyText"],
            fontName="Times-Bold",
            fontSize=5.2,
            leading=5.9,
            alignment=TA_CENTER,
            textColor=INK,
            wordWrap="CJK",
        ),
    }


STYLES = make_styles()


def inline_markup(text: str) -> str:
    text = html.escape(text)

    def link_repl(match: re.Match[str]) -> str:
        label, url = match.groups()
        return f'<link href="{html.escape(url)}" color="#0645ad">{label}</link>'

    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", link_repl, text)
    text = re.sub(r"`([^`]+)`", r'<font name="Courier">\1</font>', text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"\*([^*]+)\*", r"<i>\1</i>", text)
    return text


def split_front_matter(markdown: str) -> FrontMatter:
    lines = markdown.splitlines()
    title = ""
    meta: dict[str, str] = {}
    abstract: list[str] = []
    keywords = ""

    i = 0
    if i < len(lines) and lines[i].startswith("# "):
        title = lines[i][2:].strip()
        i += 1

    while i < len(lines):
        stripped = lines[i].strip()
        meta_match = re.match(r"^\*\*([^*]+):\*\*\s*(.*)$", stripped)
        if meta_match:
            label, value = meta_match.groups()
            meta[label.strip().lower()] = value.strip()
            i += 1
            continue
        if stripped == "## Abstract":
            i += 1
            break
        i += 1

    para: list[str] = []
    while i < len(lines):
        stripped = lines[i].strip()
        if stripped == "## Keywords":
            if para:
                abstract.append(" ".join(para).strip())
                para.clear()
            i += 1
            break
        if not stripped:
            if para:
                abstract.append(" ".join(para).strip())
                para.clear()
        else:
            para.append(stripped)
        i += 1

    while i < len(lines):
        stripped = lines[i].strip()
        if stripped.startswith("## "):
            break
        if stripped:
            keywords = stripped
        i += 1

    return FrontMatter(
        title=title,
        meta=meta,
        abstract=abstract,
        keywords=keywords,
        main_markdown="\n".join(lines[i:]).strip(),
    )


def roman(num: int) -> str:
    vals = [
        (10, "X"),
        (9, "IX"),
        (5, "V"),
        (4, "IV"),
        (1, "I"),
    ]
    out = ""
    for value, glyph in vals:
        while num >= value:
            out += glyph
            num -= value
    return out


def clean_heading_number(text: str) -> str:
    return re.sub(r"^\d+(?:\.\d+)?\.?\s*", "", text).strip()


class HeadingState:
    def __init__(self) -> None:
        self.section = 0
        self.subsection = 0

    def h2(self, text: str) -> str:
        lower = text.lower()
        if lower.startswith("references"):
            return "REFERENCES"
        if lower.startswith("appendix"):
            return text.upper()
        self.section += 1
        self.subsection = 0
        return f"{roman(self.section)}. {clean_heading_number(text).upper()}"

    def h3(self, text: str) -> str:
        self.subsection += 1
        letter = chr(ord("A") + self.subsection - 1)
        return f"{letter}. {clean_heading_number(text)}"


def header_story(front: FrontMatter) -> list:
    author = front.meta.get("author", "Sai Shashank Narang")
    affiliation = front.meta.get(
        "affiliation", "School of Computer Science and Engineering"
    )
    institution = front.meta.get("institution", "Lovely Professional University")
    location = front.meta.get("location", "Phagwara, India")
    email = front.meta.get("email", "saishashanknarang@gmail.com")

    return [
        Spacer(1, 0.20 * inch),
        Paragraph(inline_markup(front.title), STYLES["Title"]),
        Paragraph(inline_markup(author), STYLES["Author"]),
        Paragraph(inline_markup(affiliation), STYLES["Author"]),
        Paragraph(inline_markup(institution), STYLES["Author"]),
        Paragraph(inline_markup(location), STYLES["Author"]),
        Paragraph(inline_markup(email), STYLES["Author"]),
    ]


def abstract_story(front: FrontMatter) -> list:
    abstract = " ".join(front.abstract).strip()
    story = [
        Paragraph("<b><i>Abstract</i></b>&mdash;" + inline_markup(abstract), STYLES["Abstract"])
    ]
    if front.keywords:
        story.append(
            Paragraph("<b><i>Keywords</i></b>&mdash;" + inline_markup(front.keywords), STYLES["Keywords"])
        )
    return story


def add_paragraph(story: list, lines: list[str], no_indent: bool = False) -> None:
    if not lines:
        return
    text = " ".join(line.strip() for line in lines).strip()
    if text:
        style = STYLES["BodyNoIndent"] if no_indent else STYLES["Body"]
        story.append(Paragraph(inline_markup(text), style))
    lines.clear()


def append_list(story: list, items: Iterable[str], numbered: bool = False) -> None:
    for idx, item in enumerate(items, start=1):
        marker = f"{idx}." if numbered else "-"
        story.append(
            Paragraph(
                inline_markup(f"{marker} {item}"),
                ParagraphStyle(
                    "InlineList",
                    parent=STYLES["Bullet"],
                    leftIndent=8,
                    firstLineIndent=-8,
                    spaceAfter=1.4,
                ),
            )
        )


def table_widths(rows: list[list[str]], max_cols: int) -> list[float]:
    weights = []
    for col in range(max_cols):
        sample = " ".join(row[col] if col < len(row) else "" for row in rows[:5])
        if len(sample) < 12 or re.search(r"\d|%", sample):
            weights.append(0.72)
        elif len(sample) > 55:
            weights.append(1.55)
        else:
            weights.append(1.05)
    total = sum(weights)
    return [COLUMN_WIDTH * w / total for w in weights]


def parse_table(lines: list[str]) -> Table:
    rows: list[list[str]] = []
    for line in lines:
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if all(set(cell) <= {"-", ":", " "} for cell in cells):
            continue
        rows.append(cells)
    max_cols = max(len(row) for row in rows)
    for row in rows:
        while len(row) < max_cols:
            row.append("")

    data = []
    for idx, row in enumerate(rows):
        style = STYLES["TableHeader"] if idx == 0 else STYLES["TableCell"]
        data.append([Paragraph(inline_markup(cell), style) for cell in row])

    table = Table(data, colWidths=table_widths(rows, max_cols), repeatRows=1, hAlign="CENTER")
    table.setStyle(
        TableStyle(
            [
                ("GRID", (0, 0), (-1, -1), 0.25, colors.black),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 1.7),
                ("RIGHTPADDING", (0, 0), (-1, -1), 1.7),
                ("TOPPADDING", (0, 0), (-1, -1), 2.0),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 2.0),
            ]
        )
    )
    return table


def parse_image(line: str):
    match = re.match(r"!\[([^\]]*)\]\(([^)]+)\)", line.strip())
    if not match:
        return None
    alt, path = match.groups()
    image_path = (ROOT / path).resolve()
    if not image_path.exists():
        return Paragraph(f"<i>Missing image: {html.escape(path)}</i>", STYLES["Caption"])

    img = Image(str(image_path))
    max_width = COLUMN_WIDTH * 0.94
    if img.drawWidth > max_width:
        ratio = max_width / img.drawWidth
        img.drawWidth *= ratio
        img.drawHeight *= ratio
    return KeepTogether([img, Spacer(1, 3), Paragraph("Fig. " + inline_markup(alt), STYLES["Caption"])])


def format_code_block(lines: list[str]) -> str:
    wrapped: list[str] = []
    for line in lines:
        if len(line) <= 54:
            wrapped.append(line)
            continue
        indent = len(line) - len(line.lstrip(" "))
        prefix = " " * min(indent + 2, 8)
        wrapped.extend(
            textwrap.wrap(
                line,
                width=54,
                subsequent_indent=prefix,
                break_long_words=True,
                break_on_hyphens=False,
            )
        )
    return "\n".join(wrapped)


def build_body(markdown: str) -> list:
    story: list = []
    heading_state = HeadingState()
    paragraph_lines: list[str] = []
    table_lines: list[str] = []
    code_lines: list[str] = []
    bullet_lines: list[str] = []
    numbered_lines: list[str] = []
    in_code = False

    def flush_bullets() -> None:
        nonlocal bullet_lines, numbered_lines
        if bullet_lines:
            append_list(story, bullet_lines, numbered=False)
            bullet_lines = []
        if numbered_lines:
            append_list(story, numbered_lines, numbered=True)
            numbered_lines = []

    def flush_table() -> None:
        nonlocal table_lines
        if table_lines:
            story.append(KeepTogether([parse_table(table_lines)]))
            story.append(Spacer(1, 4))
            table_lines = []

    for raw in markdown.splitlines():
        line = raw.rstrip()
        stripped = line.strip()

        if stripped.startswith("```"):
            add_paragraph(story, paragraph_lines)
            flush_bullets()
            flush_table()
            if in_code:
                story.append(Preformatted(format_code_block(code_lines), STYLES["Code"]))
                code_lines = []
                in_code = False
            else:
                in_code = True
            continue

        if in_code:
            code_lines.append(line)
            continue

        if not stripped:
            add_paragraph(story, paragraph_lines)
            flush_bullets()
            flush_table()
            continue

        if stripped.startswith("|") and stripped.endswith("|"):
            add_paragraph(story, paragraph_lines)
            flush_bullets()
            table_lines.append(line)
            continue
        flush_table()

        image = parse_image(stripped)
        if image is not None:
            add_paragraph(story, paragraph_lines)
            flush_bullets()
            story.append(image)
            continue

        if stripped.startswith("- "):
            add_paragraph(story, paragraph_lines)
            numbered_lines = []
            bullet_lines.append(stripped[2:].strip())
            continue

        numbered_match = re.match(r"^\d+\. (.*)", stripped)
        if numbered_match:
            add_paragraph(story, paragraph_lines)
            bullet_lines = []
            numbered_lines.append(numbered_match.group(1).strip())
            continue

        flush_bullets()

        if stripped.startswith(">"):
            add_paragraph(story, paragraph_lines)
            story.append(Paragraph(inline_markup(stripped.lstrip("> ").strip()), STYLES["Quote"]))
            continue

        if stripped.startswith("## "):
            add_paragraph(story, paragraph_lines)
            heading = heading_state.h2(stripped[3:].strip())
            story.append(Paragraph(inline_markup(heading), STYLES["H2"]))
            if heading == "REFERENCES":
                story.append(Spacer(1, 1))
            continue

        if stripped.startswith("### "):
            add_paragraph(story, paragraph_lines)
            story.append(Paragraph(inline_markup(heading_state.h3(stripped[4:].strip())), STYLES["H3"]))
            continue

        paragraph_lines.append(stripped)

    add_paragraph(story, paragraph_lines)
    flush_bullets()
    flush_table()
    return story


def on_first_page(canvas, doc) -> None:
    canvas.saveState()
    canvas.setFont("Times-Italic", 6.3)
    canvas.setFillColor(INK)
    canvas.drawString(LEFT_MARGIN, PAGE_HEIGHT - 0.24 * inch, "Capstone Research Manuscript")
    canvas.setFont("Times-Roman", 6.3)
    canvas.drawString(LEFT_MARGIN, 0.28 * inch, "School of Computer Science and Engineering, Lovely Professional University, Phagwara, India")
    canvas.drawCentredString(PAGE_WIDTH / 2, 0.28 * inch, str(doc.page))
    canvas.restoreState()


def on_later_page(canvas, doc) -> None:
    canvas.saveState()
    canvas.setFont("Times-Roman", 6.3)
    canvas.setFillColor(MUTED)
    canvas.drawString(LEFT_MARGIN, 0.28 * inch, "School of Computer Science and Engineering, Lovely Professional University, Phagwara, India")
    canvas.drawCentredString(PAGE_WIDTH / 2, 0.28 * inch, str(doc.page))
    canvas.restoreState()


def build_doc() -> BaseDocTemplate:
    doc = BaseDocTemplate(
        str(OUTPUT),
        pagesize=A4,
        leftMargin=LEFT_MARGIN,
        rightMargin=RIGHT_MARGIN,
        topMargin=TOP_MARGIN,
        bottomMargin=BOTTOM_MARGIN,
        title="Evaluating LLM Agent Teams in Real-World-Inspired Scenarios",
        author="Sai Shashank Narang",
    )

    first_header_height = 2.70 * inch
    first_header_y = PAGE_HEIGHT - TOP_MARGIN - first_header_height
    body_top = first_header_y - 0.04 * inch
    body_height = body_top - BOTTOM_MARGIN

    first_frames = [
        Frame(LEFT_MARGIN, first_header_y, FULL_WIDTH, first_header_height, id="first_header", showBoundary=0),
        Frame(LEFT_MARGIN, BOTTOM_MARGIN, COLUMN_WIDTH, body_height, id="first_left", showBoundary=0),
        Frame(LEFT_MARGIN + COLUMN_WIDTH + GUTTER, BOTTOM_MARGIN, COLUMN_WIDTH, body_height, id="first_right", showBoundary=0),
    ]
    later_height = PAGE_HEIGHT - TOP_MARGIN - BOTTOM_MARGIN
    later_frames = [
        Frame(LEFT_MARGIN, BOTTOM_MARGIN, COLUMN_WIDTH, later_height, id="left", showBoundary=0),
        Frame(LEFT_MARGIN + COLUMN_WIDTH + GUTTER, BOTTOM_MARGIN, COLUMN_WIDTH, later_height, id="right", showBoundary=0),
    ]
    doc.addPageTemplates(
        [
            PageTemplate(id="First", frames=first_frames, onPage=on_first_page, autoNextPageTemplate="TwoColumn"),
            PageTemplate(id="TwoColumn", frames=later_frames, onPage=on_later_page),
        ]
    )
    return doc


def main() -> int:
    front = split_front_matter(SOURCE.read_text(encoding="utf-8"))
    story = header_story(front)
    story.append(FrameBreak())
    story.extend(abstract_story(front))
    story.extend(build_body(front.main_markdown))
    build_doc().build(story)
    print(OUTPUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
