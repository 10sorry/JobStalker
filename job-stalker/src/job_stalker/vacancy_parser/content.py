from __future__ import annotations

from typing import List, Tuple

from .models import ListItem, VacancySection, SectionType, Confidence
from .anchors import SectionAnchor, get_section_boundaries
from .normalizer import (
    get_lines,
    is_empty_line,
    is_bullet_line,
    get_bullet_content,
    get_indentation,
)


def parse_section_content(
    normalized_text: str,
    anchors: List[SectionAnchor],
) -> List[VacancySection]:
    lines = get_lines(normalized_text)
    total_lines = len(lines)
    sections: List[VacancySection] = []

    if anchors:
        first_anchor_line = anchors[0].line_number
        if first_anchor_line > 0:
            intro_lines = lines[:first_anchor_line]
            intro_content = _parse_lines_to_items(intro_lines)
            if intro_content:
                sections.append(
                    VacancySection(
                        type=SectionType.INTRO,
                        title="",
                        content=intro_content,
                        raw_content="\n".join(intro_lines),
                        confidence=Confidence.HIGH,
                        start_line=0,
                        end_line=first_anchor_line - 1,
                    )
                )
    else:
        content = _parse_lines_to_items(lines)
        if content:
            sections.append(
                VacancySection(
                    type=SectionType.INTRO,
                    title="",
                    content=content,
                    raw_content=normalized_text,
                    confidence=Confidence.LOW,
                    start_line=0,
                    end_line=total_lines - 1,
                )
            )
        return sections

    # Parse each section
    boundaries = get_section_boundaries(anchors, total_lines)

    for anchor, start_line, end_line in boundaries:
        content_start = start_line + 1
        section_lines = lines[content_start : end_line + 1]

        while section_lines and is_empty_line(section_lines[0]):
            section_lines.pop(0)
            content_start += 1

        while section_lines and is_empty_line(section_lines[-1]):
            section_lines.pop()

        content = _parse_lines_to_items(section_lines)

        sections.append(
            VacancySection(
                type=anchor.section_type,
                title=anchor.line_text,
                content=content,
                raw_content="\n".join(section_lines),
                confidence=anchor.confidence,
                start_line=start_line,
                end_line=end_line,
            )
        )

    return sections


def _parse_lines_to_items(lines: List[str]) -> List[ListItem]:
    if not lines:
        return []
    items: List[ListItem] = []
    current_item: List[str] = []
    current_is_bullet = False
    current_indent = 0

    for line in lines:
        stripped = line.strip()

        if not stripped:
            if current_item:
                items.append(_create_item(current_item, current_is_bullet))
                current_item = []
                current_is_bullet = False
            continue

        line_indent = get_indentation(line)
        is_bullet = is_bullet_line(line)

        # New bullet item
        if is_bullet:
            if current_item:
                items.append(_create_item(current_item, current_is_bullet))

            current_item = [get_bullet_content(line)]
            current_is_bullet = True
            current_indent = line_indent + 2  # "- " = 2 chars

        elif current_is_bullet and _is_continuation(
            line, stripped, line_indent, current_indent
        ):
            current_item.append(stripped)

        else:
            if current_item and current_is_bullet:
                items.append(_create_item(current_item, current_is_bullet))
                current_item = []
                current_is_bullet = False

            if current_item and not current_is_bullet:
                current_item.append(stripped)
            else:
                current_item = [stripped]
                current_is_bullet = False
                current_indent = line_indent

    if current_item:
        items.append(_create_item(current_item, current_is_bullet))

    return items


def _is_continuation(
    line: str,
    stripped: str,
    line_indent: int,
    expected_indent: int,
) -> bool:
    if line_indent < expected_indent - 2:  # Allow some tolerance
        if line_indent == 0 and expected_indent > 4:
            return False

    if len(stripped) < 30 and (stripped.endswith(":") or stripped.endswith("：")):
        return False

    if stripped.startswith("- "):
        return False

    if len(stripped) < 15 and stripped.endswith(":"):
        return False

    return True


def _create_item(lines: List[str], is_bullet: bool) -> ListItem:
    text = "\n".join(lines)
    return ListItem(text=text, is_continuation=len(lines) > 1)


def merge_intro_with_about(sections: List[VacancySection]) -> List[VacancySection]:
    intro = None
    has_about_company = False

    for section in sections:
        if section.type == SectionType.INTRO:
            intro = section
        elif section.type == SectionType.ABOUT_COMPANY:
            has_about_company = True

    if intro and not has_about_company and _looks_like_about_company(intro):
        intro.type = SectionType.ABOUT_COMPANY
        intro.confidence = Confidence.MEDIUM

    return sections


def _looks_like_about_company(section: VacancySection) -> bool:
    text = section.raw_content.lower()

    company_signals = [
        "компани",
        "команд",
        "мы ",
        "наш",
        "company",
        "team",
        "we ",
        "разрабатыва",
        "занимаемся",
        "специализир",
    ]

    non_company_signals = [
        "требован",
        "обязанност",
        "задач",
        "requirements",
        "responsibilities",
        "опыт работы",
        "навыки",
        "skills",
    ]

    company_score = sum(1 for s in company_signals if s in text)
    non_company_score = sum(1 for s in non_company_signals if s in text)

    return company_score >= 2 and non_company_score == 0 and len(section.content) < 5


def split_long_paragraphs(sections: List[VacancySection]) -> List[VacancySection]:
    for section in sections:
        new_content: List[ListItem] = []

        for item in section.content:
            if not item.is_continuation and len(item.text) > 200:
                sentences = _split_sentences(item.text)
                if len(sentences) > 1:
                    for sentence in sentences:
                        if sentence.strip():
                            new_content.append(ListItem(text=sentence.strip()))
                else:
                    new_content.append(item)
            else:
                new_content.append(item)

        section.content = new_content

    return sections


def _split_sentences(text: str) -> List[str]:
    import re

    pattern = r"(?<=[.!?])\s+(?=[A-ZА-ЯЁ])"

    parts = re.split(pattern, text)

    return [p for p in parts if len(p) > 20]
