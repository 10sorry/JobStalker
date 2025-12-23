from __future__ import annotations

from typing import List

from .models import (
    ParsedVacancy,
    VacancySection,
    VacancySemantics,
    Confidence,
    SectionType,
)
from .normalizer import normalize_text
from .anchors import detect_section_anchors, SectionAnchor
from .content import (
    parse_section_content,
    merge_intro_with_about,
    split_long_paragraphs,
)
from .semantics import extract_semantics


def parse_vacancy(raw_text: str) -> ParsedVacancy:
    if not raw_text or not raw_text.strip():
        return ParsedVacancy(
            raw_text=raw_text or "",
            normalized_text="",
            sections=[],
            semantics=VacancySemantics(),
            overall_confidence=Confidence.LOW,
            warnings=["Empty input text"],
        )

    warnings: List[str] = []

    normalized = normalize_text(raw_text)

    if len(normalized) < 50:
        warnings.append("Very short text after normalization")

    anchors = detect_section_anchors(normalized)

    if not anchors:
        warnings.append("No section headers detected")

    sections = parse_section_content(normalized, anchors)

    # Post-processing
    sections = merge_intro_with_about(sections)
    sections = split_long_paragraphs(sections)

    semantics = extract_semantics(normalized, sections)

    overall_confidence = _calculate_overall_confidence(sections, anchors, warnings)

    warnings.extend(_validate_result(sections, semantics))

    return ParsedVacancy(
        raw_text=raw_text,
        normalized_text=normalized,
        sections=sections,
        semantics=semantics,
        overall_confidence=overall_confidence,
        warnings=warnings,
    )


def _calculate_overall_confidence(
    sections: List[VacancySection],
    anchors: List[SectionAnchor],
    warnings: List[str],
) -> Confidence:
    if not sections:
        return Confidence.LOW

    high_count = sum(1 for s in sections if s.confidence == Confidence.HIGH)
    medium_count = sum(1 for s in sections if s.confidence == Confidence.MEDIUM)
    low_count = sum(1 for s in sections if s.confidence == Confidence.LOW)

    has_requirements = any(s.type == SectionType.REQUIREMENTS for s in sections)
    has_responsibilities = any(s.type == SectionType.RESPONSIBILITIES for s in sections)
    has_key_sections = has_requirements or has_responsibilities

    total = len(sections)
    if total == 0:
        return Confidence.LOW

    confidence_score = (high_count * 3 + medium_count * 2 + low_count * 1) / (total * 3)

    if has_key_sections:
        confidence_score += 0.1

    if len(warnings) >= 3:
        confidence_score -= 0.2
    elif len(warnings) >= 1:
        confidence_score -= 0.1

    # Map to confidence level
    if confidence_score >= 0.7:
        return Confidence.HIGH
    elif confidence_score >= 0.4:
        return Confidence.MEDIUM
    else:
        return Confidence.LOW


def _validate_result(
    sections: List[VacancySection],
    semantics: VacancySemantics,
) -> List[str]:
    warnings: List[str] = []

    section_types = [s.type for s in sections]
    seen = set()
    for st in section_types:
        if st in seen and st != SectionType.INTRO:
            warnings.append(f"Duplicate section detected: {st.value}")
        seen.add(st)

    for section in sections:
        if section.type != SectionType.INTRO and len(section.content) == 0:
            warnings.append(f"Empty section: {section.type.value}")

    if semantics.salary:
        if semantics.salary.min and semantics.salary.max:
            if semantics.salary.min > semantics.salary.max:
                warnings.append("Invalid salary range (min > max)")

    return warnings


def parse_vacancy_to_dict(raw_text: str) -> dict:
    result = parse_vacancy(raw_text)
    return result.to_dict()


def format_for_display(parsed: ParsedVacancy) -> dict:
    sections_display = []

    section_display_info = {
        SectionType.INTRO: ("", ""),
        SectionType.REQUIREMENTS: ("Requirements", "📋"),
        SectionType.RESPONSIBILITIES: ("Responsibilities", "💼"),
        SectionType.NICE_TO_HAVE: ("Nice to Have", "⭐"),
        SectionType.TECH_STACK: ("Tech Stack", "🛠️"),
        SectionType.BENEFITS: ("Benefits", "🎁"),
        SectionType.ABOUT_COMPANY: ("About Company", "🏢"),
        SectionType.ABOUT_JOB: ("About Position", "📝"),
        SectionType.TEAM: ("Team", "👥"),
        SectionType.SALARY: ("Salary", "💰"),
        SectionType.EXPERIENCE: ("Experience", "📈"),
        SectionType.WORK_FORMAT: ("Work Format", "📍"),
        SectionType.CONTACT: ("Contact", "📧"),
    }

    for section in parsed.sections:
        display_title, icon = section_display_info.get(
            section.type, (section.type.value.replace("_", " ").title(), "📄")
        )

        items = []
        for item in section.content:
            items.append(
                {
                    "text": item.text,
                    "is_multiline": item.is_continuation,
                }
            )

        sections_display.append(
            {
                "type": section.type.value,
                "title": display_title,
                "icon": icon,
                "items": items,
                "confidence": section.confidence.value,
                "is_intro": section.type == SectionType.INTRO,
            }
        )

    salary_display = None
    if parsed.semantics.salary:
        salary_display = {
            "text": parsed.semantics.salary.to_display(),
            "min": parsed.semantics.salary.min,
            "max": parsed.semantics.salary.max,
            "currency": parsed.semantics.salary.currency,
        }

    # Format contacts for display
    contacts_display = []
    for contact in parsed.semantics.contacts:
        contacts_display.append(
            {
                "type": contact.type.value,
                "value": contact.value,
                "display": contact.display,
                "url": contact.url,
                "label": contact.label,
                "icon": contact.icon,
            }
        )

    return {
        "sections": sections_display,
        "semantics": {
            "salary": salary_display,
            "experience": parsed.semantics.experience_years,
            "remote": parsed.semantics.remote,
            "employment_type": parsed.semantics.employment_type,
            "technologies": parsed.semantics.technologies,
            "contacts": contacts_display,
            "job_title": parsed.semantics.job_title,
            "company_name": parsed.semantics.company_name,
        },
        "confidence": parsed.overall_confidence.value,
        "warnings": parsed.warnings,
        "needs_llm": parsed.needs_llm_clarification(),
    }
