from __future__ import annotations

import re
from typing import List, Optional, Set, Tuple

from .models import (
    Salary,
    SalaryPeriod,
    VacancySection,
    VacancySemantics,
    SectionType,
    ContactInfo,
    ContactType,
)
from .rules import (
    CURRENCY_PATTERNS,
    SALARY_PATTERNS,
    EXPERIENCE_PATTERNS,
    REMOTE_POSITIVE,
    REMOTE_NEGATIVE,
    HYBRID_INDICATORS,
    EMPLOYMENT_PATTERNS,
    TECH_KEYWORDS,
)


def extract_semantics(
    normalized_text: str,
    sections: List[VacancySection],
) -> VacancySemantics:
    text_lower = normalized_text.lower()

    salary = _extract_salary(normalized_text, sections)
    experience = _extract_experience(text_lower)
    remote = _extract_remote_status(text_lower)
    employment = _extract_employment_type(text_lower)
    technologies = _extract_technologies(text_lower, sections)
    contacts = _extract_contacts(normalized_text, sections)
    job_title = _extract_job_title(normalized_text, sections)
    company = _extract_company_name(normalized_text, sections)

    return VacancySemantics(
        salary=salary,
        experience_years=experience,
        remote=remote,
        employment_type=employment,
        technologies=technologies,
        contacts=contacts,
        job_title=job_title,
        company_name=company,
    )


def _extract_salary(
    text: str,
    sections: List[VacancySection],
) -> Optional[Salary]:
    salary_section = next((s for s in sections if s.type == SectionType.SALARY), None)

    if salary_section:
        result = _parse_salary_text(salary_section.raw_content)
        if result:
            return result

    benefits_section = next(
        (s for s in sections if s.type == SectionType.BENEFITS), None
    )

    if benefits_section:
        result = _parse_salary_text(benefits_section.raw_content)
        if result:
            return result

    # Search in full text
    return _parse_salary_text(text)


def _parse_salary_text(text: str) -> Optional[Salary]:
    text_lower = text.lower()

    for pattern, pattern_type in SALARY_PATTERNS:
        match = pattern.search(text_lower)
        if match:
            groups = match.groupdict()

            currency = _detect_currency(text_lower, match.group(0))

            min_val: Optional[int] = None
            max_val: Optional[int] = None

            if pattern_type == "range":
                min_val = _parse_salary_value(groups.get("min", ""))
                max_val = _parse_salary_value(groups.get("max", ""))
            elif pattern_type == "from":
                min_val = _parse_salary_value(groups.get("min", ""))
            elif pattern_type == "upto":
                max_val = _parse_salary_value(groups.get("max", ""))
            elif pattern_type == "single":
                val = _parse_salary_value(
                    groups.get("val", ""), multiplier=groups.get("mult", "")
                )
                min_val = max_val = val

            if min_val is not None or max_val is not None:
                gross = _detect_gross_net(text_lower)

                period = _detect_period(text_lower)

                return Salary(
                    min=min_val,
                    max=max_val,
                    currency=currency,
                    period=period,
                    gross=gross,
                    raw_text=match.group(0).strip(),
                )

    return None


def _parse_salary_value(value_str: str, multiplier: str = "") -> Optional[int]:
    if not value_str:
        return None

    # Remove spaces and commas
    clean = value_str.replace(" ", "").replace(",", "").replace(".", "")

    try:
        value = int(clean)
    except ValueError:
        return None

    # Apply multiplier (k, тыс, etc.)
    if multiplier:
        mult_lower = multiplier.lower()
        if mult_lower in ("k", "к", "т", "тыс", "тыс."):
            value *= 1000

    return value


def _detect_currency(text: str, context: str) -> str:
    for check_text in [context, text]:
        for currency, patterns in CURRENCY_PATTERNS.items():
            for pattern in patterns:
                if pattern in check_text.lower():
                    return currency

    if "$" in context:
        return "USD"
    return "RUB"


def _detect_gross_net(text: str) -> Optional[bool]:
    gross_patterns = ["gross", "до вычета", "до налог", "гросс", "брутто"]
    net_patterns = ["net", "на руки", "после налог", "нетто", "чистыми"]

    for pattern in gross_patterns:
        if pattern in text:
            return True

    for pattern in net_patterns:
        if pattern in text:
            return False

    return None


def _detect_period(text: str) -> SalaryPeriod:
    year_patterns = ["в год", "годовой", "per year", "annual", "/year", "/y"]
    hour_patterns = ["в час", "per hour", "/hour", "/h"]

    for pattern in year_patterns:
        if pattern in text:
            return SalaryPeriod.YEAR

    for pattern in hour_patterns:
        if pattern in text:
            return SalaryPeriod.HOUR

    return SalaryPeriod.MONTH


def _extract_experience(text: str) -> Optional[Tuple[int, int]]:
    for pattern in EXPERIENCE_PATTERNS:
        match = pattern.search(text)
        if match:
            groups = match.groups()

            if len(groups) == 2:
                try:
                    min_years = int(groups[0])
                    max_years = int(groups[1])
                    return (min_years, max_years)
                except (ValueError, TypeError):
                    pass
            elif len(groups) == 1:
                try:
                    years = int(groups[0])
                    if "+" in match.group(0):
                        return (years, 99)
                    return (years, years)
                except (ValueError, TypeError):
                    pass

    return None


def _extract_remote_status(text: str) -> Optional[bool]:
    text = text.lower()

    for indicator in HYBRID_INDICATORS:
        if indicator in text:
            return None

    remote_count = sum(1 for r in REMOTE_POSITIVE if r in text)
    office_count = sum(1 for o in REMOTE_NEGATIVE if o in text)

    if remote_count > 0 and office_count == 0:
        return True
    elif office_count > 0 and remote_count == 0:
        return False

    return None


def _extract_employment_type(text: str) -> Optional[str]:
    for emp_type, keywords in EMPLOYMENT_PATTERNS.items():
        for keyword in keywords:
            if keyword in text:
                return emp_type

    return None


def _extract_technologies(
    text: str,
    sections: List[VacancySection],
) -> List[str]:
    found: Set[str] = set()

    priority_sections = [
        SectionType.TECH_STACK,
        SectionType.REQUIREMENTS,
        SectionType.NICE_TO_HAVE,
    ]

    for section_type in priority_sections:
        section = next((s for s in sections if s.type == section_type), None)
        if section:
            _find_tech_in_text(section.raw_content.lower(), found)

    if len(found) < 3:
        _find_tech_in_text(text, found)

    return sorted(found)


def _find_tech_in_text(text: str, found: Set[str]) -> None:
    for tech in TECH_KEYWORDS:
        if " " in tech:
            if tech in text:
                found.add(tech)
        else:
            pattern = rf"\b{re.escape(tech)}\b"
            if re.search(pattern, text, re.IGNORECASE):
                found.add(tech)


def _extract_contacts(
    text: str,
    sections: List[VacancySection],
) -> List[ContactInfo]:
    contacts: List[ContactInfo] = []
    seen_values: Set[str] = set()

    def add_contact(contact_type: ContactType, value: str, label: Optional[str] = None):
        normalized = value.lower().strip()
        if normalized not in seen_values:
            seen_values.add(normalized)
            contacts.append(
                ContactInfo(
                    type=contact_type,
                    value=value.strip(),
                    label=label,
                )
            )

    contact_section = next((s for s in sections if s.type == SectionType.CONTACT), None)
    search_text = contact_section.raw_content if contact_section else text

    texts_to_search = [search_text]
    if contact_section and search_text != text:
        texts_to_search.append(text)

    for search_in in texts_to_search:
        email_pattern = r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"
        for email in re.findall(email_pattern, search_in):
            if not any(
                fp in email.lower() for fp in ["example.com", "test.com", "domain.com"]
            ):
                add_contact(ContactType.EMAIL, email)

        # Telegram usernames: @username
        tg_pattern = r"@([a-zA-Z][a-zA-Z0-9_]{4,31})"
        for match in re.finditer(tg_pattern, search_in):
            username = match.group(0)
            context_start = max(0, match.start() - 30)
            context = search_in[context_start : match.start()].lower()
            label = None
            if "hr" in context or "рекрутер" in context or "recruiter" in context:
                label = "HR"
            elif "lead" in context or "руководитель" in context:
                label = "Lead"
            add_contact(ContactType.TELEGRAM, username, label)

        # t.me links
        tme_pattern = r"(?:https?://)?t\.me/([a-zA-Z0-9_]+)"
        for match in re.finditer(tme_pattern, search_in):
            username = f"@{match.group(1)}"
            add_contact(ContactType.TELEGRAM, username)

        # Phone numbers (international formats)
        phone_patterns = [
            r"\+7[\s\-]?\(?\d{3}\)?[\s\-]?\d{3}[\s\-]?\d{2}[\s\-]?\d{2}",  # Russian +7
            r"\+380[\s\-]?\(?\d{2}\)?[\s\-]?\d{3}[\s\-]?\d{2}[\s\-]?\d{2}",  # Ukrainian +380
            r"\+1[\s\-]?\(?\d{3}\)?[\s\-]?\d{3}[\s\-]?\d{4}",  # US/Canada +1
            r"\+\d{1,3}[\s\-]?\(?\d{2,4}\)?[\s\-]?\d{3,4}[\s\-]?\d{2,4}",  # Generic international
            r"8[\s\-]?\(?\d{3}\)?[\s\-]?\d{3}[\s\-]?\d{2}[\s\-]?\d{2}",  # Russian 8-xxx
        ]
        for pattern in phone_patterns:
            for phone in re.findall(pattern, search_in):
                # Clean up the phone number
                clean = re.sub(r"[\s\-\(\)]", "", phone)
                if len(clean) >= 10:
                    add_contact(ContactType.PHONE, phone)

        # WhatsApp links
        wa_pattern = (
            r"(?:https?://)?(?:wa\.me|api\.whatsapp\.com/send\?phone=)/?(\+?\d+)"
        )
        for match in re.finditer(wa_pattern, search_in):
            phone = match.group(1)
            add_contact(ContactType.WHATSAPP, phone)

        # LinkedIn profiles
        li_patterns = [
            r"(?:https?://)?(?:www\.)?linkedin\.com/in/([a-zA-Z0-9\-]+)",
            r"(?:https?://)?(?:www\.)?linkedin\.com/company/([a-zA-Z0-9\-]+)",
        ]
        for pattern in li_patterns:
            for match in re.finditer(pattern, search_in):
                add_contact(ContactType.LINKEDIN, match.group(0))

        # Application forms (Google Forms, Typeform, Airtable, etc.)
        form_patterns = [
            r"(?:https?://)?(?:docs\.google\.com/forms/[^\s<>\"]+)",
            r"(?:https?://)?(?:forms\.gle/[a-zA-Z0-9]+)",
            r"(?:https?://)?(?:[a-zA-Z0-9-]+\.typeform\.com/to/[a-zA-Z0-9]+)",
            r"(?:https?://)?(?:airtable\.com/[^\s<>\"]+)",
            r"(?:https?://)?(?:tally\.so/r/[a-zA-Z0-9]+)",
            r"(?:https?://)?(?:forms\.yandex\.[a-z]+/[^\s<>\"]+)",
            r"(?:https?://)?(?:surveymonkey\.com/r/[a-zA-Z0-9]+)",
        ]
        for pattern in form_patterns:
            for match in re.findall(pattern, search_in, re.IGNORECASE):
                add_contact(ContactType.FORM, match)

        # Skype
        skype_pattern = r"(?:skype|скайп)[\s:]+([a-zA-Z][a-zA-Z0-9._\-]{5,31})"
        for match in re.finditer(skype_pattern, search_in, re.IGNORECASE):
            add_contact(ContactType.SKYPE, match.group(1))

        # Discord (username#1234 or server invites)
        discord_patterns = [
            r"([a-zA-Z0-9_]{2,32}#\d{4})",  # Username#discriminator
            r"(?:https?://)?discord\.gg/([a-zA-Z0-9]+)",  # Server invite
            r"(?:https?://)?discord\.com/invite/([a-zA-Z0-9]+)",
        ]
        for pattern in discord_patterns:
            for match in re.findall(pattern, search_in):
                value = match if "#" in match else f"discord.gg/{match}"
                add_contact(ContactType.DISCORD, value)

        # Generic career/jobs page links (in contact context)
        if (
            contact_section
            or "откликнуться" in search_in.lower()
            or "apply" in search_in.lower()
        ):
            career_pattern = r"(?:https?://)?(?:www\.)?[a-zA-Z0-9\-]+\.[a-zA-Z]{2,}/(?:careers?|jobs?|vacancies|вакансии)[^\s<>\"]*"
            for match in re.findall(career_pattern, search_in, re.IGNORECASE):
                add_contact(ContactType.WEBSITE, match)

    return contacts


def _extract_job_title(
    text: str,
    sections: List[VacancySection],
) -> Optional[str]:
    lines = [l.strip() for l in text.split("\n") if l.strip()]

    if not lines:
        return None

    first_line = lines[0]
    if len(first_line) < 80 and not first_line.endswith(":"):
        title_indicators = [
            "developer",
            "engineer",
            "designer",
            "manager",
            "analyst",
            "разработчик",
            "инженер",
            "дизайнер",
            "менеджер",
            "аналитик",
            "специалист",
            "ведущий",
            "senior",
            "junior",
            "middle",
            "lead",
        ]
        if any(ind in first_line.lower() for ind in title_indicators):
            return first_line

    # Look for explicit title patterns
    title_pattern = r"(?:вакансия|позиция|position|vacancy|job title)[\s:]+(.+)"
    match = re.search(title_pattern, text, re.IGNORECASE)
    if match:
        return match.group(1).strip()[:100]  # Limit length

    return None


def _extract_company_name(
    text: str,
    sections: List[VacancySection],
) -> Optional[str]:
    about = next((s for s in sections if s.type == SectionType.ABOUT_COMPANY), None)

    if about:
        company_pattern = (
            r"(?:мы\s*[-—]\s*|компания\s+)([A-ZА-ЯЁ][a-zA-Zа-яА-ЯёЁ\s&.]+)"
        )
        match = re.search(company_pattern, about.raw_content)
        if match:
            return match.group(1).strip()[:80]

    employer_pattern = r"(?:работодатель|компания|company|employer)[\s:]+([A-ZА-ЯЁ][a-zA-Zа-яА-ЯёЁ\s&.]+)"
    match = re.search(employer_pattern, text, re.IGNORECASE)
    if match:
        return match.group(1).strip()[:80]

    return None
