from __future__ import annotations


from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class SectionType(str, Enum):
    REQUIREMENTS = "requirements"
    RESPONSIBILITIES = "responsibilities"
    NICE_TO_HAVE = "nice_to_have"
    TECH_STACK = "tech_stack"
    BENEFITS = "benefits"
    ABOUT_COMPANY = "about_company"
    ABOUT_JOB = "about_job"
    TEAM = "team"
    SALARY = "salary"
    EXPERIENCE = "experience"
    WORK_FORMAT = "work_format"
    CONTACT = "contact"
    INTRO = "intro"


class Confidence(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class SalaryPeriod(str, Enum):
    MONTH = "month"
    YEAR = "year"
    HOUR = "hour"
    PROJECT = "project"


class ListItem(BaseModel):
    text: str = Field(..., description="Full item text (may include newlines)")
    is_continuation: bool = Field(
        default=False, description="Whether this continues previous item"
    )

    def __str__(self) -> str:
        return self.text


class ContactType(str, Enum):
    EMAIL = "email"
    TELEGRAM = "telegram"
    PHONE = "phone"
    WHATSAPP = "whatsapp"
    LINKEDIN = "linkedin"
    WEBSITE = "website"
    FORM = "form"
    SKYPE = "skype"
    DISCORD = "discord"
    OTHER = "other"


class ContactInfo(BaseModel):
    type: ContactType = Field(..., description="Contact type")
    value: str = Field(..., description="Contact value (username, email, phone, URL)")
    display: str = Field(default="", description="Display-friendly text")
    url: Optional[str] = Field(None, description="Clickable URL if applicable")
    label: Optional[str] = Field(
        None, description="Optional label (e.g., 'HR', 'Recruiter')"
    )

    def __init__(self, **data):
        super().__init__(**data)
        if not self.display:
            self.display = self.value
        if not self.url:
            self.url = self._generate_url()

    def _generate_url(self) -> Optional[str]:
        if self.type == ContactType.EMAIL:
            return f"mailto:{self.value}"
        elif self.type == ContactType.TELEGRAM:
            username = self.value.lstrip("@").replace("t.me/", "")
            return f"https://t.me/{username}"
        elif self.type == ContactType.PHONE:
            clean_phone = "".join(c for c in self.value if c.isdigit() or c == "+")
            return f"tel:{clean_phone}"
        elif self.type == ContactType.WHATSAPP:
            clean_phone = "".join(c for c in self.value if c.isdigit())
            return f"https://wa.me/{clean_phone}"
        elif self.type == ContactType.LINKEDIN:
            if "linkedin.com" in self.value:
                return (
                    self.value
                    if self.value.startswith("http")
                    else f"https://{self.value}"
                )
            return f"https://linkedin.com/in/{self.value}"
        elif self.type == ContactType.SKYPE:
            return f"skype:{self.value}?chat"
        elif self.type == ContactType.DISCORD:
            return None  # Discord usernames aren't directly linkable
        elif self.type in (ContactType.WEBSITE, ContactType.FORM):
            return (
                self.value if self.value.startswith("http") else f"https://{self.value}"
            )
        return None

    @property
    def icon(self) -> str:
        icons = {
            ContactType.EMAIL: "📧",
            ContactType.TELEGRAM: "✈️",
            ContactType.PHONE: "📞",
            ContactType.WHATSAPP: "💬",
            ContactType.LINKEDIN: "💼",
            ContactType.WEBSITE: "🌐",
            ContactType.FORM: "📝",
            ContactType.SKYPE: "🔵",
            ContactType.DISCORD: "🎮",
            ContactType.OTHER: "📎",
        }
        return icons.get(self.type, "📎")


class Salary(BaseModel):
    min: Optional[int] = Field(None, description="Minimum salary")
    max: Optional[int] = Field(None, description="Maximum salary")
    currency: str = Field(default="USD", description="Currency code")
    period: SalaryPeriod = Field(
        default=SalaryPeriod.MONTH, description="Payment period"
    )
    gross: Optional[bool] = Field(
        None, description="True if gross, False if net, None if unknown"
    )
    raw_text: str = Field(default="", description="Original salary text")

    @property
    def is_range(self) -> bool:
        return self.min is not None and self.max is not None and self.min != self.max

    @property
    def single_value(self) -> Optional[int]:
        if self.min == self.max:
            return self.min
        return None

    def to_display(self) -> str:
        if self.min is None and self.max is None:
            return self.raw_text or "Not specified"

        currency_symbols = {"USD": "$", "EUR": "€", "RUB": "₽", "GBP": "£"}
        symbol = currency_symbols.get(self.currency, self.currency)

        if self.is_range:
            result = f"{symbol}{self.min:,} - {symbol}{self.max:,}"
        elif self.min:
            result = f"from {symbol}{self.min:,}"
        elif self.max:
            result = f"up to {symbol}{self.max:,}"
        else:
            return self.raw_text or "Not specified"

        if self.period != SalaryPeriod.MONTH:
            result += f"/{self.period.value}"

        if self.gross is True:
            result += " (gross)"
        elif self.gross is False:
            result += " (net)"

        return result


class VacancySemantics(BaseModel):
    salary: Optional[Salary] = Field(None, description="Parsed salary info")
    experience_years: Optional[tuple[int, int]] = Field(
        None, description="Min/max years range"
    )
    remote: Optional[bool] = Field(None, description="Is remote work available")
    relocation: Optional[bool] = Field(None, description="Relocation support")
    employment_type: Optional[str] = Field(
        None, description="full-time/part-time/contract"
    )
    technologies: List[str] = Field(
        default_factory=list, description="Detected tech stack"
    )
    languages: List[str] = Field(default_factory=list, description="Required languages")
    locations: List[str] = Field(default_factory=list, description="Work locations")
    company_name: Optional[str] = Field(None, description="Detected company name")
    job_title: Optional[str] = Field(None, description="Detected job title")
    contacts: List[ContactInfo] = Field(
        default_factory=list, description="Structured contact info"
    )


class VacancySection(BaseModel):
    type: SectionType = Field(..., description="Section type")
    title: str = Field(default="", description="Original section title from text")
    content: List[ListItem] = Field(
        default_factory=list, description="Section content as list items"
    )
    raw_content: str = Field(default="", description="Original unparsed content")
    confidence: Confidence = Field(
        default=Confidence.MEDIUM, description="Detection confidence"
    )
    start_line: int = Field(default=0, description="Start line in normalized text")
    end_line: int = Field(default=0, description="End line in normalized text")

    @property
    def text(self) -> str:
        return "\n".join(str(item) for item in self.content)

    @property
    def items(self) -> List[str]:
        return [str(item) for item in self.content]


class ParsedVacancy(BaseModel):
    raw_text: str = Field(..., description="Original input text")
    normalized_text: str = Field(default="", description="Text after normalization")
    sections: List[VacancySection] = Field(
        default_factory=list, description="Detected sections"
    )
    semantics: VacancySemantics = Field(
        default_factory=VacancySemantics, description="Extracted semantics"
    )
    overall_confidence: Confidence = Field(
        default=Confidence.MEDIUM, description="Overall parse confidence"
    )
    warnings: List[str] = Field(default_factory=list, description="Parsing warnings")

    def get_section(self, section_type: SectionType) -> Optional[VacancySection]:
        for section in self.sections:
            if section.type == section_type:
                return section
        return None

    def get_sections(self, section_type: SectionType) -> List[VacancySection]:
        return [s for s in self.sections if s.type == section_type]

    @property
    def requirements(self) -> Optional[VacancySection]:
        return self.get_section(SectionType.REQUIREMENTS)

    @property
    def responsibilities(self) -> Optional[VacancySection]:
        return self.get_section(SectionType.RESPONSIBILITIES)

    @property
    def benefits(self) -> Optional[VacancySection]:
        return self.get_section(SectionType.BENEFITS)

    @property
    def intro(self) -> Optional[VacancySection]:
        return self.get_section(SectionType.INTRO)

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump(mode="json")

    def needs_llm_clarification(self) -> bool:
        if self.overall_confidence == Confidence.LOW:
            return True
        low_confidence_sections = sum(
            1 for s in self.sections if s.confidence == Confidence.LOW
        )
        return low_confidence_sections > len(self.sections) / 2
