from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Type, TypeVar, Union
from pydantic import BaseModel, Field, field_validator, model_validator


class ModelType(str, Enum):
    MISTRAL = "mistral"
    GEMINI = "gemini"
    LLAMA = "llama"
    GROQ = "groq"


class PositionType(str, Enum):
    DEVELOPER = "developer"
    DESIGNER = "designer"
    ARTIST = "artist"
    MANAGER = "manager"
    QA = "qa"
    PRODUCER = "producer"
    HR = "hr"
    MARKETING = "marketing"
    OTHER = "other"


class ExperienceLevel(str, Enum):
    JUNIOR = "junior"
    MIDDLE = "middle"
    SENIOR = "senior"


class VacancySource(str, Enum):
    TELEGRAM = "telegram"
    LINKEDIN = "linkedin"
    HEADHUNTER = "headhunter"
    HABR = "habr"
    CUSTOM = "custom"


class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    ERROR = "error"
    CANCELLED = "cancelled"


class AppSettings(BaseModel):
    model_type: str = Field(default="mistral", description="AI model to use")
    days_back: int = Field(default=7, ge=1, le=365, description="Days to search back")
    custom_prompt: str = Field(default="", description="Custom filter prompt")
    resume_summary: str = Field(default="", description="Resume summary")
    channels: List[str] = Field(default_factory=list, description="Telegram channels to monitor")
    enable_stage2: bool = Field(default=False, description="Run recruiter analysis automatically")
    keyword_filter: str = Field(default="", description="Keyword filter for basic search")
    search_mode: str = Field(default="basic", description="Search mode: basic or advanced")

    @field_validator("channels", mode="before")
    @classmethod
    def parse_channels(cls, v: Any) -> List[str]:
        if isinstance(v, str):
            return [c.strip() for c in v.split(",") if c.strip()]
        return v or []

    @field_validator("model_type")
    @classmethod
    def validate_model_type(cls, v: str) -> str:
        valid = {"mistral", "gemini", "llama", "llama3.2:3b", "mistral7", "groq"}
        groq_models = {"llama-3.1", "llama-3.2", "mixtral", "gemma2"}
        if (
            v.lower() in valid
            or any(v.lower().startswith(m) for m in valid)
            or any(v.lower().startswith(g) for g in groq_models)
        ):
            return v
        return "mistral"


class VacancyBase(BaseModel):
    id: str = Field(..., description="Unique vacancy ID")
    channel: str = Field(..., description="Source channel/company name")
    text: str = Field(..., min_length=1, description="Vacancy text")
    date: str = Field(..., description="Publication date")
    link: Optional[str] = Field(None, description="Link to original posting")
    source: VacancySource = Field(default=VacancySource.TELEGRAM, description="Source platform")
    title: Optional[str] = Field(None, description="Job title (for custom vacancies)")


class VacancyAnalysisResult(BaseModel):
    suitable: bool = Field(..., description="Whether vacancy matches criteria")
    reasons_fit: List[str] = Field(default_factory=list, description="Reasons why it fits")
    reasons_reject: List[str] = Field(default_factory=list, description="Reasons for rejection")
    position_type: str = Field(default="other", description="Detected position type")
    summary: str = Field(default="", description="Analysis summary")
    match_score: int = Field(default=0, ge=0, le=100, description="Match score 0-100")

    @field_validator("match_score", mode="before")
    @classmethod
    def parse_match_score(cls, v: Any) -> int:
        if isinstance(v, str):
            try:
                return int(v)
            except ValueError:
                return 0
        return int(v) if v else 0


class RecruiterAnalysisResult(BaseModel):
    match_score: int = Field(default=0, ge=0, le=100)
    strong_sides: List[str] = Field(default_factory=list)
    weak_sides: List[str] = Field(default_factory=list)
    missing_skills: List[str] = Field(default_factory=list)
    risks: List[str] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)
    verdict: str = Field(default="")
    cover_letter_hint: str = Field(default="")

    @field_validator("match_score", mode="before")
    @classmethod
    def parse_match_score(cls, v: Any) -> int:
        if isinstance(v, str):
            try:
                return int(v)
            except ValueError:
                return 0
        return int(v) if v else 0


class ResumeComparisonResult(BaseModel):
    match_score: int = Field(default=0, ge=0, le=100)
    strong_sides: List[str] = Field(default_factory=list)
    weak_sides: List[str] = Field(default_factory=list)
    missing_skills: List[str] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)
    improved_resume: str = Field(default="")
    cover_letter_hint: str = Field(default="")


class Vacancy(VacancyBase):
    analysis: str = Field(default="", description="Stage 1 analysis text")
    is_new: bool = Field(default=True, description="Whether vacancy is new")
    added_at: Optional[str] = Field(None, description="When added to storage")
    recruiter_analysis: Optional[RecruiterAnalysisResult] = None
    comparison: Optional[Dict[str, Any]] = None

    model_config = {"extra": "allow"}


class ResumeData(BaseModel):
    experience_years: Union[int, float] = Field(default=0, ge=0)
    level: ExperienceLevel = Field(default=ExperienceLevel.JUNIOR)
    key_skills: List[str] = Field(default_factory=list, max_length=20)
    projects: List[str] = Field(default_factory=list)
    summary: str = Field(default="")
    name: Optional[str] = None
    raw_text: Optional[str] = Field(None, description="Original resume text")

    @field_validator("level", mode="before")
    @classmethod
    def parse_level(cls, v: Any) -> ExperienceLevel:
        if isinstance(v, str):
            try:
                return ExperienceLevel(v.lower())
            except ValueError:
                return ExperienceLevel.JUNIOR
        if isinstance(v, ExperienceLevel):
            return v
        return ExperienceLevel.JUNIOR


class PhoneAuthRequest(BaseModel):
    phone: str = Field(..., min_length=10, max_length=20)

    @field_validator("phone")
    @classmethod
    def normalize_phone(cls, v: str) -> str:
        v = v.strip()
        if not v.startswith("+"):
            v = "+" + v
        return v


class CodeSubmitRequest(BaseModel):
    code: str = Field(..., min_length=4, max_length=10)

    @field_validator("code")
    @classmethod
    def validate_code(cls, v: str) -> str:
        v = v.strip()
        if not v.isdigit():
            raise ValueError("Code must contain only digits")
        return v


class PasswordSubmitRequest(BaseModel):
    password: str = Field(..., min_length=1)


class ImproveResumeRequest(BaseModel):
    vacancy_text: str = Field(..., min_length=20)
    vacancy_title: str = Field(default="")
    vacancy_id: str = Field(default="")
    recruiter_analysis: Optional[RecruiterAnalysisResult] = None


class ResumeSetRequest(BaseModel):
    resume_data: Dict[str, Any]


class CustomVacancyRequest(BaseModel):
    text: str = Field(..., min_length=30, description="Full vacancy description")
    title: Optional[str] = Field(None, description="Job title")
    company: Optional[str] = Field(None, description="Company name")
    source: VacancySource = Field(default=VacancySource.CUSTOM, description="Source platform")
    link: Optional[str] = Field(None, description="Link to original posting")
    skip_analysis: bool = Field(default=False, description="Skip AI analysis and add directly")


class StatusResponse(BaseModel):
    status: str
    message: Optional[str] = None
    error: Optional[str] = None


class StatsResponse(BaseModel):
    found: int = Field(default=0, ge=0)
    processed: int = Field(default=0, ge=0)
    rejected: int = Field(default=0, ge=0)
    suitable: int = Field(default=0, ge=0)


class VacanciesResponse(BaseModel):
    vacancies: List[Vacancy] = Field(default_factory=list)


class SessionResponse(BaseModel):
    settings: AppSettings
    stats: StatsResponse
    resume_data: Optional[ResumeData] = None
    is_monitoring: bool = False


class AuthStatusResponse(BaseModel):
    status: Dict[str, Any]
    user: Optional[Dict[str, Any]] = None


class ModelsResponse(BaseModel):
    models: List[Dict[str, Any]] = Field(default_factory=list)


class ImprovementStatusResponse(BaseModel):
    status: TaskStatus
    result: Optional[ResumeComparisonResult] = None


class WSMessageBase(BaseModel):
    type: str


class WSVacancyMessage(WSMessageBase):
    type: str = "vacancy"
    vacancy: Vacancy


class WSStatsMessage(WSMessageBase):
    type: str = "stats"
    stats: StatsResponse


class WSStatusMessage(WSMessageBase):
    type: str = "status"
    message: str
    icon: str = ""


class WSProgressMessage(WSMessageBase):
    type: str = "progress"
    percent: int = Field(ge=0, le=100)
    remaining: Optional[int] = None


class WSVacancyUpdateMessage(WSMessageBase):
    type: str = "vacancy_update"
    vacancy_id: str
    recruiter_analysis: RecruiterAnalysisResult


class WSResumeImprovedMessage(WSMessageBase):
    type: str = "resume_improved"
    vacancy_id: str
    result: Optional[ResumeComparisonResult] = None
    error: Optional[str] = None


class WSStreamMessage(WSMessageBase):
    type: str = "stream"
    stream_type: str
    chunk: str


class WSSystemMonitorMessage(WSMessageBase):
    type: str = "system_monitor"
    gpu: Optional[Dict[str, Any]] = None
    cpu: Dict[str, Any]
    has_gpu: bool = False


def validate_vacancy_text(text: str) -> str:
    text = text.strip()
    if len(text) < 30:
        raise ValueError("Vacancy text too short (min 30 chars)")
    return text


def validate_channel(channel: str) -> str:
    channel = channel.strip()
    if not channel:
        raise ValueError("Channel cannot be empty")
    if not (channel.startswith("@") or channel.startswith("-") or channel.isdigit()):
        if channel.replace("_", "").isalnum():
            channel = "@" + channel
    return channel


T_Model = TypeVar("T_Model", bound=BaseModel)


def parse_ai_response(response: Dict[str, Any], model_class: Type[T_Model]) -> T_Model:
    try:
        return model_class.model_validate(response)
    except Exception:
        return model_class()
