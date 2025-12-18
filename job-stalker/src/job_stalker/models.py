"""
Pydantic Models for JobStalker

Provides data validation for:
- API requests/responses
- Vacancy data
- Resume data
- Settings
- AI analysis results
"""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Type, TypeVar, Union
from pydantic import BaseModel, Field, field_validator, model_validator


# ============== ENUMS ==============

class ModelType(str, Enum):
    """Supported AI model types"""
    MISTRAL = "mistral"
    GEMINI = "gemini"
    LLAMA = "llama"


class PositionType(str, Enum):
    """Job position types"""
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
    """Experience level categories"""
    JUNIOR = "junior"
    MIDDLE = "middle"
    SENIOR = "senior"


class VacancySource(str, Enum):
    """Source of the vacancy"""
    TELEGRAM = "telegram"
    LINKEDIN = "linkedin"
    HEADHUNTER = "headhunter"
    HABR = "habr"
    CUSTOM = "custom"


class TaskStatus(str, Enum):
    """Background task status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    ERROR = "error"
    CANCELLED = "cancelled"


# ============== SETTINGS ==============

class AppSettings(BaseModel):
    """Application settings model"""
    model_type: str = Field(default="mistral", description="AI model to use")
    days_back: int = Field(default=7, ge=1, le=365, description="Days to search back")
    custom_prompt: str = Field(default="", description="Custom filter prompt")
    resume_summary: str = Field(default="", description="Resume summary")
    channels: List[str] = Field(default_factory=list, description="Telegram channels to monitor")
    enable_stage2: bool = Field(default=False, description="Run recruiter analysis automatically")
    keyword_filter: str = Field(default="", description="Keyword filter for basic search")
    search_mode: str = Field(default="basic", description="Search mode: basic or advanced")

    @field_validator('channels', mode='before')
    @classmethod
    def parse_channels(cls, v: Any) -> List[str]:
        """Parse channels from string or list"""
        if isinstance(v, str):
            return [c.strip() for c in v.split(',') if c.strip()]
        return v or []

    @field_validator('model_type')
    @classmethod
    def validate_model_type(cls, v: str) -> str:
        """Validate model type"""
        valid = {'mistral', 'gemini', 'llama', 'llama3.2:3b', 'mistral7'}
        # Allow any model name that starts with valid prefixes
        if v.lower() in valid or any(v.lower().startswith(m) for m in valid):
            return v
        return 'mistral'


# ============== VACANCY ==============

class VacancyBase(BaseModel):
    """Base vacancy model"""
    id: str = Field(..., description="Unique vacancy ID")
    channel: str = Field(..., description="Source channel/company name")
    text: str = Field(..., min_length=1, description="Vacancy text")
    date: str = Field(..., description="Publication date")
    link: Optional[str] = Field(None, description="Link to original posting")
    source: VacancySource = Field(default=VacancySource.TELEGRAM, description="Source platform")
    title: Optional[str] = Field(None, description="Job title (for custom vacancies)")


class VacancyAnalysisResult(BaseModel):
    """Stage 1: Quick filter result"""
    suitable: bool = Field(..., description="Whether vacancy matches criteria")
    reasons_fit: List[str] = Field(default_factory=list, description="Reasons why it fits")
    reasons_reject: List[str] = Field(default_factory=list, description="Reasons for rejection")
    position_type: str = Field(default="other", description="Detected position type")
    summary: str = Field(default="", description="Analysis summary")
    match_score: int = Field(default=0, ge=0, le=100, description="Match score 0-100")

    @field_validator('match_score', mode='before')
    @classmethod
    def parse_match_score(cls, v: Any) -> int:
        """Parse match score from various types"""
        if isinstance(v, str):
            try:
                return int(v)
            except ValueError:
                return 0
        return int(v) if v else 0


class RecruiterAnalysisResult(BaseModel):
    """Stage 2: Recruiter analysis result"""
    match_score: int = Field(default=0, ge=0, le=100)
    strong_sides: List[str] = Field(default_factory=list)
    weak_sides: List[str] = Field(default_factory=list)
    missing_skills: List[str] = Field(default_factory=list)
    risks: List[str] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)
    verdict: str = Field(default="")
    cover_letter_hint: str = Field(default="")

    @field_validator('match_score', mode='before')
    @classmethod
    def parse_match_score(cls, v: Any) -> int:
        if isinstance(v, str):
            try:
                return int(v)
            except ValueError:
                return 0
        return int(v) if v else 0


class ResumeComparisonResult(BaseModel):
    """Stage 3: Resume improvement result"""
    match_score: int = Field(default=0, ge=0, le=100)
    strong_sides: List[str] = Field(default_factory=list)
    weak_sides: List[str] = Field(default_factory=list)
    missing_skills: List[str] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)
    improved_resume: str = Field(default="")
    cover_letter_hint: str = Field(default="")


class Vacancy(VacancyBase):
    """Full vacancy model with analysis"""
    analysis: str = Field(default="", description="Stage 1 analysis text")
    is_new: bool = Field(default=True, description="Whether vacancy is new")
    added_at: Optional[str] = Field(None, description="When added to storage")
    recruiter_analysis: Optional[RecruiterAnalysisResult] = None
    comparison: Optional[Dict[str, Any]] = None

    model_config = {"extra": "allow"}


# ============== RESUME ==============

class ResumeData(BaseModel):
    """Parsed resume data"""
    experience_years: Union[int, float] = Field(default=0, ge=0)
    level: ExperienceLevel = Field(default=ExperienceLevel.JUNIOR)
    key_skills: List[str] = Field(default_factory=list, max_length=20)
    projects: List[str] = Field(default_factory=list)
    summary: str = Field(default="")
    name: Optional[str] = None
    raw_text: Optional[str] = Field(None, description="Original resume text")

    @field_validator('level', mode='before')
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


# ============== API REQUESTS ==============

class PhoneAuthRequest(BaseModel):
    """Phone authentication request"""
    phone: str = Field(..., min_length=10, max_length=20)

    @field_validator('phone')
    @classmethod
    def normalize_phone(cls, v: str) -> str:
        """Normalize phone number"""
        v = v.strip()
        if not v.startswith('+'):
            v = '+' + v
        return v


class CodeSubmitRequest(BaseModel):
    """Verification code submit request"""
    code: str = Field(..., min_length=4, max_length=10)

    @field_validator('code')
    @classmethod
    def validate_code(cls, v: str) -> str:
        """Validate code format"""
        v = v.strip()
        if not v.isdigit():
            raise ValueError('Code must contain only digits')
        return v


class PasswordSubmitRequest(BaseModel):
    """2FA password submit request"""
    password: str = Field(..., min_length=1)


class ImproveResumeRequest(BaseModel):
    """Resume improvement request"""
    vacancy_text: str = Field(..., min_length=20)
    vacancy_title: str = Field(default="")
    vacancy_id: str = Field(default="")
    recruiter_analysis: Optional[RecruiterAnalysisResult] = None


class ResumeSetRequest(BaseModel):
    """Request to set active resume from stored data"""
    resume_data: Dict[str, Any]


class CustomVacancyRequest(BaseModel):
    """Request to add a custom vacancy"""
    text: str = Field(..., min_length=30, description="Full vacancy description")
    title: Optional[str] = Field(None, description="Job title")
    company: Optional[str] = Field(None, description="Company name")
    source: VacancySource = Field(default=VacancySource.CUSTOM, description="Source platform")
    link: Optional[str] = Field(None, description="Link to original posting")
    skip_analysis: bool = Field(default=False, description="Skip AI analysis and add directly")


# ============== API RESPONSES ==============

class StatusResponse(BaseModel):
    """Generic status response"""
    status: str
    message: Optional[str] = None
    error: Optional[str] = None


class StatsResponse(BaseModel):
    """Statistics response"""
    found: int = Field(default=0, ge=0)
    processed: int = Field(default=0, ge=0)
    rejected: int = Field(default=0, ge=0)
    suitable: int = Field(default=0, ge=0)


class VacanciesResponse(BaseModel):
    """Vacancies list response"""
    vacancies: List[Vacancy] = Field(default_factory=list)


class SessionResponse(BaseModel):
    """Session state response"""
    settings: AppSettings
    stats: StatsResponse
    resume_data: Optional[ResumeData] = None
    is_monitoring: bool = False


class AuthStatusResponse(BaseModel):
    """Authentication status response"""
    status: Dict[str, Any]
    user: Optional[Dict[str, Any]] = None


class ModelsResponse(BaseModel):
    """Available models response"""
    models: List[Dict[str, Any]] = Field(default_factory=list)


class ImprovementStatusResponse(BaseModel):
    """Resume improvement status"""
    status: TaskStatus
    result: Optional[ResumeComparisonResult] = None


# ============== WEBSOCKET MESSAGES ==============

class WSMessageBase(BaseModel):
    """Base WebSocket message"""
    type: str


class WSVacancyMessage(WSMessageBase):
    """New vacancy message"""
    type: str = "vacancy"
    vacancy: Vacancy


class WSStatsMessage(WSMessageBase):
    """Stats update message"""
    type: str = "stats"
    stats: StatsResponse


class WSStatusMessage(WSMessageBase):
    """Status update message"""
    type: str = "status"
    message: str
    icon: str = ""


class WSProgressMessage(WSMessageBase):
    """Progress update message"""
    type: str = "progress"
    percent: int = Field(ge=0, le=100)
    remaining: Optional[int] = None


class WSVacancyUpdateMessage(WSMessageBase):
    """Vacancy update (Stage 2 results)"""
    type: str = "vacancy_update"
    vacancy_id: str
    recruiter_analysis: RecruiterAnalysisResult


class WSResumeImprovedMessage(WSMessageBase):
    """Resume improved message"""
    type: str = "resume_improved"
    vacancy_id: str
    result: Optional[ResumeComparisonResult] = None
    error: Optional[str] = None


class WSStreamMessage(WSMessageBase):
    """AI streaming message"""
    type: str = "stream"
    stream_type: str
    chunk: str


class WSSystemMonitorMessage(WSMessageBase):
    """System monitor message"""
    type: str = "system_monitor"
    gpu: Optional[Dict[str, Any]] = None
    cpu: Dict[str, Any]
    has_gpu: bool = False


# ============== VALIDATION HELPERS ==============

def validate_vacancy_text(text: str) -> str:
    """Validate vacancy text"""
    text = text.strip()
    if len(text) < 30:
        raise ValueError("Vacancy text too short (min 30 chars)")
    return text


def validate_channel(channel: str) -> str:
    """Validate channel identifier"""
    channel = channel.strip()
    if not channel:
        raise ValueError("Channel cannot be empty")
    # Channel can be @username or numeric ID
    if not (channel.startswith('@') or channel.startswith('-') or channel.isdigit()):
        # Try to add @ prefix if looks like username
        if channel.replace('_', '').isalnum():
            channel = '@' + channel
    return channel


T_Model = TypeVar('T_Model', bound=BaseModel)


def parse_ai_response(response: Dict[str, Any], model_class: Type[T_Model]) -> T_Model:
    """Safely parse AI response into model"""
    try:
        return model_class.model_validate(response)
    except Exception:
        # Return default instance on parse error
        return model_class()
