from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Pattern, Set
import re

from .models import SectionType


@dataclass(frozen=True)
class SectionRule:
    section_type: SectionType

    # Primary keywords (high weight)
    keywords_ru: FrozenSet[str] = field(default_factory=frozenset)
    keywords_en: FrozenSet[str] = field(default_factory=frozenset)

    # Secondary/alternative keywords (medium weight)
    keywords_alt: FrozenSet[str] = field(default_factory=frozenset)

    # Verbs that typically appear in this section's content
    content_verbs_ru: FrozenSet[str] = field(default_factory=frozenset)
    content_verbs_en: FrozenSet[str] = field(default_factory=frozenset)

    # Scoring weights
    keyword_primary_score: int = 50
    keyword_alt_score: int = 30
    verb_score: int = 10
    colon_bonus: int = 15
    standalone_line_bonus: int = 10
    position_score_multiplier: float = 1.0


SECTION_RULES: Dict[SectionType, SectionRule] = {
    SectionType.REQUIREMENTS: SectionRule(
        section_type=SectionType.REQUIREMENTS,
        keywords_ru=frozenset(
            {
                "требования",
                "требуется",
                "что нужно",
                "что мы ждём",
                "что мы ждем",
                "необходимые навыки",
                "обязательные требования",
                "ключевые навыки",
                "навыки",
                "квалификация",
                "компетенции",
                "нам важно",
            }
        ),
        keywords_en=frozenset(
            {
                "requirements",
                "required",
                "must have",
                "hard skills",
                "skills",
                "qualifications",
                "what we need",
                "what you need",
                "you have",
                "we expect",
                "we require",
                "essential",
            }
        ),
        keywords_alt=frozenset(
            {
                "ожидаем",
                "ищем кандидата",
                "looking for",
                "ideal candidate",
            }
        ),
        content_verbs_ru=frozenset(
            {
                "знание",
                "опыт",
                "умение",
                "владение",
                "понимание",
            }
        ),
        content_verbs_en=frozenset(
            {
                "experience",
                "knowledge",
                "understanding",
                "proficiency",
            }
        ),
        keyword_primary_score=60,
    ),
    SectionType.RESPONSIBILITIES: SectionRule(
        section_type=SectionType.RESPONSIBILITIES,
        keywords_ru=frozenset(
            {
                "обязанности",
                "задачи",
                "что предстоит",
                "чем предстоит заниматься",
                "чем будете заниматься",
                "что нужно будет делать",
                "основные задачи",
                "вам предстоит",
                "ваши задачи",
                "функционал",
            }
        ),
        keywords_en=frozenset(
            {
                "responsibilities",
                "duties",
                "job responsibilities",
                "what you will do",
                "your role",
                "you will",
                "key responsibilities",
                "day to day",
                "tasks",
                "scope",
            }
        ),
        keywords_alt=frozenset(
            {
                "функции",
                "role",
                "position",
            }
        ),
        content_verbs_ru=frozenset(
            {
                "разрабатывать",
                "создавать",
                "участвовать",
                "работать",
                "писать",
                "внедрять",
                "поддерживать",
                "проектировать",
                "анализировать",
            }
        ),
        content_verbs_en=frozenset(
            {
                "develop",
                "create",
                "build",
                "design",
                "implement",
                "maintain",
                "write",
                "analyze",
                "collaborate",
                "lead",
                "manage",
            }
        ),
        keyword_primary_score=60,
    ),
    SectionType.NICE_TO_HAVE: SectionRule(
        section_type=SectionType.NICE_TO_HAVE,
        keywords_ru=frozenset(
            {
                "будет плюсом",
                "желательно",
                "дополнительно",
                "преимуществом будет",
                "бонусом будет",
                "приветствуется",
                "хорошо если",
                "будет здорово",
            }
        ),
        keywords_en=frozenset(
            {
                "nice to have",
                "preferred",
                "bonus",
                "plus",
                "good to have",
                "preferred qualifications",
                "would be a plus",
                "advantageous",
            }
        ),
        keywords_alt=frozenset(
            {
                "optional",
                "желательные",
                "дополнительные требования",
            }
        ),
        keyword_primary_score=55,
    ),
    SectionType.TECH_STACK: SectionRule(
        section_type=SectionType.TECH_STACK,
        keywords_ru=frozenset(
            {
                "стек",
                "технологии",
                "инструменты",
                "стек технологий",
                "технический стек",
                "используем",
                "работаем с",
            }
        ),
        keywords_en=frozenset(
            {
                "tech stack",
                "stack",
                "technologies",
                "tools",
                "our stack",
                "we use",
                "technology stack",
                "technical stack",
            }
        ),
        keywords_alt=frozenset(
            {
                "frameworks",
                "фреймворки",
                "языки",
                "languages",
            }
        ),
        keyword_primary_score=50,
    ),
    SectionType.BENEFITS: SectionRule(
        section_type=SectionType.BENEFITS,
        keywords_ru=frozenset(
            {
                "условия",
                "мы предлагаем",
                "что предлагаем",
                "мы гарантируем",
                "наше предложение",
                "бонусы",
                "плюшки",
                "преимущества",
                "почему мы",
                "что вы получите",
            }
        ),
        keywords_en=frozenset(
            {
                "benefits",
                "compensation",
                "perks",
                "what we offer",
                "we offer",
                "compensation and benefits",
                "why us",
                "why join",
                "package",
            }
        ),
        keywords_alt=frozenset(
            {
                "offer",
                "предложение",
                "бенефиты",
                "соцпакет",
            }
        ),
        content_verbs_ru=frozenset(
            {
                "оплата",
                "дмс",
                "обучение",
                "отпуск",
                "график",
            }
        ),
        content_verbs_en=frozenset(
            {
                "salary",
                "insurance",
                "vacation",
                "remote",
                "flexible",
            }
        ),
        keyword_primary_score=55,
    ),
    SectionType.ABOUT_COMPANY: SectionRule(
        section_type=SectionType.ABOUT_COMPANY,
        keywords_ru=frozenset(
            {
                "о компании",
                "о нас",
                "кто мы",
                "компания",
                "наша компания",
                "мы —",
                "мы -",
                "мы это",
            }
        ),
        keywords_en=frozenset(
            {
                "about company",
                "about the company",
                "about us",
                "who we are",
                "company",
                "our company",
                "we are",
            }
        ),
        keywords_alt=frozenset(
            {
                "студия",
                "studio",
                "team",
                "команда проекта",
            }
        ),
        keyword_primary_score=45,
        position_score_multiplier=1.2,
    ),
    SectionType.ABOUT_JOB: SectionRule(
        section_type=SectionType.ABOUT_JOB,
        keywords_ru=frozenset(
            {
                "о вакансии",
                "о позиции",
                "о работе",
                "описание вакансии",
                "описание позиции",
            }
        ),
        keywords_en=frozenset(
            {
                "about the job",
                "about the position",
                "about the role",
                "position description",
                "job description",
                "the role",
            }
        ),
        keywords_alt=frozenset(
            {
                "overview",
                "summary",
                "обзор",
            }
        ),
        keyword_primary_score=45,
    ),
    SectionType.TEAM: SectionRule(
        section_type=SectionType.TEAM,
        keywords_ru=frozenset(
            {
                "команда",
                "о команде",
                "наша команда",
                "коллектив",
            }
        ),
        keywords_en=frozenset(
            {
                "team",
                "about team",
                "about the team",
                "our team",
                "the team",
            }
        ),
        keywords_alt=frozenset(
            {
                "colleagues",
                "коллеги",
            }
        ),
        keyword_primary_score=40,
    ),
    SectionType.SALARY: SectionRule(
        section_type=SectionType.SALARY,
        keywords_ru=frozenset(
            {
                "зарплата",
                "оплата",
                "вознаграждение",
                "оклад",
                "заработная плата",
                "доход",
                "ставка",
            }
        ),
        keywords_en=frozenset(
            {
                "salary",
                "compensation",
                "pay",
                "rate",
                "wage",
            }
        ),
        keywords_alt=frozenset(
            {
                "package",
                "income",
                "з/п",
                "зп",
            }
        ),
        keyword_primary_score=50,
    ),
    SectionType.EXPERIENCE: SectionRule(
        section_type=SectionType.EXPERIENCE,
        keywords_ru=frozenset(
            {
                "опыт работы",
                "требуемый опыт",
                "стаж",
                "уровень",
            }
        ),
        keywords_en=frozenset(
            {
                "experience",
                "required experience",
                "years of experience",
                "seniority",
                "level",
            }
        ),
        keywords_alt=frozenset(
            {
                "junior",
                "middle",
                "senior",
                "lead",
                "джуниор",
                "мидл",
                "сеньор",
            }
        ),
        keyword_primary_score=45,
    ),
    SectionType.WORK_FORMAT: SectionRule(
        section_type=SectionType.WORK_FORMAT,
        keywords_ru=frozenset(
            {
                "локация",
                "формат",
                "график",
                "режим работы",
                "место работы",
                "формат работы",
                "где работать",
            }
        ),
        keywords_en=frozenset(
            {
                "location",
                "work format",
                "work mode",
                "schedule",
                "workplace",
                "where",
                "remote",
                "office",
                "hybrid",
            }
        ),
        keywords_alt=frozenset(
            {
                "удалённо",
                "удаленно",
                "офис",
                "гибрид",
                "relocation",
            }
        ),
        keyword_primary_score=45,
    ),
    SectionType.CONTACT: SectionRule(
        section_type=SectionType.CONTACT,
        keywords_ru=frozenset(
            {
                "контакты",
                "откликнуться",
                "как откликнуться",
                "отклик",
                "связаться",
                "написать",
                "резюме отправлять",
            }
        ),
        keywords_en=frozenset(
            {
                "contact",
                "apply",
                "how to apply",
                "get in touch",
                "reach out",
                "send resume",
                "submit",
            }
        ),
        keywords_alt=frozenset(
            {
                "hr",
                "recruiter",
                "рекрутер",
                "email",
                "telegram",
            }
        ),
        keyword_primary_score=50,
        position_score_multiplier=0.8,
    ),
}


# MARKERS
BULLET_MARKERS: Set[str] = {
    "•",
    "·",
    "‣",
    "▪",
    "▸",
    "►",
    "★",
    "✓",
    "✔",
    "☑",
    "◆",
    "●",
    "○",
    "→",
    "➔",
    "➜",
    "➤",
    "⁃",
    "⦁",
    "⦿",
    "◉",
    "◎",
    "-",
    "–",
    "—",
    "*",
}

EMOJI_PATTERN: Pattern[str] = re.compile(
    r"[\U0001F300-\U0001F9FF"
    r"\U0001FA00-\U0001FAFF"
    r"\U00002600-\U000027BF"
    r"\U0001F600-\U0001F64F"
    r"]+",
    flags=re.UNICODE,
)

DECORATIVE_PATTERNS: List[Pattern[str]] = [
    re.compile(r"^[=\-_]{3,}$"),  # Horizontal lines
    re.compile(r"^[*#]{3,}$"),  # Asterisk/hash lines
    re.compile(r"^\s*[~`]{3,}\s*$"),  # Code block markers
    re.compile(r"^[\s\u200b\u200c\u200d\ufeff]+$"),  # Invisible chars
]

HEADER_SUFFIX_PATTERN: Pattern[str] = re.compile(r"[:：]\s*$")

NOISE_PREFIXES: Set[str] = {
    "вакансия:",
    "vacancy:",
    "позиция:",
    "position:",
    "🔥",
    "💼",
    "📌",
    "🚀",
    "💡",
    "✨",
    "🎯",
    "👉",
}


CURRENCY_PATTERNS: Dict[str, List[str]] = {
    "USD": ["$", "usd", "долларов", "долл", "баксов"],
    "EUR": ["€", "eur", "евро"],
    "RUB": ["₽", "руб", "rub", "рублей", "р.", "тыс.руб", "тыс руб"],
    "GBP": ["£", "gbp", "фунтов"],
    "PLN": ["pln", "zł", "злотых"],
    "UAH": ["грн", "uah", "гривен"],
}

SALARY_PATTERNS: List[tuple[Pattern[str], str]] = [
    # $1000 - $2000, 1000$ - 2000$
    (
        re.compile(
            r"(?P<curr1>[$€₽£])?\s*(?P<min>\d[\d\s,.]*\d|\d+)\s*"
            r"(?P<curr2>[$€₽£])?\s*"
            r"[-–—до]\s*"
            r"(?P<curr3>[$€₽£])?\s*(?P<max>\d[\d\s,.]*\d|\d+)\s*"
            r"(?P<curr4>[$€₽£])?",
            re.IGNORECASE,
        ),
        "range",
    ),
    # от 100000 / from $100k
    (
        re.compile(
            r"(?:от|from)\s*(?P<curr1>[$€₽£])?\s*(?P<min>\d[\d\s,.]*\d?[kкт]?)\s*(?P<curr2>[$€₽£])?",
            re.IGNORECASE,
        ),
        "from",
    ),
    # до 200000 / up to $200k
    (
        re.compile(
            r"(?:до|up\s*to)\s*(?P<curr1>[$€₽£])?\s*(?P<max>\d[\d\s,.]*\d?[kкт]?)\s*(?P<curr2>[$€₽£])?",
            re.IGNORECASE,
        ),
        "upto",
    ),
    # $150k, 150000 USD, 150 тыс руб
    (
        re.compile(
            r"(?P<curr1>[$€₽£])?\s*(?P<val>\d[\d\s,.]*\d?)\s*"
            r"(?P<mult>[kкт]|тыс\.?|тысяч)?\s*"
            r"(?P<curr2>[$€₽£]|usd|eur|rub|руб\.?|долл\.?|евро)?",
            re.IGNORECASE,
        ),
        "single",
    ),
]

# Experience patterns
EXPERIENCE_PATTERNS: List[Pattern[str]] = [
    re.compile(
        r"(?:от\s*)?(\d+)\s*[-–—до]\s*(\d+)\s*(?:лет|года?|years?)", re.IGNORECASE
    ),
    re.compile(
        r"(\d+)\+?\s*(?:лет|года?|years?)\s*(?:опыта|experience)?", re.IGNORECASE
    ),
    re.compile(
        r"(?:опыт|experience)[\s:]*(\d+)\+?\s*(?:лет|года?|years?)?", re.IGNORECASE
    ),
]

# Remote work indicators
REMOTE_POSITIVE: Set[str] = {
    "удалённо",
    "удаленно",
    "удалёнка",
    "удаленка",
    "remote",
    "remotely",
    "work from home",
    "wfh",
    "из дома",
    "дистанционно",
    "full remote",
}

REMOTE_NEGATIVE: Set[str] = {
    "офис",
    "office",
    "on-site",
    "onsite",
    "очно",
    "в офисе",
}

HYBRID_INDICATORS: Set[str] = {
    "гибрид",
    "hybrid",
    "гибридный",
    "частично удалённо",
    "частично офис",
}

# Employment type patterns
EMPLOYMENT_PATTERNS: Dict[str, Set[str]] = {
    "full-time": {"full-time", "полная занятость", "фулл-тайм", "полный день"},
    "part-time": {"part-time", "частичная занятость", "парт-тайм", "неполный день"},
    "contract": {"contract", "контракт", "проектная работа", "проект"},
    "freelance": {"freelance", "фриланс", "фрилансер"},
    "internship": {"internship", "стажировка", "intern", "стажёр"},
}


TECH_KEYWORDS: Set[str] = {
    # Languages
    "python",
    "javascript",
    "typescript",
    "java",
    "kotlin",
    "swift",
    "c++",
    "c#",
    "go",
    "golang",
    "rust",
    "ruby",
    "php",
    "scala",
    # Frontend
    "react",
    "vue",
    "angular",
    "svelte",
    "nextjs",
    "next.js",
    "nuxt",
    # Backend
    "node",
    "nodejs",
    "node.js",
    "django",
    "flask",
    "fastapi",
    "spring",
    "express",
    "nest",
    "nestjs",
    # Mobile
    "ios",
    "android",
    "flutter",
    "react native",
    "swiftui",
    "jetpack compose",
    # Data
    "sql",
    "postgresql",
    "postgres",
    "mysql",
    "mongodb",
    "redis",
    "elasticsearch",
    "kafka",
    "rabbitmq",
    "clickhouse",
    # DevOps
    "docker",
    "kubernetes",
    "k8s",
    "aws",
    "gcp",
    "azure",
    "terraform",
    "ci/cd",
    "jenkins",
    "gitlab ci",
    "github actions",
    # Game dev
    "unity",
    "unreal",
    "unreal engine",
    "godot",
    "cocos",
    "c++",
    "blueprints",
    "hlsl",
    "glsl",
    "shaders",
    # ML/AI
    "machine learning",
    "ml",
    "deep learning",
    "pytorch",
    "tensorflow",
    "llm",
    "nlp",
    "computer vision",
    "opencv",
    # Other
    "git",
    "linux",
    "agile",
    "scrum",
    "jira",
    "figma",
    "rest",
    "graphql",
    "microservices",
    "api",
}
