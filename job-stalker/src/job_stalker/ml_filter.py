import asyncio
import re
import logging
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Callable, List
from dataclasses import dataclass, field

from .ai_client import AIClientFactory, OllamaClient, GeminiClient

log = logging.getLogger("ml_filter")

try:
    from .config import GEMINI_API_KEY, GROQ_API_KEY
except ImportError:
    GEMINI_API_KEY = None
    GROQ_API_KEY = None

RESUME_DATA: Optional[Dict] = None
_resume_lock = asyncio.Lock()

DATA_DIR = Path("data")
OUTPUT_DIR = Path("output")


@dataclass
class RecruiterAnalysis:
    match_score: int = 0
    strong_sides: List[str] = field(default_factory=list)
    weak_sides: List[str] = field(default_factory=list)
    missing_skills: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    risks: List[str] = field(default_factory=list)
    verdict: str = ""
    cover_letter_hint: str = ""


@dataclass
class ResumeComparison:
    match_score: int = 0
    strong_sides: List[str] = field(default_factory=list)
    weak_sides: List[str] = field(default_factory=list)
    missing_skills: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    improved_resume: str = ""
    cover_letter_hint: str = ""


@dataclass
class VacancyAnalysis:
    suitable: bool
    analysis: str = ""
    match_score: int = 0
    recruiter_analysis: Optional[RecruiterAnalysis] = None
    comparison: Optional[ResumeComparison] = None

    def __bool__(self):
        return self.suitable


_stream_callback: Optional[Callable] = None


def set_stream_callback(callback: Optional[Callable]):
    global _stream_callback
    _stream_callback = callback
    ollama = AIClientFactory.get_ollama_client()
    ollama.set_stream_callback(callback)


async def notify_stream(chunk: str, stream_type: str = "analysis"):
    if _stream_callback:
        try:
            await _stream_callback({"type": "stream", "stream_type": stream_type, "chunk": chunk})
        except Exception as e:
            log.warning(f"Stream callback error: {e}")


# Default filter prompt example
DEFAULT_FILTER_PROMPT_EXAMPLE = """Ищу позиции:
✅ Unreal Engine разработчик (junior, junior+, middle)
✅ C++ разработчик в геймдеве
✅ Game programmer
✅ Technical Artist с программированием

НЕ подходят:
❌ Senior позиции (3+ года опыта)
❌ Unity-only без Unreal
❌ Менеджеры, HR, маркетологи
❌ Художники без программирования
❌ QA без кода"""


def get_filter_prompt(custom_prompt: str = "", resume_summary: str = "") -> str:
    if not custom_prompt or not custom_prompt.strip():
        return ""

    base = f"""Ты AI-ассистент для фильтрации вакансий. Проанализируй вакансию согласно критериям пользователя.

КРИТЕРИИ ПОЛЬЗОВАТЕЛЯ:
{custom_prompt.strip()}

АЛГОРИТМ АНАЛИЗА:
1. Определи ТИП ПОЗИЦИИ (роль): developer/programmer/engineer VS artist/designer/manager/qa и т.д.
2. Проверь ИСКЛЮЧЕНИЯ: если вакансия попадает под любой пункт исключений из критериев пользователя - сразу suitable: false
3. Проверь ТРЕБОВАНИЯ: только если вакансия НЕ попадает под исключения - проверь соответствие требованиям

⚠️ КРИТИЧЕСКИ ВАЖНО:
- Роль/позиция важнее упоминания технологий! Если вакансия для "artist" или "designer", но упоминает нужные технологии - это НЕ подходит для "developer"
- Если в критериях есть исключения по ролям (например "художники", "дизайнеры", "менеджеры") - такие вакансии ВСЕГДА suitable: false
- Исключения имеют приоритет над требованиями

Ответь ТОЛЬКО JSON (без markdown, без комментариев):
{{
  "suitable": true или false,
  "reasons_fit": ["почему подходит"],
  "reasons_reject": ["почему не подходит"],
  "position_type": "developer/manager/designer/artist/qa/other",
  "summary": "краткий вывод на русском",
  "match_score": число от 0 до 100 (насколько вакансия соответствует критериям{" и резюме" if resume_summary else ""})
}}"""

    if resume_summary:
        base = f"Резюме кандидата: {resume_summary}\n\n{base}"

    return base


def get_default_prompt(resume_summary: str = "") -> str:
    return get_filter_prompt(DEFAULT_FILTER_PROMPT_EXAMPLE, resume_summary)


def get_comparison_prompt(vacancy_text: str, resume_text: str) -> str:
    return f"""Ты — опытный карьерный ассистент и эксперт по оптимизации резюме под системы отслеживания кандидатов (ATS).

Задача:

Я дам тебе описание вакансии и своё резюме. Твоя задача — адаптировать резюме так, чтобы оно максимально совпадало с описанием вакансии.

ВАКАНСИЯ:
{vacancy_text[:2000]}

ТЕКУЩЕЕ РЕЗЮМЕ:
{resume_text[:2000]}

Правила:

1. Выдели все ключевые слова из описания вакансии:

должность
навыки
инструменты и технологии
обязанности
отраслевые термины
soft skills
ключевые фразы

2. Сравни описание вакансии с моим резюме:

если навык уже есть — усили его формулировку
если навык есть, но описан слабо — перепиши и подчеркни опыт
если навыка нет, но у меня был похожий опыт — добавь релевантную формулировку
если навыка нет и нельзя предположить — то пропусти его и переходи к следующему

3. Перестрой структуру резюме:

перемести самый релевантный опыт выше
перепиши summary в начале с использованием ключевых слов
подбирай формулировки, похожие на вакансию (но не копируй слово в слово)

4. Оформление (обязательно ATS-дружелюбное):

без таблиц, иконок, картинок
только стандартные блоки текстом

Итог:
Дай полностью переписанное резюме, адаптированное под эту вакансию, с естественно встроенными ключевыми словами.
В поле "improved_resume" напиши ПОЛНЫЙ текст улучшенного резюме (не краткий, а полноценный документ), адаптированный под эту вакансию. Добавь релевантные ключевые слова из вакансии, подчеркни подходящий опыт.

ВАЖНО: Верни ТОЛЬКО ОДИН валидный JSON объект (без markdown, без escape символов, без разбиения):
{{
  "match_score": число от 0 до 100,
  "strong_sides": ["сильная сторона 1", "сильная сторона 2"],
  "weak_sides": ["слабая сторона 1"],
  "missing_skills": ["недостающий навык 1", "недостающий навык 2"],
  "recommendations": ["рекомендация 1", "рекомендация 2"],
  "cover_letter_hint": "подсказка для сопроводительного письма",
  "improved_resume": "ПОЛНЫЙ ТЕКСТ улучшенного резюме здесь, минимум 500 символов"
}}

JSON:"""


def extract_json_safely(text: str) -> dict:
    original_text = text

    text = text.replace("```json", "").replace("```", "").strip()

    if text.startswith("JSON:"):
        text = text[5:].strip()

    text = text.replace("\\_", "_")

    log.debug(f"extract_json_safely: input length={len(text)}")

    depth = 0
    start = -1
    for i, char in enumerate(text):
        if char == "{":
            if depth == 0:
                start = i
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0 and start != -1:
                json_str = text[start : i + 1]
                try:
                    parsed = json.loads(json_str)
                    log.info(f"JSON parsed (method 1), keys: {list(parsed.keys())}")
                    return parsed
                except json.JSONDecodeError:
                    continue

    all_matches = []
    depth = 0
    start = -1
    for i, char in enumerate(text):
        if char == "{":
            if depth == 0:
                start = i
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0 and start != -1:
                all_matches.append((start, i + 1, text[start : i + 1]))
                start = -1

    all_matches.sort(key=lambda x: len(x[2]), reverse=True)
    for start_idx, end_idx, json_str in all_matches:
        try:
            parsed = json.loads(json_str)
            log.info(f"JSON parsed (method 2), keys: {list(parsed.keys())}")
            return parsed
        except Exception:
            continue

    if len(all_matches) >= 2:
        log.info(f"Found {len(all_matches)} JSON objects, trying to merge...")
        try:
            merged = {}
            for start_idx, end_idx, json_str in all_matches:
                try:
                    obj = json.loads(json_str)
                    merged.update(obj)
                except Exception:
                    continue
            if merged:
                log.info(f"JSON merged (method 2.5), keys: {list(merged.keys())}")
                return merged
        except Exception as e:
            log.warning(f"Method 2.5 failed: {e}")

    json_match = re.search(r"\{.*\}", text, re.DOTALL)
    if json_match:
        try:
            parsed = json.loads(json_match.group(0))
            log.info(f"JSON parsed (method 3), keys: {list(parsed.keys())}")
            return parsed
        except Exception as e:
            log.warning(f"Method 3 failed: {e}")

    log.warning("Standard methods failed, trying special improved_resume extraction...")
    improved_match = re.search(r'"improved_resume"\s*:\s*"([^"]*(?:\\"[^"]*)*)"', text, re.DOTALL)
    if improved_match:
        improved_text = improved_match.group(1).replace('\\"', '"')
        for start_idx, end_idx, json_str in all_matches:
            try:
                parsed = json.loads(json_str)
                parsed["improved_resume"] = improved_text
                log.info(f"JSON parsed with special extraction, keys: {list(parsed.keys())}")
                return parsed
            except Exception:
                continue

    log.warning("Trying auto-fix for incomplete JSON...")
    if "{" in text:
        start_idx = text.find("{")
        json_part = text[start_idx:]

        depth = 0
        for char in json_part:
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1

        if depth > 0:
            json_part += "}" * depth
            log.info(f"Added {depth} closing braces")

            try:
                parsed = json.loads(json_part)
                log.info(f"JSON auto-fixed, keys: {list(parsed.keys())}")
                return parsed
            except Exception as e:
                log.warning(f"Auto-fix failed: {e}")

    log.error(f"All JSON extraction methods failed! Text length: {len(original_text)}")
    return {}


def normalize_resume_data(data: dict) -> dict:
    normalized = {}

    # Experience
    if "experience_years" in data:
        normalized["experience_years"] = data["experience_years"]
    elif "experience" in data:
        exp = data["experience"]
        if isinstance(exp, list):
            normalized["experience_years"] = len(exp) * 2
            projects = []
            for item in exp:
                if isinstance(item, dict):
                    company = item.get("company", item.get("project", ""))
                    position = item.get("positionTitle", item.get("position", ""))
                    if company or position:
                        projects.append(f"{position} @ {company}".strip(" @"))
            if projects:
                normalized["projects"] = projects
        elif isinstance(exp, (int, float)):
            normalized["experience_years"] = exp

    # Level
    if "level" in data:
        normalized["level"] = data["level"]
    else:
        years = normalized.get("experience_years", 0)
        if isinstance(years, (int, float)):
            normalized["level"] = "junior" if years <= 2 else "middle" if years <= 5 else "senior"

    # Skills
    skills = []
    if "key_skills" in data:
        skills = data["key_skills"]
    elif "skills" in data:
        sk = data["skills"]
        if isinstance(sk, list):
            skills = sk
        elif isinstance(sk, dict):
            for items in sk.values():
                if isinstance(items, list):
                    skills.extend(items)
    normalized["key_skills"] = skills[:10]

    if "projects" not in normalized:
        normalized["projects"] = data.get("projects", [])
    normalized["summary"] = data.get("summary", "")
    if "name" in data:
        normalized["name"] = data["name"]

    return normalized


async def load_resume(file_path: str, model_type: str = "mistral") -> dict:
    global RESUME_DATA

    log.info(f"load_resume: model={model_type}, file={file_path}")

    if not os.path.exists(file_path):
        return {"error": "File not found"}

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            resume_text = f.read()
        log.info(f"Resume: {len(resume_text)} chars")
    except Exception as e:
        return {"error": f"Read error: {e}"}

    prompt = f"""Анализируй резюме. Верни JSON:
{{
  "experience_years": число,
  "level": "junior"/"middle"/"senior",
  "key_skills": ["навык1", "навык2"],
  "projects": ["проект1"],
  "summary": "краткий обзор"
}}

Резюме:
{resume_text[:2000]}

JSON:"""

    try:
        from .ai_client import GroqClient

        is_groq = (
            model_type == "groq"
            or model_type.startswith("groq:")
            or model_type in GroqClient.AVAILABLE_MODELS
        )
        api_key = GROQ_API_KEY if is_groq else GEMINI_API_KEY
        client = AIClientFactory.get_client(model_type, api_key, _stream_callback)
        response = await client.generate_with_retry(
            prompt, stream_type="resume_analysis", num_predict=1024
        )

        if not response.success:
            return {"error": response.error}

        raw_data = extract_json_safely(response.text)

        async with _resume_lock:
            RESUME_DATA = normalize_resume_data(raw_data)
            RESUME_DATA["raw_text"] = resume_text

        log.info(
            f"Resume: level={RESUME_DATA.get('level')}, exp={RESUME_DATA.get('experience_years')}"
        )
        return RESUME_DATA

    except Exception as e:
        log.error(f"Resume analysis error: {e}")
        return {"error": str(e)}


async def set_resume_data(resume_data: dict) -> dict:
    global RESUME_DATA

    if not resume_data:
        return {"error": "Empty resume data"}

    try:
        normalized = normalize_resume_data(resume_data)
        if resume_data.get("raw_text"):
            normalized["raw_text"] = resume_data["raw_text"]

        async with _resume_lock:
            RESUME_DATA = normalized

        save_session()
        log.info(
            f"Resume restored: level={normalized.get('level')}, exp={normalized.get('experience_years')}"
        )
        return normalized
    except Exception as e:
        log.error(f"set_resume_data error: {e}")
        return {"error": str(e)}


async def get_available_ollama_models() -> List[Dict[str, str]]:
    client = AIClientFactory.get_ollama_client()
    return await client.list_models()


async def recruiter_analysis(
    vacancy_text: str, resume_text: str, enable_stream: bool = False
) -> RecruiterAnalysis:
    prompt = f"""Ты — строгий, опытный, но справедливый IT-рекрутер с 10+ годами опыта найма.

ТВОЯ ЗАДАЧА: Оценить кандидата по конкретной вакансии. Будь честен и критичен — это поможет кандидату понять реальные шансы.

ВАКАНСИЯ:
{vacancy_text[:2000]}

РЕЗЮМЕ КАНДИДАТА:
{resume_text[:2000]}

АЛГОРИТМ ОЦЕНКИ:
1. Сравни требования вакансии с опытом кандидата
2. Найди сильные стороны, которые выделяют кандидата
3. Найди слабые стороны и пробелы в опыте
4. Определи критические недостающие навыки
5. Оцени риски для кандидата (почему могут отказать)
6. Дай честную оценку шансов (0-100)

КРИТЕРИИ ОЦЕНКИ match_score:
- 80-100: Отличный кандидат, высокие шансы на оффер
- 60-79: Хороший кандидат, стоит подаваться
- 40-59: Средний кандидат, шансы 50/50
- 20-39: Слабый кандидат, нужна серьёзная подготовка
- 0-19: Не соответствует требованиям

Ответь ТОЛЬКО валидным JSON (без markdown):
{{
  "match_score": число от 0 до 100,
  "strong_sides": ["конкретная сильная сторона 1", "сильная сторона 2"],
  "weak_sides": ["конкретная слабая сторона 1", "слабая сторона 2"],
  "missing_skills": ["критический навык 1", "навык 2"],
  "risks": ["риск отказа 1", "риск 2"],
  "recommendations": ["что конкретно улучшить 1", "рекомендация 2"],
  "verdict": "краткий вердикт рекрутера (1-2 предложения)",
  "cover_letter_hint": "ключевая мысль для сопроводительного письма"
}}

JSON:"""

    try:
        log.info("Stage 2: Running recruiter analysis...")

        client = AIClientFactory.get_ollama_client()
        stream_type = "recruiter_analysis" if enable_stream else None
        response = await client.generate_with_retry(
            prompt, stream_type=stream_type, num_predict=1024
        )

        if not response.success:
            log.error(f"Recruiter analysis failed: {response.error}")
            return RecruiterAnalysis(verdict="Analysis failed")

        output = response.text.replace("\\_", "_")
        data = extract_json_safely(output)

        if not data:
            log.error("Recruiter analysis JSON parsing failed")
            return RecruiterAnalysis(verdict="Could not complete analysis")

        match_score = data.get("match_score", 0)
        if isinstance(match_score, str):
            try:
                match_score = int(match_score)
            except Exception:
                match_score = 0

        result = RecruiterAnalysis(
            match_score=match_score,
            strong_sides=data.get("strong_sides", []),
            weak_sides=data.get("weak_sides", []),
            missing_skills=data.get("missing_skills", []),
            risks=data.get("risks", []),
            recommendations=data.get("recommendations", []),
            verdict=data.get("verdict", ""),
            cover_letter_hint=data.get("cover_letter_hint", ""),
        )

        log.info(f"Recruiter analysis done: match_score={match_score}")
        return result

    except Exception as e:
        log.error(f"Recruiter analysis error: {e}")
        return RecruiterAnalysis(verdict=f"Error: {str(e)}")


async def quick_resume_analysis(vacancy_text: str, resume_text: str) -> dict:
    result = await recruiter_analysis(vacancy_text, resume_text)
    return {
        "match_score": result.match_score,
        "strong_sides": result.strong_sides,
        "weak_sides": result.weak_sides,
        "missing_skills": result.missing_skills,
        "recommendations": result.recommendations,
        "cover_letter_hint": result.cover_letter_hint,
    }


async def generate_improved_resume(
    vacancy_text: str, resume_text: str, analysis: Optional[RecruiterAnalysis] = None
) -> str:
    analysis_context = ""
    if analysis and analysis.match_score > 0:
        analysis_context = f"""
АНАЛИЗ РЕКРУТЕРА (учти при улучшении):
- Оценка соответствия: {analysis.match_score}%
- Сильные стороны: {", ".join(analysis.strong_sides[:3]) if analysis.strong_sides else "не указаны"}
- Слабые стороны: {", ".join(analysis.weak_sides[:3]) if analysis.weak_sides else "не указаны"}
- Недостающие навыки: {", ".join(analysis.missing_skills[:3]) if analysis.missing_skills else "не указаны"}
- Рекомендации: {", ".join(analysis.recommendations[:2]) if analysis.recommendations else "не указаны"}

"""

    prompt = f"""Ты — эксперт по оптимизации резюме под ATS (Applicant Tracking System) и HR-специалистов.

ЗАДАЧА: Создай улучшенную версию резюме, адаптированную под конкретную вакансию.

ВАКАНСИЯ:
{vacancy_text[:2000]}

ИСХОДНОЕ РЕЗЮМЕ:
{resume_text[:2000]}
{analysis_context}
ПРАВИЛА УЛУЧШЕНИЯ:
1. Усиль формулировки релевантных навыков (НЕ выдумывай опыт!)
2. Аккуратно добавь недостающие компетенции, если они хоть как-то связаны с реальным опытом
3. Подчеркни сильные стороны, выявленные рекрутером
4. Адаптируй тон под стиль вакансии (формальный/неформальный)
5. Перемести самый релевантный опыт выше
6. Перепиши summary с ключевыми словами из вакансии
7. ATS-friendly: без таблиц, иконок, картинок, сложного форматирования

ВАЖНО:
- НЕ выдумывай опыт, которого нет
- НЕ добавляй навыки, которые нельзя подтвердить
- Формат: чистый markdown

СТРУКТУРА ОТВЕТА:
# [Имя кандидата]

**[Целевая должность из вакансии]**

## Summary
[3-4 предложения с ключевыми словами из вакансии]

## Опыт работы
**[Компания] — [Должность]** | [Даты]
- [Достижение с метриками]
- [Релевантный опыт]

## Навыки
- [Категория]: [навыки через запятую]

## Образование
[Образование]

---

Верни ТОЛЬКО текст улучшенного резюме (минимум 500 символов):"""

    try:
        log.info("Stage 3: Generating improved resume...")

        client = AIClientFactory.get_ollama_client()
        response = await client.generate_with_retry(
            prompt, stream_type="improved_resume", num_predict=3072
        )

        if not response.success:
            return "# Error\n\nFailed to generate improved resume."

        output = response.text.strip()

        if output.startswith("```markdown"):
            output = output[11:]
        if output.startswith("```"):
            output = output[3:]
        if output.endswith("```"):
            output = output[:-3]

        output = output.strip()
        log.info(f"Improved resume generated: {len(output)} chars")

        return output

    except Exception as e:
        log.error(f"Resume generation error: {e}")
        return "# Error\n\nFailed to generate improved resume."


async def generate_improved_resume_markdown(vacancy_text: str, resume_text: str) -> str:
    return await generate_improved_resume(vacancy_text, resume_text, None)


async def compare_with_resume(
    vacancy_text: str,
    vacancy_title: str = "",
    existing_analysis: Optional[RecruiterAnalysis] = None,
) -> ResumeComparison:
    if not RESUME_DATA or "raw_text" not in RESUME_DATA:
        log.warning("No resume for comparison")
        return ResumeComparison()

    resume_text = RESUME_DATA["raw_text"]
    log.info(f"Resume text length: {len(resume_text)}")
    log.info(f"Vacancy text length: {len(vacancy_text)}")

    try:
        if existing_analysis and existing_analysis.match_score > 0:
            log.info("Using existing recruiter analysis...")
            analysis = existing_analysis
        else:
            log.info("Stage 2: Running recruiter analysis...")
            analysis = await recruiter_analysis(vacancy_text, resume_text)

        log.info("Stage 3: Generating improved resume...")
        improved_resume = await generate_improved_resume(vacancy_text, resume_text, analysis)

        result = ResumeComparison(
            match_score=analysis.match_score,
            strong_sides=analysis.strong_sides,
            weak_sides=analysis.weak_sides,
            missing_skills=analysis.missing_skills,
            recommendations=analysis.recommendations,
            improved_resume=improved_resume,
            cover_letter_hint=analysis.cover_letter_hint,
        )

        log.info(
            f"Stage 3 done: score={result.match_score}, improved_len={len(result.improved_resume)}"
        )
        return result

    except Exception as e:
        log.error(f"Comparison error: {e}", exc_info=True)
        return ResumeComparison()


async def analyze_vacancy(text: str, model_type: str = "mistral") -> VacancyAnalysis:
    if len(text.strip()) < 20:
        return VacancyAnalysis(False, "Text too short")

    try:
        from .web_ui import get_current_settings

        settings = get_current_settings()
        custom_prompt = settings.get("custom_prompt", "")
        resume_summary = settings.get("resume_summary", "")
        log.info(f"Loaded custom_prompt length: {len(custom_prompt)} chars")
    except Exception as e:
        log.error(f"Failed to load settings: {e}")
        custom_prompt = ""
        resume_summary = ""

    if not custom_prompt or not custom_prompt.strip():
        log.info("Filter prompt empty - vacancy auto-approved")
        return VacancyAnalysis(
            suitable=True,
            analysis="Filter not configured - vacancy added automatically.\n\nConfigure criteria in Settings -> Search Filter Prompt",
            match_score=0,
            recruiter_analysis=None,
        )

    filter_prompt = get_filter_prompt(custom_prompt, resume_summary)
    vacancy_text_short = text.strip()[:1500]
    full_prompt = f"{filter_prompt}\n\nВАКАНСИЯ:\n{vacancy_text_short}\n\nJSON:"

    log.info(f"Analyzing with {model_type.upper()}...")

    try:
        from .ai_client import GroqClient

        is_groq = (
            model_type == "groq"
            or model_type.startswith("groq:")
            or model_type in GroqClient.AVAILABLE_MODELS
        )
        api_key = GROQ_API_KEY if is_groq else GEMINI_API_KEY
        client = AIClientFactory.get_client(model_type, api_key)
        response = await client.generate_with_retry(full_prompt, num_predict=2048)

        if not response.success:
            return VacancyAnalysis(False, f"Error: {response.error}")

        data = extract_json_safely(response.text)

        suitable = data.get("suitable", False)
        if isinstance(suitable, str):
            suitable = suitable.lower() in ("true", "yes", "да", "1")

        position_type = data.get("position_type", "").lower()

        log.info(f"LLM Response: suitable={suitable}, position_type={position_type}")

        if suitable and position_type:
            role_translations = {
                "artist": ["художник", "артист"],
                "designer": ["дизайнер", "design"],
                "manager": ["менеджер", "управляющий", "руководитель"],
                "qa": ["тестировщик", "тестер", "качество"],
                "producer": ["продюсер"],
                "marketing": ["маркетолог", "маркетинг"],
                "animator": ["аниматор", "анимация"],
                "modeller": ["моделлер", "модельер", "3d model"],
                "hr": ["рекрутер", "hr специалист"],
            }

            role_variants = [position_type.lower()]
            for eng, rus_list in role_translations.items():
                if eng in position_type.lower():
                    role_variants.extend(rus_list)

            negative_markers = [
                "не подходит",
                "не хочу",
                "отбрасывать",
                "исключить",
                "отбросить",
                "не твоего направления",
                "не по профилю",
                "без",
                "кроме",
                "❌",
                "NOT suitable",
                "exclude",
                "не относится",
            ]

            for marker in negative_markers:
                if marker.lower() in custom_prompt.lower():
                    parts = custom_prompt.lower().split(marker.lower())
                    for part in parts[1:]:
                        context = part[:500]
                        for variant in role_variants:
                            if variant in context:
                                log.warning(
                                    f"CONTRADICTION: position_type='{position_type}' found in negative context"
                                )
                                suitable = False
                                if not data.get("reasons_reject"):
                                    data["reasons_reject"] = []
                                if isinstance(data["reasons_reject"], str):
                                    data["reasons_reject"] = [data["reasons_reject"]]
                                data["reasons_reject"].append(
                                    f"Role '{position_type}' explicitly excluded"
                                )
                                break
                        if not suitable:
                            break
                    if not suitable:
                        break

        analysis_parts = []
        reasons_fit = data.get("reasons_fit", [])
        reasons_reject = data.get("reasons_reject", data.get("reasons_lack", []))
        summary = data.get("summary", "")

        if isinstance(reasons_fit, str):
            reasons_fit = [reasons_fit]
        if isinstance(reasons_reject, str):
            reasons_reject = [reasons_reject]

        if reasons_fit:
            analysis_parts.append("**Fits:**\n" + "\n".join(f"  - {r}" for r in reasons_fit))
        if reasons_reject:
            analysis_parts.append(
                "**Doesn't fit:**\n" + "\n".join(f"  - {r}" for r in reasons_reject)
            )
        if summary:
            analysis_parts.append(f"**Summary:** {summary}")
        if position_type:
            analysis_parts.append(f"**Type:** {position_type}")

        analysis_text = "\n\n".join(analysis_parts) if analysis_parts else response.text[:500]

        match_score = data.get("match_score", 0)
        if isinstance(match_score, str):
            try:
                match_score = int(match_score)
            except Exception:
                match_score = 0

        return VacancyAnalysis(
            suitable=suitable,
            analysis=analysis_text,
            match_score=match_score,
            recruiter_analysis=None,
        )

    except Exception as e:
        log.error(f"Vacancy analysis error: {e}")
        return VacancyAnalysis(False, f"Error: {e}")


async def ml_interesting_async(text: str) -> VacancyAnalysis:
    try:
        from .web_ui import get_current_settings

        settings = get_current_settings()
        model_type = settings.get("model_type", "mistral")
    except Exception:
        model_type = "mistral"

    return await analyze_vacancy(text, model_type)


SESSION_FILE = DATA_DIR / "session.json"


def save_session():
    if RESUME_DATA:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        session = {
            "resume_data": {k: v for k, v in RESUME_DATA.items() if k != "_original"},
            "saved_at": datetime.now().isoformat(),
        }
        try:
            with open(SESSION_FILE, "w", encoding="utf-8") as f:
                json.dump(session, f, ensure_ascii=False, indent=2)
        except Exception as e:
            log.error(f"Session save error: {e}")


def load_session():
    global RESUME_DATA
    try:
        if SESSION_FILE.exists():
            with open(SESSION_FILE, "r", encoding="utf-8") as f:
                session = json.load(f)
            RESUME_DATA = session.get("resume_data")
            if RESUME_DATA:
                log.info("Session loaded")
                return True
    except Exception as e:
        log.error(f"Session load error: {e}")
    return False


load_session()
