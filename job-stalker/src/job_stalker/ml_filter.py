import asyncio
import re
import os
import logging
import json
from datetime import datetime
from typing import Dict, Optional, Callable, AsyncGenerator, List
import aiohttp
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from dataclasses import dataclass, field

log = logging.getLogger("ml_filter")
console = Console()

# URL для Ollama API (локальный)
OLLAMA_API_URL = "http://localhost:11434/api/generate"

# Импортируем API ключ из config
try:
    from .config import GEMINI_API_KEY
except ImportError:
    GEMINI_API_KEY = None

# Глобальная переменная для хранения данных резюме
RESUME_DATA: Optional[Dict] = None

# Output directory для PDF
OUTPUT_DIR = "./output"


@dataclass
class ResumeComparison:
    """Результат сравнения вакансии с резюме"""
    match_score: int = 0
    strong_sides: List[str] = field(default_factory=list)
    weak_sides: List[str] = field(default_factory=list)
    missing_skills: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    improved_resume: str = ""
    cover_letter_hint: str = ""

@dataclass
class VacancyAnalysis:
    """Результат анализа вакансии"""
    suitable: bool
    analysis: str = ""
    comparison: Optional[ResumeComparison] = None
    improved_resume_path: Optional[str] = None
    
    def __bool__(self):
        return self.suitable



# Дефолтный пример промпта для UI (пользователь может его редактировать)
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
    """
    Генерирует промпт для фильтрации вакансий.
    
    Логика:
    - Если custom_prompt пустой -> все вакансии подходят (suitable: true)
    - Если custom_prompt заполнен -> LLM фильтрует по критериям
    """
    
    # Если промпт пустой - не нужен LLM анализ, все подходят
    if not custom_prompt or not custom_prompt.strip():
        return ""
    
    # Базовый промпт с критериями пользователя
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
  "summary": "краткий вывод на русском"
}}"""

    if resume_summary:
        base = f"Резюме кандидата: {resume_summary}\n\n{base}"
    
    return base


# Для обратной совместимости
def get_default_prompt(resume_summary: str = "") -> str:
    return get_filter_prompt(DEFAULT_FILTER_PROMPT_EXAMPLE, resume_summary)


def get_comparison_prompt(vacancy_text: str, resume_text: str) -> str:
    """Промпт для сравнения вакансии с резюме"""
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
если навыка нет и нельзя предположить —

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

Верни ТОЛЬКО JSON (без markdown):
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

_stream_callback: Optional[Callable] = None

def set_stream_callback(callback: Optional[Callable]):
    """Устанавливает callback для streaming"""
    global _stream_callback
    _stream_callback = callback

async def notify_stream(chunk: str, stream_type: str = "analysis"):
    """Отправляет чанк через callback"""
    if _stream_callback:
        try:
            await _stream_callback({
                "type": "stream",
                "stream_type": stream_type,
                "chunk": chunk
            })
        except Exception as e:
            log.warning(f"Stream callback error: {e}")

async def ollama_stream(prompt: str, model: str = "mistral7") -> AsyncGenerator[str, None]:
    """Streaming генерация через Ollama API"""
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": True
    }
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                OLLAMA_API_URL, 
                json=payload,
                timeout=aiohttp.ClientTimeout(total=300)
            ) as response:
                if response.status != 200:
                    error = await response.text()
                    log.error(f"Ollama API error: {error}")
                    yield f"[ERROR: {response.status}]"
                    return
                
                async for line in response.content:
                    if line:
                        try:
                            data = json.loads(line.decode('utf-8'))
                            chunk = data.get('response', '')
                            if chunk:
                                yield chunk
                            if data.get('done', False):
                                break
                        except json.JSONDecodeError:
                            continue
    except aiohttp.ClientError as e:
        log.error(f"Ollama connection error: {e}")
        yield f"[ERROR: {e}]"
    except Exception as e:
        log.error(f"Ollama stream error: {e}")
        yield f"[ERROR: {e}]"

async def ollama_generate(prompt: str, model: str = "mistral7", stream_type: str = None) -> str:
    """Генерация с опциональным streaming"""
    full_response = ""
    
    if stream_type:
        await notify_stream("[START]", stream_type)
    
    async for chunk in ollama_stream(prompt, model):
        full_response += chunk
        if stream_type:
            await notify_stream(chunk, stream_type)
            await asyncio.sleep(0.005)  # Уменьшил задержку
    
    if stream_type:
        await notify_stream("[END]", stream_type)
    
    return full_response



def extract_json_safely(text: str) -> dict:
    """Безопасное извлечение JSON из текста AI"""
    text = text.replace('```json', '').replace('```', '').strip()
    
    # Метод 1: Сбалансированные скобки
    depth = 0
    start = -1
    for i, char in enumerate(text):
        if char == '{':
            if depth == 0:
                start = i
            depth += 1
        elif char == '}':
            depth -= 1
            if depth == 0 and start != -1:
                try:
                    return json.loads(text[start:i+1])
                except json.JSONDecodeError:
                    continue
    
    # Метод 2: Regex
    json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(0))
        except:
            pass
    
    # Метод 3: Greedy
    json_match = re.search(r'\{.*\}', text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(0))
        except:
            pass
    
    return {}



def normalize_resume_data(data: dict) -> dict:
    """Нормализует данные резюме"""
    normalized = {}
    
    # Experience
    if 'experience_years' in data:
        normalized['experience_years'] = data['experience_years']
    elif 'experience' in data:
        exp = data['experience']
        if isinstance(exp, list):
            normalized['experience_years'] = len(exp) * 2
            projects = []
            for item in exp:
                if isinstance(item, dict):
                    company = item.get('company', item.get('project', ''))
                    position = item.get('positionTitle', item.get('position', ''))
                    if company or position:
                        projects.append(f"{position} @ {company}".strip(' @'))
            if projects:
                normalized['projects'] = projects
        elif isinstance(exp, (int, float)):
            normalized['experience_years'] = exp
    
    # Level
    if 'level' in data:
        normalized['level'] = data['level']
    else:
        years = normalized.get('experience_years', 0)
        if isinstance(years, (int, float)):
            normalized['level'] = 'junior' if years <= 2 else 'middle' if years <= 5 else 'senior'
    
    # Skills
    skills = []
    if 'key_skills' in data:
        skills = data['key_skills']
    elif 'skills' in data:
        sk = data['skills']
        if isinstance(sk, list):
            skills = sk
        elif isinstance(sk, dict):
            for items in sk.values():
                if isinstance(items, list):
                    skills.extend(items)
    normalized['key_skills'] = skills[:10]
    
    if 'projects' not in normalized:
        normalized['projects'] = data.get('projects', [])
    normalized['summary'] = data.get('summary', '')
    if 'name' in data:
        normalized['name'] = data['name']
    
    return normalized


async def load_resume(file_path: str, model_type: str = "mistral") -> dict:
    """Загрузка и анализ резюме"""
    global RESUME_DATA
    
    log.info(f"🚀 load_resume: model={model_type}, file={file_path}")
    
    if not os.path.exists(file_path):
        return {"error": "File not found"}
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            resume_text = f.read()
        log.info(f"📄 Resume: {len(resume_text)} chars")
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
        if model_type == "gemini" and GEMINI_API_KEY:
            output = await _call_gemini(prompt)
        elif model_type == "mistral":
            output = await ollama_generate(prompt, "mistral7", "resume_analysis")
        else:
            output = await ollama_generate(prompt, "llama3.2:3b", "resume_analysis")
        
        raw_data = extract_json_safely(output)
        RESUME_DATA = normalize_resume_data(raw_data)
        RESUME_DATA['raw_text'] = resume_text
        
        log.info(f"✅ Resume: level={RESUME_DATA.get('level')}, exp={RESUME_DATA.get('experience_years')}")
        
        return RESUME_DATA
        
    except Exception as e:
        log.error(f"Resume analysis error: {e}")
        return {"error": str(e)}


async def _call_gemini(prompt: str) -> str:
    """Вызов Gemini API"""
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={GEMINI_API_KEY}"
    
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.3, "maxOutputTokens": 1000}
    }
    
    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=30)) as response:
            if response.status != 200:
                raise Exception(f"Gemini API error {response.status}")
            data = await response.json()
    
    return data['candidates'][0]['content']['parts'][0]['text'].strip()



async def compare_with_resume(vacancy_text: str, vacancy_title: str = "") -> ResumeComparison:
    """Сравнение вакансии с резюме - вызывается ОТДЕЛЬНО по запросу"""
    if not RESUME_DATA or 'raw_text' not in RESUME_DATA:
        log.warning("No resume for comparison")
        return ResumeComparison()
    
    resume_text = RESUME_DATA['raw_text']
    log.info(f"📝 Resume text length: {len(resume_text)}")
    log.info(f"📝 Vacancy text length: {len(vacancy_text)}")
    
    prompt = get_comparison_prompt(vacancy_text, resume_text)
    
    log.info("🔄 Comparing vacancy with resume...")
    
    try:
        output = await ollama_generate(prompt, "mistral7", "comparison")
        log.info(f"📝 ML output length: {len(output)}")
        
        data = extract_json_safely(output)
        log.info(f"📝 Parsed data keys: {list(data.keys())}")
        log.info(f"📝 improved_resume length: {len(data.get('improved_resume', ''))}")
        
        result = ResumeComparison(
            match_score=data.get('match_score', 0),
            strong_sides=data.get('strong_sides', []),
            weak_sides=data.get('weak_sides', []),
            missing_skills=data.get('missing_skills', []),
            recommendations=data.get('recommendations', []),
            improved_resume=data.get('improved_resume', ''),
            cover_letter_hint=data.get('cover_letter_hint', '')
        )
        
        log.info(f"✅ Comparison done: score={result.match_score}, improved_len={len(result.improved_resume)}")
        return result
        
    except Exception as e:
        log.error(f"Comparison error: {e}")
        return ResumeComparison()



async def analyze_vacancy(text: str, model_type: str = "mistral") -> VacancyAnalysis:
    """
    Анализ вакансии - БЕЗ comparison.
    
    Логика:
    - Если custom_prompt пустой -> вакансия автоматически подходит
    - Если custom_prompt заполнен -> LLM фильтрует по критериям
    """
    if len(text.strip()) < 20:
        return VacancyAnalysis(False, "Текст слишком короткий")
    
    try:
        from .web_ui import get_current_settings
        settings = get_current_settings()
        custom_prompt = settings.get("custom_prompt", "")
        resume_summary = settings.get("resume_summary", "")
        log.info(f"📋 Loaded custom_prompt length: {len(custom_prompt)} chars")
        log.info(f"📋 First 100 chars of custom_prompt: {custom_prompt[:100]}")
    except Exception as e:
        log.error(f"⚠️ Failed to load settings: {e}")
        custom_prompt = ""
        resume_summary = ""

    # НОВАЯ ЛОГИКА: если промпт пустой - все вакансии подходят
    if not custom_prompt or not custom_prompt.strip():
        log.info("📋 Filter prompt empty - vacancy auto-approved")
        return VacancyAnalysis(
            suitable=True,
            analysis="✅ Фильтр не настроен — вакансия добавлена автоматически.\n\n💡 Настройте критерии в Settings → Search Filter Prompt",
            comparison=None
        )

    # Генерируем промпт с критериями пользователя
    filter_prompt = get_filter_prompt(custom_prompt, resume_summary)
    log.info(f"📋 Generated filter_prompt length: {len(filter_prompt)} chars")

    vacancy_text_short = text.strip()[:1500]
    full_prompt = f"{filter_prompt}\n\nВАКАНСИЯ:\n{vacancy_text_short}\n\nJSON:"

    log.info(f"📊 Analyzing with {model_type.upper()}...")
    log.info(f"📄 Vacancy text (first 200 chars): {text.strip()[:200]}")
    
    try:
        if model_type == "mistral":
            output = await ollama_generate(full_prompt, "mistral7")
        elif model_type == "gemini" and GEMINI_API_KEY:
            output = await _call_gemini(full_prompt)
        else:
            output = await ollama_generate(full_prompt, "llama3.2:3b")
        
        data = extract_json_safely(output)

        suitable = data.get('suitable', False)
        if isinstance(suitable, str):
            suitable = suitable.lower() in ('true', 'yes', 'да', '1')

        # Проверка position_type (если указан)
        position_type = data.get('position_type', '').lower()

        # Логируем результат анализа
        log.info(f"🤖 LLM Response: suitable={suitable}, position_type={position_type}")
        log.info(f"📝 LLM summary: {data.get('summary', 'N/A')[:150]}")

        # ЗАЩИТА ОТ ПРОТИВОРЕЧИЙ LLM
        # Если LLM вернул suitable=True, но position_type упоминается в критериях пользователя
        # в негативном контексте (отбросить, не подходит, исключить и т.д.) - это ошибка LLM
        if suitable and position_type:
            # Словарь переводов ролей (английский -> русские варианты)
            role_translations = {
                'artist': ['художник', 'артист'],
                'designer': ['дизайнер', 'design'],
                'manager': ['менеджер', 'управляющий', 'руководитель'],
                'qa': ['тестировщик', 'тестер', 'качество'],
                'producer': ['продюсер'],
                'marketing': ['маркетолог', 'маркетинг'],
                'animator': ['аниматор', 'анимация'],
                'modeller': ['моделлер', 'модельер', '3d model'],
                'hr': ['рекрутер', 'hr специалист']
            }

            # Собираем все возможные варианты названий для этой роли
            role_variants = [position_type.lower()]
            for eng, rus_list in role_translations.items():
                if eng in position_type.lower():
                    role_variants.extend(rus_list)

            # Ищем упоминание этой роли в негативном контексте в промпте пользователя
            negative_markers = ['не подходит', 'не хочу', 'отбрасывать', 'исключить', 'отбросить',
                              'не твоего направления', 'не по профилю', 'без', 'кроме',
                              '❌', 'NOT suitable', 'exclude', 'не относится']

            # Проверяем каждый маркер
            for marker in negative_markers:
                if marker.lower() in custom_prompt.lower():
                    # Ищем упоминание position_type рядом с негативным маркером
                    # Разбиваем промпт на части по негативным маркерам
                    parts = custom_prompt.lower().split(marker.lower())
                    for part in parts[1:]:  # Проверяем текст ПОСЛЕ негативного маркера
                        # Берем следующие 500 символов после маркера
                        context = part[:500]
                        # Проверяем есть ли там упоминание любого варианта роли
                        for variant in role_variants:
                            if variant in context:
                                log.warning(f"⚠️ CONTRADICTION: position_type='{position_type}' (variant: '{variant}') found in negative context near '{marker}'")
                                log.warning(f"   LLM said suitable=True but this role is explicitly excluded. Overriding to suitable=False")
                                suitable = False
                                # Добавляем причину отклонения
                                if not data.get('reasons_reject'):
                                    data['reasons_reject'] = []
                                if isinstance(data['reasons_reject'], str):
                                    data['reasons_reject'] = [data['reasons_reject']]
                                data['reasons_reject'].append(f"Роль '{position_type}' явно исключена в критериях пользователя")
                                break
                        if not suitable:  # Если уже исправили - выходим
                            break
                    if not suitable:  # Если уже исправили - выходим
                        break

        # Формируем анализ
        analysis_parts = []

        reasons_fit = data.get('reasons_fit', [])
        reasons_reject = data.get('reasons_reject', data.get('reasons_lack', []))
        summary = data.get('summary', '')
        
        if isinstance(reasons_fit, str):
            reasons_fit = [reasons_fit]
        if isinstance(reasons_reject, str):
            reasons_reject = [reasons_reject]
        
        if reasons_fit:
            analysis_parts.append("✅ **Подходит:**\n" + "\n".join(f"  • {r}" for r in reasons_fit))
        if reasons_reject:
            analysis_parts.append("❌ **Не подходит:**\n" + "\n".join(f"  • {r}" for r in reasons_reject))
        if summary:
            analysis_parts.append(f"📋 **Вывод:** {summary}")
        if position_type:
            analysis_parts.append(f"🏷️ **Тип:** {position_type}")
        
        analysis_text = "\n\n".join(analysis_parts) if analysis_parts else output[:500]
        
        return VacancyAnalysis(
            suitable=suitable,
            analysis=analysis_text,
            comparison=None
        )
        
    except Exception as e:
        log.error(f"Vacancy analysis error: {e}")
        return VacancyAnalysis(False, f"⚠️ Ошибка: {e}")


async def ml_interesting_async(text: str) -> VacancyAnalysis:
    """Главная функция анализа"""
    try:
        from .web_ui import get_current_settings
        settings = get_current_settings()
        model_type = settings.get("model_type", "mistral")
    except:
        model_type = "mistral"
    
    return await analyze_vacancy(text, model_type)



SESSION_FILE = "./data/session.json"

def save_session():
    """Сохраняет сессию"""
    if RESUME_DATA:
        os.makedirs("./data", exist_ok=True)
        session = {
            "resume_data": {k: v for k, v in RESUME_DATA.items() if k != '_original'},
            "saved_at": datetime.now().isoformat()
        }
        try:
            with open(SESSION_FILE, 'w', encoding='utf-8') as f:
                json.dump(session, f, ensure_ascii=False, indent=2)
        except Exception as e:
            log.error(f"Session save error: {e}")

def load_session():
    """Загружает сессию"""
    global RESUME_DATA
    try:
        if os.path.exists(SESSION_FILE):
            with open(SESSION_FILE, 'r', encoding='utf-8') as f:
                session = json.load(f)
            RESUME_DATA = session.get('resume_data')
            if RESUME_DATA:
                log.info(f"📂 Session loaded")
                return True
    except Exception as e:
        log.error(f"Session load error: {e}")
    return False

# Загружаем при импорте
load_session()
