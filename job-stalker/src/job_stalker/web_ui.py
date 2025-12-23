from __future__ import annotations

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from typing import Any, Dict, List, Optional
import asyncio
import json
import os
import logging
from pathlib import Path
import psutil


from .vacancy_storage import (
    save_vacancy,
    load_all_vacancies,
    mark_all_as_old,
    clear_all_vacancies,
    load_tracked_vacancies,
    save_tracked_vacancies,
    add_to_tracker,
    remove_from_tracker,
    update_tracker_status,
    update_tracked_vacancy,
    get_tracked_vacancy,
    is_in_tracker,
)
from .telegram_auth import (
    get_auth_status,
    start_qr_auth,
    start_phone_auth,
    submit_code,
    submit_password,
    logout,
    get_user_info,
    set_status_callback,
    is_authorized,
)
from .state import get_state, AppState
from .gpu_monitor import get_gpu_info, has_gpu, get_gpu_detector
from .ai_client import AIClientFactory, close_session
from .models import (
    AppSettings,
    PhoneAuthRequest,
    CodeSubmitRequest,
    PasswordSubmitRequest,
    ImproveResumeRequest,
    CustomVacancyRequest,
    ResumeSetRequest,
    StatusResponse,
    StatsResponse,
    RecruiterAnalysisResult,
    VacancySource,
)
from .exceptions import (
    JobStalkerError,
    ResumeNotLoadedError,
    ValidationError,
    handle_exception,
)

log = logging.getLogger("web_ui")

try:
    from .config import GEMINI_API_KEY, GROQ_API_KEY
except ImportError:
    GEMINI_API_KEY = None
    GROQ_API_KEY = None

_gpu_detector = get_gpu_detector()
log.info(f"GPU detection: type={_gpu_detector.gpu_type}, name={_gpu_detector.gpu_name}")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Package paths
BASE_DIR = Path(__file__).parent
static_dir = BASE_DIR / "static-files"
templates_dir = BASE_DIR / "templates-html"

app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
templates = Jinja2Templates(directory=str(templates_dir))

# Settings persistence file
DATA_DIR = Path("data")
SETTINGS_FILE = DATA_DIR / "settings.json"


# Use AppSettings from models (backward compatible alias)
Settings = AppSettings


def load_settings():
    state = get_state()
    try:
        if SETTINGS_FILE.exists():
            with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
                loaded = json.load(f)
                state._settings.update(loaded)
                log.info(f"Settings loaded from {SETTINGS_FILE}")
                log.info(f"  - custom_prompt: {len(loaded.get('custom_prompt', ''))} chars")
                log.info(f"  - channels: {loaded.get('channels', [])}")
        else:
            log.warning(f"Settings file not found: {SETTINGS_FILE}")
    except Exception as e:
        log.error(f"Settings load error: {e}")


def save_settings_to_file():
    state = get_state()
    try:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
            json.dump(state.settings, f, ensure_ascii=False, indent=2)
    except Exception as e:
        log.error(f"Settings save error: {e}")


# Load settings at startup
load_settings()


@property
def monitoring_active() -> bool:
    return get_state().monitoring_active


def get_current_settings():
    return get_state().settings


def update_stats(
    found: int = None, processed: int = None, rejected: int = None, suitable: int = None
):
    state = get_state()
    if found is not None:
        state._stats.found = found
    if processed is not None:
        state._stats.processed = processed
    if rejected is not None:
        state._stats.rejected = rejected
    if suitable is not None:
        state._stats.suitable = suitable

    # Schedule async broadcast
    asyncio.create_task(broadcast_stats())


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/api/vacancies")
async def get_vacancies():
    vacancies = load_all_vacancies()
    return JSONResponse({"vacancies": vacancies})


@app.post("/api/vacancies/custom")
async def add_custom_vacancy(request: CustomVacancyRequest):
    import uuid
    from datetime import datetime
    from .ml_filter import ml_interesting_async

    state = get_state()
    settings = state.settings

    # Generate vacancy ID and prepare data
    vacancy_id = str(uuid.uuid4())
    vacancy_date = datetime.now().strftime("%Y-%m-%d %H:%M")
    channel_name = request.company or request.source.value

    # Build vacancy dict for tracker
    vacancy = {
        "id": vacancy_id,
        "channel": channel_name,
        "text": request.text,
        "date": vacancy_date,
        "link": request.link,
        "source": request.source.value,
        "title": request.title,
        "tracker_status": "wishlist",
    }

    if request.skip_analysis:
        vacancy["analysis"] = "Добавлено вручную без анализа"
        add_to_tracker(vacancy)
        return JSONResponse(
            {"status": "added", "vacancy_id": vacancy_id, "vacancy": vacancy, "analyzed": False}
        )

    # Run Stage 1 analysis
    try:
        analysis_result = await ml_interesting_async(request.text)

        if analysis_result:
            vacancy["analysis"] = analysis_result.analysis
            vacancy["match_score"] = analysis_result.match_score
            vacancy["suitable"] = analysis_result.suitable

            add_to_tracker(vacancy)

            # Stage 2 is now triggered manually via "Ask Recruiter" button

            return JSONResponse(
                {
                    "status": "added",
                    "vacancy_id": vacancy_id,
                    "vacancy": vacancy,
                    "analyzed": True,
                    "suitable": analysis_result.suitable,
                    "match_score": analysis_result.match_score,
                }
            )
        else:
            vacancy["analysis"] = "Ошибка анализа"
            add_to_tracker(vacancy)
            return JSONResponse(
                {
                    "status": "added",
                    "vacancy_id": vacancy_id,
                    "vacancy": vacancy,
                    "analyzed": False,
                    "error": "Analysis failed",
                }
            )

    except Exception as e:
        log.error(f"Error adding custom vacancy: {e}", exc_info=True)
        return JSONResponse({"status": "error", "error": str(e)}, status_code=500)


_stage2_in_progress: set[str] = set()


@app.post("/api/stage2/start")
async def api_start_stage2(request: Request):
    from .ml_filter import recruiter_analysis

    data = await request.json()
    vacancy_id = data.get("vacancy_id")
    vacancy_text = data.get("vacancy_text")
    vacancy_title = data.get("vacancy_title", "")

    if not vacancy_id or not vacancy_text:
        return JSONResponse(
            {"status": "error", "error": "Missing vacancy_id or vacancy_text"}, status_code=400
        )

    from .ml_filter import RESUME_DATA

    if not RESUME_DATA or "raw_text" not in RESUME_DATA:
        return JSONResponse({"status": "error", "error": "No resume loaded"}, status_code=400)

    state = get_state()

    if vacancy_id in _stage2_in_progress:
        return JSONResponse(
            {"status": "in_progress", "message": "Stage 2 already running for this vacancy"}
        )

    _stage2_in_progress.add(vacancy_id)

    await broadcast_message({"type": "stage2_started", "vacancy_id": vacancy_id})

    async def run_stage2():
        try:
            log.info(f"Stage 2: Starting manual analysis for {vacancy_id[:8]}")

            await broadcast_message(
                {"type": "stage2_progress", "vacancy_id": vacancy_id, "status": "analyzing"}
            )

            result = await recruiter_analysis(vacancy_text, vacancy_title)

            if result:
                # Update vacancy in storage
                from .vacancy_storage import update_vacancy

                update_vacancy(vacancy_id, {"recruiter_analysis": result.__dict__})

                await broadcast_message(
                    {
                        "type": "stage2_completed",
                        "vacancy_id": vacancy_id,
                        "recruiter_analysis": result.__dict__,
                    }
                )
                log.info(f"Stage 2: Completed for {vacancy_id[:8]}, score={result.match_score}")
            else:
                await broadcast_message(
                    {
                        "type": "stage2_error",
                        "vacancy_id": vacancy_id,
                        "error": "Analysis returned no result",
                    }
                )

        except Exception as e:
            log.error(f"Stage 2 error for {vacancy_id[:8]}: {e}")
            await broadcast_message(
                {"type": "stage2_error", "vacancy_id": vacancy_id, "error": str(e)}
            )
        finally:
            _stage2_in_progress.discard(vacancy_id)

    task = asyncio.create_task(run_stage2())
    state.track_stage2_task(task)

    return JSONResponse({"status": "started", "vacancy_id": vacancy_id})


@app.get("/api/stage2/status/{vacancy_id}")
async def api_stage2_status(vacancy_id: str):
    return JSONResponse(
        {"vacancy_id": vacancy_id, "in_progress": vacancy_id in _stage2_in_progress}
    )


@app.get("/api/tracker")
async def get_tracker():
    vacancies = load_tracked_vacancies()
    return JSONResponse({"vacancies": vacancies})


@app.post("/api/tracker/add")
async def api_add_to_tracker(request: Request):
    data = await request.json()
    vacancy = data.get("vacancy")
    if not vacancy:
        return JSONResponse({"status": "error", "error": "No vacancy data"}, status_code=400)

    success = add_to_tracker(vacancy)
    if success:
        return JSONResponse({"status": "added"})
    else:
        return JSONResponse({"status": "duplicate", "error": "Already in tracker"})


@app.post("/api/tracker/remove")
async def api_remove_from_tracker(request: Request):
    data = await request.json()
    vacancy_id = data.get("vacancy_id")
    if not vacancy_id:
        return JSONResponse({"status": "error", "error": "No vacancy_id"}, status_code=400)

    success = remove_from_tracker(vacancy_id)
    return JSONResponse({"status": "removed" if success else "not_found"})


@app.post("/api/tracker/status")
async def api_update_tracker_status(request: Request):
    data = await request.json()
    vacancy_id = data.get("vacancy_id")
    status = data.get("status")

    if not vacancy_id or not status:
        return JSONResponse(
            {"status": "error", "error": "Missing vacancy_id or status"}, status_code=400
        )

    success = update_tracker_status(vacancy_id, status)
    return JSONResponse({"status": "updated" if success else "not_found"})


@app.get("/api/tracker/{vacancy_id}")
async def get_tracked_vacancy_api(vacancy_id: str):
    vacancy = get_tracked_vacancy(vacancy_id)
    if vacancy:
        return JSONResponse({"vacancy": vacancy})
    return JSONResponse({"status": "not_found"}, status_code=404)


@app.post("/api/parse-vacancy")
async def parse_vacancy_text(request: Request):
    try:
        body = await request.json()
        text = body.get("text", "")

        if not text or len(text.strip()) < 20:
            return JSONResponse({"status": "error", "error": "Text too short"}, status_code=400)

        from .vacancy_parser import parse_vacancy
        from .vacancy_parser.parser import format_for_display

        parsed = parse_vacancy(text)
        display_data = format_for_display(parsed)

        return JSONResponse({"status": "ok", "parsed": display_data})

    except Exception as e:
        log.error(f"Parse vacancy error: {e}")
        return JSONResponse({"status": "error", "error": str(e)}, status_code=500)


@app.get("/api/session")
async def get_session():
    state = get_state()
    resume_data = None
    try:
        from .ml_filter import RESUME_DATA

        if RESUME_DATA:
            resume_data = {k: v for k, v in RESUME_DATA.items() if k != "_original"}
            resume_data["has_raw_text"] = "raw_text" in RESUME_DATA
    except Exception:
        pass

    return JSONResponse(
        {
            "settings": state.settings,
            "stats": state.get_stats_dict(),
            "resume_data": resume_data,
            "is_monitoring": state.monitoring_active,
        }
    )


@app.post("/api/start")
async def start_monitoring():
    state = get_state()

    if state.monitoring_active:
        return JSONResponse({"status": "already_running"})

    await state.reset_stats()
    mark_all_as_old()

    from .main import start_bot

    task = asyncio.create_task(start_bot())

    if await state.start_monitoring(task):
        return JSONResponse({"status": "started"})
    else:
        task.cancel()
        return JSONResponse({"status": "already_running"})


@app.post("/api/stop")
async def stop_monitoring():
    state = get_state()

    task = await state.stop_monitoring()

    if task:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    # Cancel all Stage 2 tasks
    await state.cancel_all_stage2_tasks()

    return JSONResponse({"status": "stopped"})


@app.post("/api/reset")
async def reset_all():
    state = get_state()

    clear_all_vacancies()

    try:
        from .db import reset_db

        await reset_db()
    except Exception as e:
        log.warning(f"Could not reset forwarded DB: {e}")

    await state.reset_stats()
    return JSONResponse({"status": "reset"})


@app.post("/api/upload-resume")
async def upload_resume(request: Request, model_type: str = "mistral", file_ext: str = ".txt"):
    from .ml_filter import load_resume, set_stream_callback, save_session
    import tempfile

    body = await request.body()
    state = get_state()

    # Parse file based on extension
    try:
        if file_ext.lower() == ".pdf":
            from pypdf import PdfReader

            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                tmp.write(body)
                tmp_name = tmp.name

            reader = PdfReader(tmp_name)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"

            Path(tmp_name).unlink(missing_ok=True)

        elif file_ext.lower() in [".docx", ".doc"]:
            from docx import Document

            with tempfile.NamedTemporaryFile(suffix=".docx", delete=False) as tmp:
                tmp.write(body)
                tmp_name = tmp.name

            doc = Document(tmp_name)
            text = "\n".join([paragraph.text for paragraph in doc.paragraphs])

            Path(tmp_name).unlink(missing_ok=True)

        elif file_ext.lower() in [".html", ".htm"]:
            from bs4 import BeautifulSoup

            html_content = body.decode("utf-8")
            soup = BeautifulSoup(html_content, "html.parser")

            for script in soup(["script", "style"]):
                script.decompose()

            text = soup.get_text(separator="\n", strip=True)
            lines = [line.strip() for line in text.split("\n") if line.strip()]
            text = "\n".join(lines)

        else:
            text = body.decode("utf-8")

    except Exception as e:
        return JSONResponse({"error": f"File parsing error: {str(e)}"})

    # Save extracted text to temp file for load_resume
    with tempfile.NamedTemporaryFile(suffix=".txt", mode="w", encoding="utf-8", delete=False) as tmp:
        tmp.write(text)
        temp_path = tmp.name

    set_stream_callback(broadcast_message)

    try:
        result = await load_resume(temp_path, model_type)

        if result and not result.get("error"):
            await state.update_settings({"resume_summary": result.get("summary", "")})
            save_settings_to_file()
            save_session()

        return JSONResponse(result)
    finally:
        set_stream_callback(None)
        Path(temp_path).unlink(missing_ok=True)


@app.post("/api/resume/set")
async def set_resume(request: ResumeSetRequest):
    from .ml_filter import set_resume_data, save_session

    result = await set_resume_data(request.resume_data)
    if result.get("error"):
        return JSONResponse(result, status_code=400)

    state = get_state()
    await state.update_settings({"resume_summary": result.get("summary", "")})
    save_settings_to_file()
    save_session()
    return JSONResponse({"status": "ok", "resume": result})


ImproveRequest = ImproveResumeRequest


@app.post("/api/improve-resume")
async def improve_resume_endpoint(request: ImproveResumeRequest) -> JSONResponse:
    from .ml_filter import RESUME_DATA

    if not RESUME_DATA:
        raise HTTPException(status_code=400, detail="Resume not loaded")

    state = get_state()
    vacancy_id = request.vacancy_id or str(hash(request.vacancy_text[:50]))

    # Clean up old tasks periodically
    await state.cleanup_improvement_tasks()

    # Start background task
    task = asyncio.create_task(
        run_improvement(
            vacancy_id, request.vacancy_text, request.vacancy_title, request.recruiter_analysis
        )
    )
    await state.add_improvement_task(vacancy_id, task)

    return JSONResponse({"status": "started", "vacancy_id": vacancy_id})


async def run_improvement(
    vacancy_id: str, vacancy_text: str, vacancy_title: str, existing_analysis: dict = None
):
    from .ml_filter import compare_with_resume, set_stream_callback, RecruiterAnalysis

    state = get_state()
    log.info(f"Stage 3: Starting improvement for vacancy_id={vacancy_id}")

    async def scoped_callback(msg):
        msg["vacancy_id"] = vacancy_id
        await broadcast_message(msg)

    set_stream_callback(scoped_callback)

    try:
        recruiter_analysis_obj = None
        if existing_analysis:
            # Handle both dict and Pydantic model
            if isinstance(existing_analysis, dict):
                match_score = existing_analysis.get("match_score", 0)
                if match_score > 0:
                    log.info(f"Using existing recruiter analysis (dict): match_score={match_score}")
                    recruiter_analysis_obj = RecruiterAnalysis(
                        match_score=match_score,
                        strong_sides=existing_analysis.get("strong_sides", []),
                        weak_sides=existing_analysis.get("weak_sides", []),
                        missing_skills=existing_analysis.get("missing_skills", []),
                        risks=existing_analysis.get("risks", []),
                        recommendations=existing_analysis.get("recommendations", []),
                        verdict=existing_analysis.get("verdict", ""),
                        cover_letter_hint=existing_analysis.get("cover_letter_hint", ""),
                    )
            else:
                # It's a Pydantic model (RecruiterAnalysisResult)
                match_score = getattr(existing_analysis, "match_score", 0)
                if match_score > 0:
                    log.info(
                        f"Using existing recruiter analysis (model): match_score={match_score}"
                    )
                    recruiter_analysis_obj = RecruiterAnalysis(
                        match_score=match_score,
                        strong_sides=getattr(existing_analysis, "strong_sides", []),
                        weak_sides=getattr(existing_analysis, "weak_sides", []),
                        missing_skills=getattr(existing_analysis, "missing_skills", []),
                        risks=getattr(existing_analysis, "risks", []),
                        recommendations=getattr(existing_analysis, "recommendations", []),
                        verdict=getattr(existing_analysis, "verdict", ""),
                        cover_letter_hint=getattr(existing_analysis, "cover_letter_hint", ""),
                    )

        comparison = await compare_with_resume(vacancy_text, vacancy_title, recruiter_analysis_obj)

        log.info(f"Stage 3 completed: match_score={comparison.match_score}")

        result = {
            "match_score": comparison.match_score,
            "strong_sides": comparison.strong_sides,
            "weak_sides": comparison.weak_sides,
            "missing_skills": comparison.missing_skills,
            "recommendations": comparison.recommendations,
            "cover_letter_hint": comparison.cover_letter_hint,
            "improved_resume": comparison.improved_resume,
        }

        await state.update_improvement_task(vacancy_id, "completed", result)

        message = {"type": "resume_improved", "vacancy_id": vacancy_id, "result": result}

        await broadcast_message(message)
        log.info(f"Message broadcasted to {state.ws_client_count} connections")

    except Exception as e:
        log.error(f"Error in run_improvement: {e}", exc_info=True)
        await state.update_improvement_task(vacancy_id, "error")
        await broadcast_message(
            {"type": "resume_improved", "vacancy_id": vacancy_id, "error": str(e)}
        )
    finally:
        set_stream_callback(None)


@app.get("/api/improve-resume/{vacancy_id}")
async def get_improvement_status(vacancy_id: str):
    state = get_state()
    info = await state.get_improvement_task(vacancy_id)

    if not info:
        return JSONResponse({"status": "not_found"})

    return JSONResponse({"status": info.get("status"), "result": info.get("result")})


@app.post("/api/settings")
async def save_settings(settings: Settings):
    state = get_state()
    await state.update_settings(
        {
            "model_type": settings.model_type,
            "days_back": settings.days_back,
            "custom_prompt": settings.custom_prompt,
            "resume_summary": settings.resume_summary,
            "channels": settings.channels,
            "keyword_filter": settings.keyword_filter,
            "search_mode": settings.search_mode,
            # enable_stage2 removed - Stage 2 is now triggered manually
        }
    )

    save_settings_to_file()
    log.info(
        f"Settings saved: mode={settings.search_mode}, prompt={len(settings.custom_prompt)} chars"
    )
    return JSONResponse({"status": "saved"})


@app.get("/api/settings")
async def get_settings_endpoint():
    return JSONResponse(get_state().settings)


@app.get("/api/models")
async def get_models():
    ollama_client = AIClientFactory.get_ollama_client()
    models = await ollama_client.list_models()

    if GEMINI_API_KEY:
        models.append(
            {"name": "gemini", "display_name": "Gemini (Cloud)", "size": 0, "provider": "gemini"}
        )

    if GROQ_API_KEY:
        from .ai_client import GroqClient

        groq_client = GroqClient(GROQ_API_KEY)
        groq_models = await groq_client.list_models()
        models.extend(groq_models)

    return JSONResponse({"models": models})


@app.get("/api/auth/status")
async def auth_status():
    status = await get_auth_status()
    user_info = await get_user_info() if status.get("authorized") else None

    return JSONResponse({"status": status, "user": user_info})


@app.post("/api/auth/qr")
async def auth_qr():
    set_status_callback(broadcast_message)
    result = await start_qr_auth()
    return JSONResponse(result)


@app.post("/api/auth/phone")
async def auth_phone(request: PhoneAuthRequest):
    set_status_callback(broadcast_message)
    result = await start_phone_auth(request.phone)
    return JSONResponse(result)


@app.post("/api/auth/code")
async def auth_code(request: CodeSubmitRequest):
    result = await submit_code(request.code)
    return JSONResponse(result)


@app.post("/api/auth/password")
async def auth_password(request: PasswordSubmitRequest):
    result = await submit_password(request.password)
    return JSONResponse(result)


@app.post("/api/auth/logout")
async def auth_logout():
    result = await logout()
    return JSONResponse(result)


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    state = get_state()
    await ws.accept()
    await state.add_ws_client(ws)
    log.info(f"WebSocket connected. Total: {state.ws_client_count}")

    try:
        await ws.send_json({"type": "stats", "stats": state.get_stats_dict()})
        await ws.send_json({"type": "monitoring", "active": state.monitoring_active})
    except Exception:
        pass

    try:
        while True:
            data = await ws.receive_text()
    except WebSocketDisconnect:
        pass
    except Exception as e:
        log.debug(f"WS error: {e}")
    finally:
        await state.remove_ws_client(ws)
        log.info(f"WebSocket disconnected. Total: {state.ws_client_count}")


async def broadcast_vacancy(vacancy: dict):
    save_vacancy(vacancy)
    state = get_state()
    await state.increment_stats(suitable=1)

    await broadcast_message({"type": "vacancy", "vacancy": vacancy})
    await broadcast_message({"type": "stats", "stats": state.get_stats_dict()})


async def broadcast_status(message: str, icon: str = ""):
    await broadcast_message({"type": "status", "message": message, "icon": icon})


async def broadcast_stats():
    state = get_state()
    await broadcast_message({"type": "stats", "stats": state.get_stats_dict()})


async def broadcast_progress(percent: int, remaining: int = None):
    msg = {"type": "progress", "percent": percent}
    if remaining is not None:
        msg["remaining"] = remaining
    await broadcast_message(msg)


async def broadcast_message(message: dict):
    state = get_state()
    clients = await state.get_ws_clients()
    dead_clients = []
    msg_type = message.get("type", "unknown")

    for client in clients:
        try:
            await client.send_json(message)
        except Exception as e:
            log.debug(f"Failed to send {msg_type} to client: {e}")
            dead_clients.append(client)

    if dead_clients:
        await state.cleanup_ws_clients(dead_clients)


def get_cpu_info():
    try:
        cpu_percent = psutil.cpu_percent(interval=None)
        mem = psutil.virtual_memory()
        cpu_count = psutil.cpu_count(logical=True)

        try:
            freq = psutil.cpu_freq()
            cpu_freq = freq.current if freq else 0
        except Exception:
            cpu_freq = 0

        return {
            "utilization": cpu_percent,
            "cores": cpu_count,
            "frequency_mhz": round(cpu_freq),
            "memory_percent": mem.percent,
            "memory_used_gb": round(mem.used / (1024**3), 1),
            "memory_total_gb": round(mem.total / (1024**3), 1),
        }
    except Exception as e:
        log.error(f"CPU info error: {e}")
        return {
            "utilization": 0,
            "cores": 0,
            "frequency_mhz": 0,
            "memory_percent": 0,
            "memory_used_gb": 0,
            "memory_total_gb": 0,
        }


@app.get("/api/system-monitor")
async def get_system_monitor():
    gpu_info_dict = get_gpu_info()
    cpu_info = get_cpu_info()

    return {
        "gpu": gpu_info_dict,
        "cpu": cpu_info,
        "has_gpu": gpu_info_dict is not None and gpu_info_dict.get("available", False),
    }


async def monitor_loop():
    state = get_state()
    while state.monitor_loop_active:
        gpu_info_dict = get_gpu_info()
        cpu_info = get_cpu_info()

        await broadcast_message(
            {
                "type": "system_monitor",
                "gpu": gpu_info_dict,
                "cpu": cpu_info,
                "has_gpu": gpu_info_dict is not None,
            }
        )

        await asyncio.sleep(1)


@app.post("/api/monitor/start")
async def start_monitor():
    state = get_state()
    if not state.monitor_loop_active:
        await state.set_monitor_loop_active(True)
        asyncio.create_task(monitor_loop())
    return {"status": "started"}


@app.post("/api/monitor/stop")
async def stop_monitor():
    state = get_state()
    await state.set_monitor_loop_active(False)
    return {"status": "stopped"}


@app.on_event("shutdown")
async def shutdown_event():
    state = get_state()

    # Stop monitoring
    task = await state.stop_monitoring()
    if task:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    await state.cancel_all_stage2_tasks()

    await state.set_monitor_loop_active(False)

    await close_session()

    log.info("Application shutdown complete")
