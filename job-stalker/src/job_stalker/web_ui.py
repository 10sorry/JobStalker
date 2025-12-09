from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import asyncio
import json
import os
import re
import subprocess
import psutil
from .vacancy_storage import (save_vacancy, load_all_vacancies,
                              mark_all_as_old, clear_all_vacancies)
from .telegram_auth import (
    get_auth_status, start_qr_auth, start_phone_auth,
    submit_code, submit_password, logout, get_user_info,
    set_status_callback, is_authorized
)

# ============== GPU DETECTION ==============
# Определяем какой GPU доступен

GPU_TYPE = None  # 'nvidia', 'amd', or None
GPU_NAME = None

# 1. Пробуем NVIDIA (pynvml)
try:
    import pynvml
    pynvml.nvmlInit()
    if pynvml.nvmlDeviceGetCount() > 0:
        GPU_TYPE = 'nvidia'
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        GPU_NAME = pynvml.nvmlDeviceGetName(handle)
        if isinstance(GPU_NAME, bytes):
            GPU_NAME = GPU_NAME.decode('utf-8')
        print(f"Detected NVIDIA GPU: {GPU_NAME}")
except Exception as e:
    print(f"NVIDIA detection failed: {e}")

# 2. Пробуем AMD (pyamdgpuinfo)
if GPU_TYPE is None:
    try:
        import pyamdgpuinfo
        if pyamdgpuinfo.detect_gpus() > 0:
            GPU_TYPE = 'amd'
            gpu = pyamdgpuinfo.get_gpu(0)
            GPU_NAME = gpu.name if hasattr(gpu, 'name') else 'AMD GPU'
            print(f"Detected AMD GPU via pyamdgpuinfo: {GPU_NAME}")
    except Exception as e:
        print(f"AMD pyamdgpuinfo detection failed: {e}")

# 3. Пробуем AMD через rocm-smi (CLI)
if GPU_TYPE is None:
    try:
        result = subprocess.run(['rocm-smi', '--showproductname'], 
                                capture_output=True, text=True, timeout=5)
        if result.returncode == 0 and 'GPU' in result.stdout:
            GPU_TYPE = 'amd_rocm'
            # Парсим название
            for line in result.stdout.split('\n'):
                if 'Card series' in line or 'GPU' in line:
                    GPU_NAME = line.split(':')[-1].strip() if ':' in line else 'AMD GPU'
                    break
            GPU_NAME = GPU_NAME or 'AMD GPU'
            print(f"Detected AMD GPU via rocm-smi: {GPU_NAME}")
    except Exception as e:
        print(f"AMD rocm-smi detection failed: {e}")

# 4. Пробуем через /sys/class/drm (Linux fallback для AMD)
if GPU_TYPE is None:
    try:
        drm_path = '/sys/class/drm'
        if os.path.exists(drm_path):
            for card in os.listdir(drm_path):
                if card.startswith('card') and card[4:].isdigit():
                    device_path = os.path.join(drm_path, card, 'device')
                    vendor_path = os.path.join(device_path, 'vendor')
                    if os.path.exists(vendor_path):
                        with open(vendor_path) as f:
                            vendor = f.read().strip()
                        # AMD vendor ID = 0x1002
                        if vendor == '0x1002':
                            GPU_TYPE = 'amd_sysfs'
                            # Пробуем получить имя
                            name_path = os.path.join(device_path, 'product_name')
                            if os.path.exists(name_path):
                                with open(name_path) as f:
                                    GPU_NAME = f.read().strip()
                            else:
                                GPU_NAME = 'AMD GPU'
                            print(f"Detected AMD GPU via sysfs: {GPU_NAME}")
                            break
    except Exception as e:
        print(f"AMD sysfs detection failed: {e}")

print(f"Final GPU detection: type={GPU_TYPE}, name={GPU_NAME}")

app = FastAPI()

# Пути внутри пакета
BASE_DIR = os.path.dirname(__file__)
static_dir = os.path.join(BASE_DIR, "static-files")
templates_dir = os.path.join(BASE_DIR, "templates-html")

app.mount("/static", StaticFiles(directory=static_dir), name="static")
templates = Jinja2Templates(directory=templates_dir)

# Глобальные переменные
clients = []
monitoring_active = False
monitoring_task = None

# Background tasks для улучшения резюме
improvement_tasks = {}

# Статистика
stats = {
    "found": 0,
    "processed": 0,
    "rejected": 0,
    "suitable": 0
}

# Настройки
current_settings = {
    "model_type": "mistral",
    "days_back": 7,
    "custom_prompt": "",
    "resume_summary": "",
    "channels": []
}

# Файлы персистентности
SETTINGS_FILE = "data/settings.json"

class Settings(BaseModel):
    model_type: str
    days_back: int
    custom_prompt: str
    resume_summary: str = ""
    channels: list = []


def load_settings():
    """Загрузка настроек"""
    global current_settings
    try:
        if os.path.exists(SETTINGS_FILE):
            with open(SETTINGS_FILE, 'r', encoding='utf-8') as f:
                loaded = json.load(f)
                current_settings.update(loaded)
                print(f"✅ Settings loaded from {SETTINGS_FILE}")
                print(f"   - custom_prompt: {len(loaded.get('custom_prompt', ''))} chars")
                print(f"   - channels: {loaded.get('channels', [])}")
        else:
            print(f"⚠️ Settings file not found: {SETTINGS_FILE}")
    except Exception as e:
        print(f"❌ Settings load error: {e}")


def save_settings_to_file():
    """Сохранение настроек"""
    try:
        os.makedirs("data", exist_ok=True)
        with open(SETTINGS_FILE, 'w', encoding='utf-8') as f:
            json.dump(current_settings, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"Settings save error: {e}")


# Загружаем при старте
load_settings()


# ============== API ==============

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/api/vacancies")
async def get_vacancies():
    vacancies = load_all_vacancies()
    return JSONResponse({"vacancies": vacancies})


@app.get("/api/session")
async def get_session():
    """Полное состояние сессии для восстановления UI"""
    resume_data = None
    try:
        from .ml_filter import RESUME_DATA
        if RESUME_DATA:
            # Включаем raw_text для отображения в UI
            resume_data = {k: v for k, v in RESUME_DATA.items() if k != '_original'}
            resume_data['has_raw_text'] = 'raw_text' in RESUME_DATA
    except:
        pass
    
    return JSONResponse({
        "settings": current_settings,
        "stats": stats,
        "resume_data": resume_data,
        "is_monitoring": monitoring_active
    })


@app.post("/api/start")
async def start_monitoring():
    global monitoring_active, monitoring_task, stats
    
    if monitoring_active:
        return JSONResponse({"status": "already_running"})
    
    # Сброс статистики
    stats = {"found": 0, "processed": 0, "rejected": 0, "suitable": 0}
    
    monitoring_active = True
    mark_all_as_old()
    
    from .main import start_bot
    monitoring_task = asyncio.create_task(start_bot())
    
    return JSONResponse({"status": "started"})


@app.post("/api/stop")
async def stop_monitoring():
    global monitoring_active, monitoring_task
    
    monitoring_active = False
    
    if monitoring_task:
        monitoring_task.cancel()
        try:
            await monitoring_task
        except asyncio.CancelledError:
            pass
        monitoring_task = None
    
    return JSONResponse({"status": "stopped"})


@app.post("/api/reset")
async def reset_all():
    global stats
    
    # Очищаем вакансии
    clear_all_vacancies()
    
    # Очищаем кэш обработанных сообщений (чтобы при новом скане всё пересканировалось)
    try:
        from .db import reset_db
        await reset_db()
    except Exception as e:
        print(f"Warning: Could not reset forwarded DB: {e}")
    
    stats = {"found": 0, "processed": 0, "rejected": 0, "suitable": 0}
    return JSONResponse({"status": "reset"})


@app.post("/api/upload-resume")
async def upload_resume(request: Request, model_type: str = "mistral"):
    from .ml_filter import load_resume, set_stream_callback, save_session
    import tempfile
    
    body = await request.body()
    text = body.decode('utf-8')
    
    temp_path = tempfile.mktemp(suffix=".txt")
    with open(temp_path, 'w', encoding='utf-8') as f:
        f.write(text)
    
    set_stream_callback(broadcast_message)
    
    try:
        result = await load_resume(temp_path, model_type)
        
        if result and not result.get("error"):
            current_settings["resume_summary"] = result.get("summary", "")
            save_settings_to_file()
            save_session()  # Сохраняем резюме в сессию!
        
        return JSONResponse(result)
    finally:
        set_stream_callback(None)
        try:
            os.unlink(temp_path)
        except:
            pass


class ImproveRequest(BaseModel):
    vacancy_text: str
    vacancy_title: str = ""
    vacancy_id: str = ""


@app.post("/api/improve-resume")
async def improve_resume_endpoint(request: ImproveRequest):
    """Запуск улучшения резюме в фоне"""
    from .ml_filter import RESUME_DATA
    
    if not RESUME_DATA:
        return JSONResponse({"error": "Resume not loaded"})
    
    vacancy_id = request.vacancy_id or str(hash(request.vacancy_text[:50]))
    
    # Запускаем в фоне
    task = asyncio.create_task(
        run_improvement(vacancy_id, request.vacancy_text, request.vacancy_title)
    )
    improvement_tasks[vacancy_id] = {"task": task, "status": "running"}
    
    return JSONResponse({"status": "started", "vacancy_id": vacancy_id})


async def run_improvement(vacancy_id: str, vacancy_text: str, vacancy_title: str):
    """Фоновая генерация улучшенного резюме"""
    from .ml_filter import compare_with_resume, set_stream_callback
    
    async def scoped_callback(msg):
        msg["vacancy_id"] = vacancy_id
        await broadcast_message(msg)
    
    set_stream_callback(scoped_callback)
    
    try:
        comparison = await compare_with_resume(vacancy_text, vacancy_title)
        
        result = {
            "match_score": comparison.match_score,
            "strong_sides": comparison.strong_sides,
            "weak_sides": comparison.weak_sides,
            "missing_skills": comparison.missing_skills,
            "recommendations": comparison.recommendations,
            "cover_letter_hint": comparison.cover_letter_hint,
            "improved_resume": comparison.improved_resume,
        }
        
        improvement_tasks[vacancy_id]["status"] = "completed"
        improvement_tasks[vacancy_id]["result"] = result
        
        await broadcast_message({
            "type": "resume_improved",
            "vacancy_id": vacancy_id,
            "result": result
        })
        
    except Exception as e:
        improvement_tasks[vacancy_id]["status"] = "error"
        await broadcast_message({
            "type": "resume_improved",
            "vacancy_id": vacancy_id,
            "error": str(e)
        })
    finally:
        set_stream_callback(None)


@app.get("/api/improve-resume/{vacancy_id}")
async def get_improvement_status(vacancy_id: str):
    if vacancy_id not in improvement_tasks:
        return JSONResponse({"status": "not_found"})
    
    info = improvement_tasks[vacancy_id]
    return JSONResponse({
        "status": info.get("status"),
        "result": info.get("result")
    })


@app.post("/api/settings")
async def save_settings(settings: Settings):
    global current_settings
    current_settings["model_type"] = settings.model_type
    current_settings["days_back"] = settings.days_back
    current_settings["custom_prompt"] = settings.custom_prompt
    current_settings["resume_summary"] = settings.resume_summary
    current_settings["channels"] = settings.channels

    save_settings_to_file()
    print(f"💾 Settings saved:")
    print(f"   - custom_prompt: {len(settings.custom_prompt)} chars")
    print(f"   - resume_summary: {len(settings.resume_summary)} chars")
    return JSONResponse({"status": "saved"})


@app.get("/api/settings")
async def get_settings():
    return JSONResponse(current_settings)


# ============== TELEGRAM AUTH ==============

@app.get("/api/auth/status")
async def auth_status():
    """Проверка статуса авторизации"""
    status = await get_auth_status()
    user_info = await get_user_info() if status.get("authorized") else None

    return JSONResponse({
        "status": status,
        "user": user_info
    })


@app.post("/api/auth/qr")
async def auth_qr():
    """Запуск QR-авторизации"""
    set_status_callback(broadcast_message)
    result = await start_qr_auth()
    return JSONResponse(result)


class PhoneAuthRequest(BaseModel):
    phone: str


@app.post("/api/auth/phone")
async def auth_phone(request: PhoneAuthRequest):
    """Запуск авторизации по номеру телефона"""
    set_status_callback(broadcast_message)
    result = await start_phone_auth(request.phone)
    return JSONResponse(result)


class CodeSubmitRequest(BaseModel):
    code: str


@app.post("/api/auth/code")
async def auth_code(request: CodeSubmitRequest):
    """Отправка кода подтверждения"""
    result = await submit_code(request.code)
    return JSONResponse(result)


class PasswordSubmitRequest(BaseModel):
    password: str


@app.post("/api/auth/password")
async def auth_password(request: PasswordSubmitRequest):
    """Отправка пароля 2FA"""
    result = await submit_password(request.password)
    return JSONResponse(result)


@app.post("/api/auth/logout")
async def auth_logout():
    """Выход из аккаунта"""
    result = await logout()
    return JSONResponse(result)


# ============== WEBSOCKET ==============

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    clients.append(ws)
    print(f"WebSocket connected. Total: {len(clients)}")
    
    # Отправляем текущее состояние
    try:
        await ws.send_json({"type": "stats", "stats": stats})
        await ws.send_json({"type": "monitoring", "active": monitoring_active})
    except:
        pass
    
    try:
        while True:
            data = await ws.receive_text()
    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"WS error: {e}")
    finally:
        if ws in clients:
            clients.remove(ws)
        print(f"WebSocket disconnected. Total: {len(clients)}")


# ============== BROADCAST ==============

async def broadcast_vacancy(vacancy: dict):
    """Рассылка новой вакансии"""
    save_vacancy(vacancy)
    stats["suitable"] = stats.get("suitable", 0) + 1
    
    await broadcast_message({"type": "vacancy", "vacancy": vacancy})
    await broadcast_message({"type": "stats", "stats": stats})


async def broadcast_status(message: str, icon: str = ""):
    await broadcast_message({"type": "status", "message": message, "icon": icon})


async def broadcast_stats():
    await broadcast_message({"type": "stats", "stats": stats})


async def broadcast_progress(percent: int, remaining: int = None):
    msg = {"type": "progress", "percent": percent}
    if remaining is not None:
        msg["remaining"] = remaining
    await broadcast_message(msg)


async def broadcast_message(message: dict):
    to_remove = []
    for client in clients:
        try:
            await client.send_json(message)
        except Exception:
            to_remove.append(client)
    for c in to_remove:
        if c in clients:
            clients.remove(c)


def update_stats(found: int = None, processed: int = None, 
                 rejected: int = None, suitable: int = None):
    global stats
    if found is not None:
        stats["found"] = found
    if processed is not None:
        stats["processed"] = processed
    if rejected is not None:
        stats["rejected"] = rejected
    if suitable is not None:
        stats["suitable"] = suitable
    
    asyncio.create_task(broadcast_stats())


def get_current_settings():
    return current_settings


# ============== SYSTEM MONITORING ==============

def get_gpu_info():
    """Получает информацию о GPU (NVIDIA или AMD)"""
    
    if GPU_TYPE is None:
        return None
    
    try:
        # ===== NVIDIA =====
        if GPU_TYPE == 'nvidia':
            import pynvml
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            
            utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
            gpu_util = utilization.gpu
            mem_util = utilization.memory
            
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            mem_used_mb = mem_info.used / (1024 * 1024)
            mem_total_mb = mem_info.total / (1024 * 1024)
            
            try:
                temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            except:
                temp = 0
            
            short_name = GPU_NAME
            match = re.search(r'(RTX|GTX|Quadro)\s*(\d{3,4})\s*(Ti|SUPER)?', GPU_NAME, re.I)
            if match:
                short_name = match.group(1).upper() + ' ' + match.group(2)
                if match.group(3):
                    short_name += ' ' + match.group(3)
            
            return {
                "available": True,
                "type": "nvidia",
                "name": GPU_NAME,
                "short_name": short_name,
                "utilization": gpu_util,
                "memory_utilization": mem_util,
                "memory_used_mb": round(mem_used_mb),
                "memory_total_mb": round(mem_total_mb),
                "temperature": temp
            }
        
        # ===== AMD via pyamdgpuinfo =====
        elif GPU_TYPE == 'amd':
            import pyamdgpuinfo
            gpu = pyamdgpuinfo.get_gpu(0)
            
            # pyamdgpuinfo API
            gpu_util = gpu.query_load() * 100 if hasattr(gpu, 'query_load') else 0
            vram_used = gpu.query_vram_usage() if hasattr(gpu, 'query_vram_usage') else 0
            vram_total = gpu.memory_info.get('vram_size', 0) if hasattr(gpu, 'memory_info') else 0
            temp = gpu.query_temperature() if hasattr(gpu, 'query_temperature') else 0
            
            mem_used_mb = vram_used / (1024 * 1024) if vram_used else 0
            mem_total_mb = vram_total / (1024 * 1024) if vram_total else 0
            mem_util = (mem_used_mb / mem_total_mb * 100) if mem_total_mb > 0 else 0
            
            short_name = GPU_NAME
            match = re.search(r'(RX|Radeon)\s*(\d{3,4})\s*(XT|XTX)?', GPU_NAME, re.I)
            if match:
                short_name = 'RX ' + match.group(2)
                if match.group(3):
                    short_name += ' ' + match.group(3)
            
            return {
                "available": True,
                "type": "amd",
                "name": GPU_NAME,
                "short_name": short_name,
                "utilization": round(gpu_util),
                "memory_utilization": round(mem_util),
                "memory_used_mb": round(mem_used_mb),
                "memory_total_mb": round(mem_total_mb),
                "temperature": round(temp) if temp else 0
            }
        
        # ===== AMD via rocm-smi =====
        elif GPU_TYPE == 'amd_rocm':
            gpu_util = 0
            mem_used_mb = 0
            mem_total_mb = 0
            temp = 0
            
            # GPU utilization
            try:
                result = subprocess.run(['rocm-smi', '--showuse'], 
                                        capture_output=True, text=True, timeout=2)
                for line in result.stdout.split('\n'):
                    if 'GPU use' in line or '%' in line:
                        match = re.search(r'(\d+)\s*%', line)
                        if match:
                            gpu_util = int(match.group(1))
                            break
            except:
                pass
            
            # Memory
            try:
                result = subprocess.run(['rocm-smi', '--showmeminfo', 'vram'], 
                                        capture_output=True, text=True, timeout=2)
                for line in result.stdout.split('\n'):
                    if 'Used' in line:
                        match = re.search(r'(\d+)', line)
                        if match:
                            mem_used_mb = int(match.group(1)) / (1024 * 1024)
                    elif 'Total' in line:
                        match = re.search(r'(\d+)', line)
                        if match:
                            mem_total_mb = int(match.group(1)) / (1024 * 1024)
            except:
                pass
            
            # Temperature
            try:
                result = subprocess.run(['rocm-smi', '--showtemp'], 
                                        capture_output=True, text=True, timeout=2)
                for line in result.stdout.split('\n'):
                    if 'Temperature' in line or 'edge' in line.lower():
                        match = re.search(r'(\d+\.?\d*)', line)
                        if match:
                            temp = float(match.group(1))
                            break
            except:
                pass
            
            mem_util = (mem_used_mb / mem_total_mb * 100) if mem_total_mb > 0 else 0
            
            short_name = GPU_NAME
            match = re.search(r'(RX|Radeon)\s*(\d{3,4})\s*(XT|XTX)?', GPU_NAME or '', re.I)
            if match:
                short_name = 'RX ' + match.group(2)
                if match.group(3):
                    short_name += ' ' + match.group(3)
            
            return {
                "available": True,
                "type": "amd",
                "name": GPU_NAME,
                "short_name": short_name or 'AMD GPU',
                "utilization": gpu_util,
                "memory_utilization": round(mem_util),
                "memory_used_mb": round(mem_used_mb),
                "memory_total_mb": round(mem_total_mb),
                "temperature": round(temp)
            }
        
        # ===== AMD via sysfs (basic) =====
        elif GPU_TYPE == 'amd_sysfs':
            gpu_util = 0
            mem_used_mb = 0
            mem_total_mb = 0
            temp = 0
            
            # Ищем hwmon для температуры и загрузки
            try:
                hwmon_base = '/sys/class/drm/card0/device/hwmon'
                if os.path.exists(hwmon_base):
                    hwmon_dir = os.path.join(hwmon_base, os.listdir(hwmon_base)[0])
                    
                    # Температура
                    temp_file = os.path.join(hwmon_dir, 'temp1_input')
                    if os.path.exists(temp_file):
                        with open(temp_file) as f:
                            temp = int(f.read().strip()) / 1000  # millidegrees to degrees
            except:
                pass
            
            # GPU busy percent
            try:
                busy_file = '/sys/class/drm/card0/device/gpu_busy_percent'
                if os.path.exists(busy_file):
                    with open(busy_file) as f:
                        gpu_util = int(f.read().strip())
            except:
                pass
            
            # VRAM
            try:
                vram_used_file = '/sys/class/drm/card0/device/mem_info_vram_used'
                vram_total_file = '/sys/class/drm/card0/device/mem_info_vram_total'
                if os.path.exists(vram_used_file):
                    with open(vram_used_file) as f:
                        mem_used_mb = int(f.read().strip()) / (1024 * 1024)
                if os.path.exists(vram_total_file):
                    with open(vram_total_file) as f:
                        mem_total_mb = int(f.read().strip()) / (1024 * 1024)
            except:
                pass
            
            mem_util = (mem_used_mb / mem_total_mb * 100) if mem_total_mb > 0 else 0
            
            short_name = GPU_NAME
            match = re.search(r'(RX|Radeon)\s*(\d{3,4})\s*(XT|XTX)?', GPU_NAME or '', re.I)
            if match:
                short_name = 'RX ' + match.group(2)
                if match.group(3):
                    short_name += ' ' + match.group(3)
            
            return {
                "available": True,
                "type": "amd",
                "name": GPU_NAME,
                "short_name": short_name or 'AMD GPU',
                "utilization": gpu_util,
                "memory_utilization": round(mem_util),
                "memory_used_mb": round(mem_used_mb),
                "memory_total_mb": round(mem_total_mb),
                "temperature": round(temp)
            }
    
    except Exception as e:
        print(f"GPU info error: {e}")
        return None
    
    return None


def get_cpu_info():
    """Получает информацию о CPU через psutil"""
    try:
        # CPU утилизация (не блокирующий вызов с interval=None использует кэш)
        cpu_percent = psutil.cpu_percent(interval=None)
        
        # Память
        mem = psutil.virtual_memory()
        
        # Количество ядер
        cpu_count = psutil.cpu_count(logical=True)
        
        # Частота (если доступна)
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
            "memory_total_gb": round(mem.total / (1024**3), 1)
        }
    except Exception as e:
        print(f"CPU info error: {e}")
        return {
            "utilization": 0,
            "cores": 0,
            "frequency_mhz": 0,
            "memory_percent": 0,
            "memory_used_gb": 0,
            "memory_total_gb": 0
        }


@app.get("/api/system-monitor")
async def get_system_monitor():
    """Endpoint для получения данных мониторинга системы"""
    gpu_info = get_gpu_info()
    cpu_info = get_cpu_info()
    
    return {
        "gpu": gpu_info,
        "cpu": cpu_info,
        "has_gpu": gpu_info is not None and gpu_info.get("available", False)
    }


# Фоновая задача для стриминга мониторинга
monitor_active = False

async def monitor_loop():
    """Фоновый цикл отправки данных мониторинга через WebSocket"""
    global monitor_active
    while monitor_active:
        gpu_info = get_gpu_info()
        cpu_info = get_cpu_info()
        
        await broadcast_message({
            "type": "system_monitor",
            "gpu": gpu_info,
            "cpu": cpu_info,
            "has_gpu": gpu_info is not None
        })
        
        await asyncio.sleep(1)  # Обновление раз в секунду


@app.post("/api/monitor/start")
async def start_monitor():
    """Запуск фонового мониторинга"""
    global monitor_active
    if not monitor_active:
        monitor_active = True
        asyncio.create_task(monitor_loop())
    return {"status": "started"}


@app.post("/api/monitor/stop")
async def stop_monitor():
    """Остановка фонового мониторинга"""
    global monitor_active
    monitor_active = False
    return {"status": "stopped"}
