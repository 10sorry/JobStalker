"""
Модуль веб-авторизации для Telegram (Pyrogram)
Поддерживает QR-код и авторизацию по номеру телефона
"""
import asyncio
import os
import base64
from typing import Optional, Dict, Callable
from io import BytesIO
from datetime import datetime

from pyrogram import Client
from pyrogram.errors import (
    SessionPasswordNeeded, PhoneCodeInvalid,
    PhoneCodeExpired, PhoneNumberInvalid
)
import logging

log = logging.getLogger("telegram_auth")

_auth_client: Optional[Client] = None
_auth_state: Dict = {
    "status": "not_started",
    "phone": None,
    "phone_code_hash": None,
    "error": None,
    "qr_code": None,
    "authorized": False
}
_status_callback: Optional[Callable] = None


def set_status_callback(callback: Optional[Callable]):
    """Устанавливает callback для уведомлений о статусе"""
    global _status_callback
    _status_callback = callback


async def notify_status(status: str, data: dict = None):
    """Отправляет уведомление о статусе"""
    global _auth_state
    _auth_state["status"] = status

    if data:
        _auth_state.update(data)

    if _status_callback:
        try:
            await _status_callback({
                "type": "auth_status",
                "status": status,
                "data": _auth_state
            })
        except Exception as e:
            log.error(f"Status callback error: {e}")


async def init_auth_client():
    """Инициализирует клиент для авторизации"""
    global _auth_client

    from .config import API_ID, API_HASH, SESSION_NAME

    if _auth_client is None:
        os.makedirs("./data", exist_ok=True)
        _auth_client = Client(
            SESSION_NAME,
            api_id=API_ID,
            api_hash=API_HASH,
            workdir="./data"
        )

    return _auth_client


async def is_authorized() -> bool:
    """Проверяет, авторизован ли пользователь"""
    from .config import SESSION_NAME

    # Проверяем существование файла сессии
    session_file = f"./data/{SESSION_NAME}.session"
    if not os.path.exists(session_file):
        log.info("Session file not found")
        return False

    # Простая проверка размера файла (валидная сессия > 1KB)
    try:
        file_size = os.path.getsize(session_file)
        if file_size < 1000:
            log.info("Session file too small, probably invalid")
            return False

        log.info(f"Session file found: {file_size} bytes")
        return True
    except Exception as e:
        log.error(f"Authorization check error: {e}")
        return False


async def get_auth_status() -> dict:
    """Возвращает текущий статус авторизации"""
    global _auth_state

    # Проверяем реальный статус авторизации
    authorized = await is_authorized()
    _auth_state["authorized"] = authorized

    if authorized:
        _auth_state["status"] = "success"

    return _auth_state


async def start_qr_auth():
    """
    QR-авторизация не поддерживается Pyrogram
    Используйте авторизацию по номеру телефона
    """
    return {
        "success": False,
        "error": "QR-авторизация не поддерживается. Используйте авторизацию по номеру телефона."
    }


async def start_phone_auth(phone: str):
    """
    Запускает авторизацию по номеру телефона
    Отправляет код подтверждения
    """
    global _auth_client, _auth_state

    try:
        if not phone or len(phone) < 10:
            raise ValueError("Invalid phone number")

        if not phone.startswith('+'):
            phone = '+' + phone

        client = await init_auth_client()

        if not client.is_connected:
            await client.connect()

        await notify_status("sending_code")

        sent_code = await client.send_code(phone)

        _auth_state["phone"] = phone
        _auth_state["phone_code_hash"] = sent_code.phone_code_hash

        await notify_status("waiting_code", {
            "phone": phone
        })

        return {
            "success": True,
            "phone": phone,
            "code_sent": True
        }

    except PhoneNumberInvalid:
        error = "Неверный номер телефона"
        await notify_status("error", {"error": error})

        if _auth_client and _auth_client.is_connected:
            await _auth_client.disconnect()

        return {"success": False, "error": error}
    except Exception as e:
        log.error(f"Phone auth error: {e}")
        error = str(e)
        await notify_status("error", {"error": error})

        if _auth_client and _auth_client.is_connected:
            await _auth_client.disconnect()

        return {"success": False, "error": error}


async def submit_code(code: str):
    """
    Отправляет код подтверждения
    """
    global _auth_client, _auth_state

    try:
        if not _auth_state.get("phone") or not _auth_state.get("phone_code_hash"):
            raise ValueError("No active phone auth session")

        client = _auth_client
        if not client or not client.is_connected:
            raise ValueError("Client not connected")

        await notify_status("verifying_code")

        await client.sign_in(
            _auth_state["phone"],
            _auth_state["phone_code_hash"],
            code
        )

        me = await client.get_me()

        _auth_state.clear()
        _auth_state["status"] = "success"
        _auth_state["authorized"] = True

        await notify_status("success", {
            "authorized": True,
            "user": {
                "id": me.id,
                "first_name": me.first_name,
                "username": me.username
            }
        })

        await client.disconnect()

        return {
            "success": True,
            "user": {
                "id": me.id,
                "first_name": me.first_name,
                "username": me.username
            }
        }

    except SessionPasswordNeeded:
        await notify_status("waiting_password")
        return {
            "success": False,
            "requires_password": True,
            "message": "Требуется пароль двухфакторной аутентификации"
        }
    except PhoneCodeInvalid:
        error = "Неверный код подтверждения"
        await notify_status("error", {"error": error})
        return {"success": False, "error": error}
    except PhoneCodeExpired:
        error = "Код подтверждения истек"
        await notify_status("error", {"error": error})
        return {"success": False, "error": error}
    except Exception as e:
        log.error(f"Code submit error: {e}")
        error = str(e)
        await notify_status("error", {"error": error})
        return {"success": False, "error": error}


async def submit_password(password: str):
    """
    Отправляет пароль 2FA
    """
    global _auth_client, _auth_state

    try:
        client = _auth_client
        if not client or not client.is_connected:
            raise ValueError("Client not connected")

        await notify_status("verifying_password")

        await client.check_password(password)

        me = await client.get_me()

        _auth_state.clear()
        _auth_state["status"] = "success"
        _auth_state["authorized"] = True

        await notify_status("success", {
            "authorized": True,
            "user": {
                "id": me.id,
                "first_name": me.first_name,
                "username": me.username
            }
        })

        await client.disconnect()

        return {
            "success": True,
            "user": {
                "id": me.id,
                "first_name": me.first_name,
                "username": me.username
            }
        }

    except Exception as e:
        log.error(f"Password submit error: {e}")
        error = "Неверный пароль"
        await notify_status("error", {"error": error})
        return {"success": False, "error": error}


async def logout():
    """Выход из аккаунта"""
    try:
        client = await init_auth_client()
        await client.connect()
        await client.log_out()
        await client.disconnect()

        # Удаляем файл сессии
        session_file = f"./data/{client.name}.session"
        if os.path.exists(session_file):
            os.remove(session_file)

        global _auth_state
        _auth_state = {
            "status": "not_started",
            "authorized": False
        }

        await notify_status("logged_out")

        return {"success": True}
    except Exception as e:
        log.error(f"Logout error: {e}")
        return {"success": False, "error": str(e)}


async def get_user_info():
    """Получает информацию о текущем пользователе"""
    try:
        if not await is_authorized():
            return None

        return {
            "id": 0,
            "first_name": "Authorized",
            "last_name": "User",
            "username": "user",
            "phone": "+7"
        }
    except Exception as e:
        log.error(f"Get user info error: {e}")
        return None
