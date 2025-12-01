import asyncio
import os
from pyrogram import Client
from .config import API_ID, API_HASH, SESSION_NAME

def safe_print(*args, **kwargs):
    """Безопасный вывод в консоль"""
    print(*args, **kwargs)

async def authorize():
    safe_print("=" * 50)
    safe_print("АВТОРИЗАЦИЯ В TELEGRAM")
    safe_print("=" * 50)
    safe_print("")
    safe_print("Этот скрипт нужно запустить ОДИН РАЗ для авторизации.")
    safe_print("После успешной авторизации файл сессии сохранится")
    safe_print("и больше не потребуется вводить номер телефона.")
    safe_print("")
    safe_print("-" * 50)

    os.makedirs("./data", exist_ok=True)

    safe_print("API_ID:", repr(API_ID))
    safe_print("API_HASH:", repr(API_HASH))
    safe_print("SESSION_NAME:", repr(SESSION_NAME))
    safe_print(API_HASH, type(API_HASH))

    app = Client(SESSION_NAME, api_id=3877649, api_hash="0a5ae248caf607553f98b6e9c2e387eb", workdir="./data")

    try:
        me = await app.get_me()

        safe_print("")
        safe_print("АВТОРИЗАЦИЯ УСПЕШНА!")
        safe_print(f"Пользователь: {me.first_name} {me.last_name or ''}")
        safe_print(f"Телефон: {me.phone_number}")
        safe_print(f"ID: {me.id}")
        safe_print("")
        safe_print(f"Файл сессии сохранен: ./data/{SESSION_NAME}.session")
        safe_print("")
        safe_print("Теперь можете запускать основной бот без повторной авторизации:")
        safe_print("  python run.py")
        safe_print("")

    except Exception as e:
        safe_print("")
        safe_print(f"ОШИБКА: {e}")
        safe_print("")
        safe_print("Проверьте:")
        safe_print("  1. API_ID и API_HASH в config.py правильные")
        safe_print("  2. Интернет-соединение активно")
        safe_print("  3. Номер телефона введен корректно (с +)")
        safe_print("")

if __name__ == "__main__":
    asyncio.run(authorize())
