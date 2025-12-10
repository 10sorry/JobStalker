"""
Vacancy Monitor Bot - Main Module
"""
import asyncio
import os
import logging
import uuid
from datetime import datetime, timedelta
from pyrogram import Client, filters
from pyrogram.handlers import MessageHandler

from .config import API_ID, API_HASH, SESSION_NAME
from .db import init_db, is_forwarded, mark_forwarded
from .ml_filter import ml_interesting_async, recruiter_analysis, RESUME_DATA
from .vacancy_storage import update_vacancy

# Логирование
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
log = logging.getLogger("main")

# Импорты web_ui
from .web_ui import (broadcast_vacancy, broadcast_status, broadcast_progress,
                    update_stats, get_current_settings, broadcast_message)

# Семафор для параллельного анализа
CONCURRENT_ANALYSIS = 3
analysis_semaphore = asyncio.Semaphore(CONCURRENT_ANALYSIS)


def is_message_recent(message_date, days_back: int) -> bool:
    """Проверка актуальности"""
    if not message_date:
        return True
    cutoff = datetime.now() - timedelta(days=days_back)
    return message_date >= cutoff


class Stats:
    """Счётчики статистики"""
    def __init__(self):
        self.processed = 0
        self.rejected = 0
        self.suitable = 0
        self.found = 0
    
    def reset(self):
        self.processed = 0
        self.rejected = 0
        self.suitable = 0
        self.found = 0


stats = Stats()


async def run_stage2_async(vacancy_id: str, vacancy_text: str):
    """Stage 2: Асинхронный анализ рекрутера (не блокирует поиск)"""
    try:
        if not RESUME_DATA or 'raw_text' not in RESUME_DATA:
            log.info(f"⏭️ Stage 2 skipped for {vacancy_id[:8]}: no resume loaded")
            return

        log.info(f"🎯 Stage 2: Starting async recruiter analysis for {vacancy_id[:8]}...")

        ra = await recruiter_analysis(vacancy_text, RESUME_DATA['raw_text'])

        if ra and ra.match_score > 0:
            recruiter_data = {
                "match_score": ra.match_score,
                "strong_sides": ra.strong_sides,
                "weak_sides": ra.weak_sides,
                "missing_skills": ra.missing_skills,
                "risks": ra.risks,
                "recommendations": ra.recommendations,
                "verdict": ra.verdict,
                "cover_letter_hint": ra.cover_letter_hint
            }

            # Сохраняем в файл
            update_vacancy(vacancy_id, {
                "recruiter_analysis": recruiter_data,
                "comparison": {"match_score": ra.match_score}
            })

            # Отправляем обновление в UI
            update_msg = {
                "type": "vacancy_update",
                "vacancy_id": vacancy_id,
                "recruiter_analysis": recruiter_data
            }
            await broadcast_message(update_msg)
            log.info(f"✅ Stage 2 done for {vacancy_id[:8]}: match_score={ra.match_score}")
        else:
            log.warning(f"⚠️ Stage 2 returned empty result for {vacancy_id[:8]}")

    except Exception as e:
        log.error(f"❌ Stage 2 error for {vacancy_id[:8]}: {e}")


async def process_message(message, channel_title: str) -> bool:
    """Обработка одного сообщения"""
    async with analysis_semaphore:
        chat_id = message.chat.id
        msg_id = message.id
        text = message.text or message.caption or ""

        if not text or len(text.strip()) < 30:
            return False

        stats.found += 1
        update_stats(found=stats.found)

        try:
            # Stage 1: Быстрая фильтрация
            result = await ml_interesting_async(text)

            stats.processed += 1
            update_stats(processed=stats.processed)

            if not result.suitable:
                stats.rejected += 1
                update_stats(rejected=stats.rejected)
                log.info(f"❌ Отклонено: {chat_id}:{msg_id}")
                return False

            # Подходит! Показываем карточку СРАЗУ
            stats.suitable += 1
            update_stats(suitable=stats.suitable)

            link = f"https://t.me/{message.chat.username}/{message.id}" if message.chat.username else None

            vacancy_id = str(uuid.uuid4())
            vacancy = {
                "id": vacancy_id,
                "channel": channel_title,
                "text": text,
                "date": str(message.date),
                "link": link,
                "analysis": result.analysis,
                "is_new": True
            }

            log.info(f"✅ Найдено: {channel_title}")

            # Отправляем карточку в UI СРАЗУ (без ожидания Stage 2)
            await broadcast_vacancy(vacancy)

            # Помечаем обработанным
            await mark_forwarded(chat_id, msg_id)

            # Stage 2: Запускаем АСИНХРОННО (не блокирует поиск следующих вакансий)
            asyncio.create_task(run_stage2_async(vacancy_id, text))

            return True

        except Exception as e:
            log.error(f"Ошибка: {e}")
            return False


async def start_bot():
    """Основная функция бота"""
    # Импортируем здесь чтобы избежать circular import
    from . import web_ui
    from .telegram_auth import is_authorized
    from .config import validate_config

    # Проверяем конфигурацию
    try:
        validate_config()
    except RuntimeError as e:
        log.error(f"❌ Ошибка конфигурации: {e}")
        await broadcast_status(f"❌ {e}", "⚠️")
        return

    # Проверяем авторизацию перед запуском
    if not await is_authorized():
        log.warning("❌ Не авторизован! Откройте веб-интерфейс для авторизации")
        await broadcast_status("❌ Требуется авторизация в Telegram", "⚠️")
        return

    await init_db()
    os.makedirs("./data", exist_ok=True)

    settings = get_current_settings()
    days_back = settings.get("days_back", 7)
    channels = settings.get("channels", [])

    # Проверяем что каналы заданы
    if not channels or len(channels) == 0:
        log.error("❌ Каналы не заданы в настройках!")
        await broadcast_status("❌ Укажите каналы в настройках", "⚠️")
        return

    log.info(f"🔍 Поиск за {days_back} дней в {len(channels)} каналах")
    await broadcast_status(f"🔍 Поиск за {days_back} дней", "🔍")

    stats.reset()

    app = Client(SESSION_NAME, api_id=API_ID, api_hash=API_HASH, workdir="./data")

    async with app:
        log.info("🚀 Бот запущен")
        await broadcast_status("🚀 Подключение...", "🔄")

        total_channels = len(channels)

        for idx, channel in enumerate(channels):
            # Проверяем флаг ВНУТРИ web_ui
            if not web_ui.monitoring_active:
                log.info("❌ Остановлено")
                break
            
            try:
                chat = await app.get_chat(channel)
                log.info(f"📡 [{idx+1}/{total_channels}] {chat.title}")
                await broadcast_status(f"📡 {chat.title}", "📡")
                
                progress = int((idx / total_channels) * 100)
                await broadcast_progress(progress, total_channels - idx)
                
                # Собираем сообщения
                messages = []
                async for message in app.get_chat_history(chat.id, limit=100):
                    if not web_ui.monitoring_active:
                        break
                    if not is_message_recent(message.date, days_back):
                        continue
                    if await is_forwarded(message.chat.id, message.id):
                        continue
                    messages.append((message, chat.title))
                
                # Параллельная обработка
                if messages:
                    await broadcast_status(f"🤖 Анализ {len(messages)} сообщений...", "🤖")
                    
                    tasks = [process_message(m, t) for m, t in messages]
                    
                    # Обрабатываем пакетами
                    for i in range(0, len(tasks), 5):
                        if not web_ui.monitoring_active:
                            break
                        batch = tasks[i:i+5]
                        await asyncio.gather(*batch, return_exceptions=True)
                        await asyncio.sleep(0.1)
                
            except Exception as e:
                log.error(f"Ошибка канала {channel}: {e}")
                continue
        
        await broadcast_progress(100, 0)
        await broadcast_status(f"✅ Найдено {stats.suitable} вакансий", "✅")
        
        # Real-time мониторинг
        if web_ui.monitoring_active:
            log.info("👀 Мониторинг...")
            await broadcast_status("👀 Мониторинг новых...", "👀")
            
            @app.on_message(filters.channel)
            async def on_new_message(client, message):
                if not web_ui.monitoring_active:
                    return
                
                chat_id = str(message.chat.id)
                chat_username = message.chat.username
                
                # Получаем актуальный список каналов из настроек
                settings = get_current_settings()
                current_channels = settings.get("channels", [])

                # Проверяем что это наш канал
                is_our_channel = False
                for ch in current_channels:
                    if str(ch) == chat_id or ch == chat_username:
                        is_our_channel = True
                        break
                
                if is_our_channel:
                    await process_message(message, message.chat.title)
            
            # Ждём пока active
            while web_ui.monitoring_active:
                await asyncio.sleep(1)
        
        log.info("🛑 Бот остановлен")


async def main():
    await start_bot()


if __name__ == "__main__":
    asyncio.run(main())
