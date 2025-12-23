import asyncio
import sys
import logging
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from pyrogram import Client, filters
from pyrogram.handlers import MessageHandler

from .config import API_ID, API_HASH, SESSION_NAME

DATA_DIR = Path("data")
from .db import init_db, is_forwarded, mark_forwarded
from .ml_filter import ml_interesting_async, recruiter_analysis, RESUME_DATA
from .vacancy_storage import update_vacancy
from .state import get_state

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("main")

from .web_ui import (
    broadcast_vacancy,
    broadcast_status,
    broadcast_progress,
    update_stats,
    get_current_settings,
    broadcast_message,
)

CONCURRENT_ANALYSIS = 3
analysis_semaphore = asyncio.Semaphore(CONCURRENT_ANALYSIS)


def is_message_recent(message_date, days_back: int) -> bool:
    if not message_date:
        return True
    cutoff = datetime.now() - timedelta(days=days_back)
    return message_date >= cutoff


class Stats:
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
    try:
        if not RESUME_DATA or "raw_text" not in RESUME_DATA:
            log.info(f"Stage 2 skipped for {vacancy_id[:8]}: no resume loaded")
            return

        log.info(f"Stage 2: Starting async recruiter analysis for {vacancy_id[:8]}...")

        ra = await recruiter_analysis(vacancy_text, RESUME_DATA["raw_text"])

        if ra and ra.match_score > 0:
            recruiter_data = {
                "match_score": ra.match_score,
                "strong_sides": ra.strong_sides,
                "weak_sides": ra.weak_sides,
                "missing_skills": ra.missing_skills,
                "risks": ra.risks,
                "recommendations": ra.recommendations,
                "verdict": ra.verdict,
                "cover_letter_hint": ra.cover_letter_hint,
            }

            update_vacancy(
                vacancy_id,
                {
                    "recruiter_analysis": recruiter_data,
                    "comparison": {"match_score": ra.match_score},
                },
            )

            update_msg = {
                "type": "vacancy_update",
                "vacancy_id": vacancy_id,
                "recruiter_analysis": recruiter_data,
            }
            await broadcast_message(update_msg)
            log.info(f"Stage 2 done for {vacancy_id[:8]}: match_score={ra.match_score}")
        else:
            log.warning(f"Stage 2 returned empty result for {vacancy_id[:8]}")

    except asyncio.CancelledError:
        log.info(f"Stage 2 cancelled for {vacancy_id[:8]}")
        raise
    except Exception as e:
        log.error(f"Stage 2 error for {vacancy_id[:8]}: {e}")


def keyword_filter_check(text: str, keyword_filter: str) -> bool:
    if not keyword_filter.strip():
        return True

    keywords = [kw.strip().lower() for kw in keyword_filter.split(",") if kw.strip()]
    text_lower = text.lower()

    for keyword in keywords:
        if keyword in text_lower:
            return True

    return False


async def process_message(message, channel_title: str) -> bool:
    async with analysis_semaphore:
        chat_id = message.chat.id
        msg_id = message.id
        text = message.text or message.caption or ""

        if not text or len(text.strip()) < 30:
            return False

        stats.found += 1
        update_stats(found=stats.found)

        try:
            settings = get_current_settings()
            search_mode = settings.get("search_mode", "basic")
            keyword_filter = settings.get("keyword_filter", "")

            if search_mode == "basic":
                if not keyword_filter_check(text, keyword_filter):
                    stats.rejected += 1
                    update_stats(rejected=stats.rejected)
                    log.info(f"Keyword filtered: {chat_id}:{msg_id}")
                    return False

                result_suitable = True
                analysis_text = (
                    f"Matched keyword filter: {keyword_filter}"
                    if keyword_filter
                    else "Basic mode: accepted"
                )

            else:
                result = await ml_interesting_async(text)
                result_suitable = result.suitable
                analysis_text = result.analysis

            stats.processed += 1
            update_stats(processed=stats.processed)

            if not result_suitable:
                stats.rejected += 1
                update_stats(rejected=stats.rejected)
                log.info(f"Rejected: {chat_id}:{msg_id}")
                return False

            stats.suitable += 1
            update_stats(suitable=stats.suitable)

            link = (
                f"https://t.me/{message.chat.username}/{message.id}"
                if message.chat.username
                else None
            )

            vacancy_id = str(uuid.uuid4())
            vacancy = {
                "id": vacancy_id,
                "channel": channel_title,
                "text": text,
                "date": str(message.date),
                "link": link,
                "analysis": analysis_text,
                "is_new": True,
            }

            log.info(f"Found: {channel_title}")

            await broadcast_vacancy(vacancy)

            await mark_forwarded(chat_id, msg_id)

            return True

        except asyncio.CancelledError:
            raise
        except Exception as e:
            log.error(f"Error: {e}")
            return False


async def start_bot():
    from .telegram_auth import is_authorized
    from .config import validate_config

    state = get_state()

    try:
        validate_config()
    except RuntimeError as e:
        log.error(f"Configuration error: {e}")
        await broadcast_status(f"Error: {e}", "Warning")
        return

    if not await is_authorized():
        log.warning("Not authorized! Open web interface for authorization")
        await broadcast_status("Telegram authorization required", "Warning")
        return

    await init_db()
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    settings = get_current_settings()
    days_back = settings.get("days_back", 7)
    channels = settings.get("channels", [])

    if not channels or len(channels) == 0:
        log.error("No channels configured!")
        await broadcast_status("Configure channels in settings", "Warning")
        return

    log.info(f"Searching {days_back} days back in {len(channels)} channels")
    await broadcast_status(f"Searching {days_back} days back", "Search")

    stats.reset()

    app = Client(SESSION_NAME, api_id=API_ID, api_hash=API_HASH, workdir=str(DATA_DIR))

    async with app:
        log.info("Bot started")
        await broadcast_status("Connecting...", "Loading")

        total_channels = len(channels)

        for idx, channel in enumerate(channels):
            if not state.monitoring_active:
                log.info("Stopped")
                break

            try:
                chat = await app.get_chat(channel)
                log.info(f"[{idx + 1}/{total_channels}] {chat.title}")
                await broadcast_status(f"{chat.title}", "Channel")

                progress = int((idx / total_channels) * 100)
                await broadcast_progress(progress, total_channels - idx)

                messages = []
                async for message in app.get_chat_history(chat.id, limit=600):
                    if not state.monitoring_active:
                        break
                    if not is_message_recent(message.date, days_back):
                        continue
                    if await is_forwarded(message.chat.id, message.id):
                        continue
                    messages.append((message, chat.title))

                if messages:
                    await broadcast_status(
                        f"Analyzing {len(messages)} messages...", "Robot"
                    )

                    tasks = [process_message(m, t) for m, t in messages]

                    for i in range(0, len(tasks), 5):
                        if not state.monitoring_active:
                            break
                        batch = tasks[i : i + 5]
                        await asyncio.gather(*batch, return_exceptions=True)
                        await asyncio.sleep(0.1)

            except asyncio.CancelledError:
                log.info("Cancelled")
                raise
            except Exception as e:
                log.error(f"Channel error {channel}: {e}")
                continue

        await broadcast_progress(100, 0)
        await broadcast_status(f"Found {stats.suitable} vacancies", "Done")

        if state.monitoring_active:
            log.info("Monitoring...")
            await broadcast_status("Monitoring for new...", "Eyes")

            @app.on_message(filters.channel)
            async def on_new_message(client, message):
                if not state.monitoring_active:
                    return

                chat_id = str(message.chat.id)
                chat_username = message.chat.username

                settings = get_current_settings()
                current_channels = settings.get("channels", [])

                is_our_channel = False
                for ch in current_channels:
                    if str(ch) == chat_id or ch == chat_username:
                        is_our_channel = True
                        break

                if is_our_channel:
                    await process_message(message, message.chat.title)

            while state.monitoring_active:
                await asyncio.sleep(1)

        log.info("Bot stopped")


async def main():
    await start_bot()


if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
