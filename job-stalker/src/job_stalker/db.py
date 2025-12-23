import aiosqlite
import logging
from pathlib import Path

log = logging.getLogger("db")

DATA_DIR = Path("data")
DB_PATH = DATA_DIR / "forwarded.db"


async def init_db():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("""
            CREATE TABLE IF NOT EXISTS forwarded (
                chat_id INTEGER,
                message_id INTEGER,
                forwarded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (chat_id, message_id)
            )
        """)
        await db.commit()
    log.info("DB initialized")


async def is_forwarded(chat_id: int, message_id: int) -> bool:
    try:
        async with aiosqlite.connect(DB_PATH) as db:
            async with db.execute(
                "SELECT 1 FROM forwarded WHERE chat_id = ? AND message_id = ?",
                (chat_id, message_id),
            ) as cursor:
                row = await cursor.fetchone()
                log.debug(f"Checked forwarded: {chat_id}:{message_id} -> {row is not None}")
                return row is not None
    except Exception as e:
        log.error(f"Error in is_forwarded: {e}")
        return False


async def mark_forwarded(chat_id: int, message_id: int):
    try:
        async with aiosqlite.connect(DB_PATH) as db:
            await db.execute(
                "INSERT OR IGNORE INTO forwarded (chat_id, message_id) VALUES (?, ?)",
                (chat_id, message_id),
            )
            await db.commit()
            log.debug(f"Marked forwarded: {chat_id}:{message_id}")
    except Exception as e:
        log.error(f"Error in mark_forwarded: {e}")


async def reset_db():
    try:
        async with aiosqlite.connect(DB_PATH) as db:
            await db.execute("DELETE FROM forwarded")
            await db.commit()
        log.info("DB reset successful")
    except Exception as e:
        log.error(f"Error in reset_db: {e}")


async def get_forwarded_count() -> int:
    try:
        async with aiosqlite.connect(DB_PATH) as db:
            async with db.execute("SELECT COUNT(*) FROM forwarded") as cursor:
                row = await cursor.fetchone()
                return row[0] if row else 0
    except Exception as e:
        log.error(f"Error in get_forwarded_count: {e}")
        return 0
