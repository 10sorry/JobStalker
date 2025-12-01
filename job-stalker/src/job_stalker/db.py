import aiosqlite
import os
import logging

log = logging.getLogger("db")

DB_PATH = "./data/forwarded.db"

async def init_db():
    """Инициализация базы данных"""
    os.makedirs("./data", exist_ok=True)
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
    """Проверка, было ли сообщение уже переслано"""
    try:
        async with aiosqlite.connect(DB_PATH) as db:
            async with db.execute(
                "SELECT 1 FROM forwarded WHERE chat_id = ? AND message_id = ?",
                (chat_id, message_id)
            ) as cursor:
                row = await cursor.fetchone()
                log.debug(f"Checked forwarded: {chat_id}:{message_id} -> {row is not None}")
                return row is not None
    except Exception as e:
        log.error(f"Error in is_forwarded: {e}")
        return False  # Fallback: считать не обработанным, чтобы не пропустить

async def mark_forwarded(chat_id: int, message_id: int):
    """Отметить сообщение как пересланное"""
    try:
        async with aiosqlite.connect(DB_PATH) as db:
            await db.execute(
                "INSERT OR IGNORE INTO forwarded (chat_id, message_id) VALUES (?, ?)",
                (chat_id, message_id)
            )
            await db.commit()
            log.debug(f"Marked forwarded: {chat_id}:{message_id}")
    except Exception as e:
        log.error(f"Error in mark_forwarded: {e}")

async def reset_db():
    """Полный сброс базы данных"""
    try:
        async with aiosqlite.connect(DB_PATH) as db:
            await db.execute("DELETE FROM forwarded")
            await db.commit()
        log.info("DB reset successful")
    except Exception as e:
        log.error(f"Error in reset_db: {e}")

async def get_forwarded_count() -> int:
    """Получить количество пересланных сообщений"""
    try:
        async with aiosqlite.connect(DB_PATH) as db:
            async with db.execute("SELECT COUNT(*) FROM forwarded") as cursor:
                row = await cursor.fetchone()
                return row[0] if row else 0
    except Exception as e:
        log.error(f"Error in get_forwarded_count: {e}")
        return 0
