from __future__ import annotations

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncGenerator, List, Optional, Tuple

import aiosqlite

from .exceptions import DatabaseError

log = logging.getLogger("database")

DATA_DIR = Path("data")
DB_PATH = DATA_DIR / "forwarded.db"


class DatabasePool:
    def __init__(self, db_path: Path = DB_PATH, pool_size: int = 5, timeout: float = 30.0):
        self.db_path = Path(db_path)
        self.pool_size = pool_size
        self.timeout = timeout
        self._pool: asyncio.Queue[aiosqlite.Connection] = asyncio.Queue(maxsize=pool_size)
        self._connections: List[aiosqlite.Connection] = []
        self._lock = asyncio.Lock()
        self._initialized = False

    async def initialize(self) -> None:
        if self._initialized:
            return

        async with self._lock:
            if self._initialized:
                return

            self.db_path.parent.mkdir(parents=True, exist_ok=True)

            for _ in range(self.pool_size):
                conn = await aiosqlite.connect(self.db_path, timeout=self.timeout)
                await conn.execute("PRAGMA journal_mode=WAL")
                await conn.execute("PRAGMA synchronous=NORMAL")
                await conn.execute("PRAGMA cache_size=10000")
                self._connections.append(conn)
                await self._pool.put(conn)

            async with self.acquire() as conn:
                await conn.execute("""
                    CREATE TABLE IF NOT EXISTS forwarded (
                        chat_id INTEGER,
                        message_id INTEGER,
                        forwarded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        PRIMARY KEY (chat_id, message_id)
                    )
                """)
                await conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_forwarded_chat
                    ON forwarded(chat_id)
                """)
                await conn.commit()

            self._initialized = True
            log.info(f"Database pool initialized with {self.pool_size} connections")

    @asynccontextmanager
    async def acquire(self) -> AsyncGenerator[aiosqlite.Connection, None]:
        if not self._initialized:
            await self.initialize()

        conn = await asyncio.wait_for(self._pool.get(), timeout=self.timeout)
        try:
            yield conn
        finally:
            await self._pool.put(conn)

    async def close(self) -> None:
        async with self._lock:
            for conn in self._connections:
                try:
                    await conn.close()
                except Exception as e:
                    log.warning(f"Error closing connection: {e}")

            self._connections.clear()
            self._initialized = False
            log.info("Database pool closed")

    @property
    def stats(self) -> dict:
        return {
            "pool_size": self.pool_size,
            "available": self._pool.qsize(),
            "in_use": self.pool_size - self._pool.qsize(),
            "initialized": self._initialized,
        }


_pool: Optional[DatabasePool] = None


def get_pool() -> DatabasePool:
    global _pool
    if _pool is None:
        _pool = DatabasePool()
    return _pool


async def init_db() -> None:
    pool = get_pool()
    await pool.initialize()


async def close_db() -> None:
    global _pool
    if _pool is not None:
        await _pool.close()
        _pool = None


async def is_forwarded(chat_id: int, message_id: int) -> bool:
    pool = get_pool()
    try:
        async with pool.acquire() as conn:
            async with conn.execute(
                "SELECT 1 FROM forwarded WHERE chat_id = ? AND message_id = ?",
                (chat_id, message_id),
            ) as cursor:
                row = await cursor.fetchone()
                return row is not None
    except Exception as e:
        raise DatabaseError("is_forwarded", cause=e)


async def mark_forwarded(chat_id: int, message_id: int) -> None:
    pool = get_pool()
    try:
        async with pool.acquire() as conn:
            await conn.execute(
                "INSERT OR IGNORE INTO forwarded (chat_id, message_id) VALUES (?, ?)",
                (chat_id, message_id),
            )
            await conn.commit()
    except Exception as e:
        raise DatabaseError("mark_forwarded", cause=e)


async def mark_forwarded_batch(messages: List[Tuple[int, int]]) -> None:
    if not messages:
        return

    pool = get_pool()
    try:
        async with pool.acquire() as conn:
            await conn.executemany(
                "INSERT OR IGNORE INTO forwarded (chat_id, message_id) VALUES (?, ?)", messages
            )
            await conn.commit()
            log.debug(f"Batch marked {len(messages)} messages as forwarded")
    except Exception as e:
        raise DatabaseError("mark_forwarded_batch", cause=e)


async def reset_db() -> None:
    pool = get_pool()
    try:
        async with pool.acquire() as conn:
            await conn.execute("DELETE FROM forwarded")
            await conn.commit()
            log.info("Database reset successful")
    except Exception as e:
        raise DatabaseError("reset_db", cause=e)


async def get_forwarded_count() -> int:
    pool = get_pool()
    try:
        async with pool.acquire() as conn:
            async with conn.execute("SELECT COUNT(*) FROM forwarded") as cursor:
                row = await cursor.fetchone()
                return row[0] if row else 0
    except Exception as e:
        raise DatabaseError("get_forwarded_count", cause=e)


async def get_forwarded_for_channel(chat_id: int) -> List[int]:
    pool = get_pool()
    try:
        async with pool.acquire() as conn:
            async with conn.execute(
                "SELECT message_id FROM forwarded WHERE chat_id = ?", (chat_id,)
            ) as cursor:
                rows = await cursor.fetchall()
                return [row[0] for row in rows]
    except Exception as e:
        raise DatabaseError("get_forwarded_for_channel", cause=e)


async def cleanup_old_records(days: int = 30) -> int:
    pool = get_pool()
    try:
        async with pool.acquire() as conn:
            result = await conn.execute(
                "DELETE FROM forwarded WHERE forwarded_at < datetime('now', ?)", (f"-{days} days",)
            )
            await conn.commit()
            deleted = result.rowcount
            if deleted > 0:
                log.info(f"Cleaned up {deleted} old records")
            return deleted
    except Exception as e:
        raise DatabaseError("cleanup_old_records", cause=e)
