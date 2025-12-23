import asyncio
import logging
from typing import Dict, Any, Optional, List, Set, Callable
from dataclasses import dataclass, field
from contextlib import asynccontextmanager
from weakref import WeakSet

log = logging.getLogger("state")


@dataclass
class Stats:
    found: int = 0
    processed: int = 0
    rejected: int = 0
    suitable: int = 0

    def reset(self):
        self.found = 0
        self.processed = 0
        self.rejected = 0
        self.suitable = 0

    def to_dict(self) -> Dict[str, int]:
        return {
            "found": self.found,
            "processed": self.processed,
            "rejected": self.rejected,
            "suitable": self.suitable,
        }


class AppState:
    _instance: Optional["AppState"] = None
    _lock: asyncio.Lock
    _initialized: bool = False

    def __new__(cls) -> "AppState":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        if self._initialized:
            return

        self._initialized = True
        self._lock = asyncio.Lock()

        self._monitoring_active: bool = False
        self._monitoring_task: Optional[asyncio.Task] = None
        self._monitor_loop_active: bool = False

        self._ws_clients: List[Any] = []
        self._ws_lock = asyncio.Lock()

        self._stats = Stats()

        self._settings: Dict[str, Any] = {
            "model_type": "mistral",
            "days_back": 7,
            "custom_prompt": "",
            "resume_summary": "",
            "channels": [],
            "enable_stage2": False,
            "keyword_filter": "",
            "search_mode": "basic",
        }

        self._improvement_tasks: Dict[str, Dict[str, Any]] = {}

        self._stage2_tasks: Set[asyncio.Task] = set()

        self._stats_callback: Optional[Callable] = None

    @property
    def monitoring_active(self) -> bool:
        return self._monitoring_active

    async def start_monitoring(self, task: asyncio.Task) -> bool:
        async with self._lock:
            if self._monitoring_active:
                return False
            self._monitoring_active = True
            self._monitoring_task = task
            return True

    async def stop_monitoring(self) -> Optional[asyncio.Task]:
        async with self._lock:
            self._monitoring_active = False
            task = self._monitoring_task
            self._monitoring_task = None
            return task

    @property
    def monitor_loop_active(self) -> bool:
        return self._monitor_loop_active

    async def set_monitor_loop_active(self, active: bool):
        async with self._lock:
            self._monitor_loop_active = active

    @property
    def stats(self) -> Stats:
        return self._stats

    async def update_stats(
        self,
        found: Optional[int] = None,
        processed: Optional[int] = None,
        rejected: Optional[int] = None,
        suitable: Optional[int] = None,
    ):
        async with self._lock:
            if found is not None:
                self._stats.found = found
            if processed is not None:
                self._stats.processed = processed
            if rejected is not None:
                self._stats.rejected = rejected
            if suitable is not None:
                self._stats.suitable = suitable

    async def increment_stats(
        self, found: int = 0, processed: int = 0, rejected: int = 0, suitable: int = 0
    ):
        async with self._lock:
            self._stats.found += found
            self._stats.processed += processed
            self._stats.rejected += rejected
            self._stats.suitable += suitable

    async def reset_stats(self):
        async with self._lock:
            self._stats.reset()

    def get_stats_dict(self) -> Dict[str, int]:
        return self._stats.to_dict()

    @property
    def settings(self) -> Dict[str, Any]:
        return self._settings.copy()

    async def update_settings(self, updates: Dict[str, Any]):
        async with self._lock:
            self._settings.update(updates)

    async def set_settings(self, settings: Dict[str, Any]):
        async with self._lock:
            self._settings = settings.copy()

    def get_setting(self, key: str, default: Any = None) -> Any:
        return self._settings.get(key, default)

    async def add_ws_client(self, client: Any):
        async with self._ws_lock:
            if client not in self._ws_clients:
                self._ws_clients.append(client)
                log.debug(f"WS client added, total: {len(self._ws_clients)}")

    async def remove_ws_client(self, client: Any):
        async with self._ws_lock:
            if client in self._ws_clients:
                self._ws_clients.remove(client)
                log.debug(f"WS client removed, total: {len(self._ws_clients)}")

    async def get_ws_clients(self) -> List[Any]:
        async with self._ws_lock:
            return self._ws_clients.copy()

    async def cleanup_ws_clients(self, dead_clients: List[Any]):
        async with self._ws_lock:
            for client in dead_clients:
                if client in self._ws_clients:
                    self._ws_clients.remove(client)
            if dead_clients:
                log.info(
                    f"Cleaned up {len(dead_clients)} dead WS clients, remaining: {len(self._ws_clients)}"
                )

    @property
    def ws_client_count(self) -> int:
        return len(self._ws_clients)

    async def add_improvement_task(self, vacancy_id: str, task: asyncio.Task):
        async with self._lock:
            self._improvement_tasks[vacancy_id] = {"task": task, "status": "running"}

    async def update_improvement_task(self, vacancy_id: str, status: str, result: Any = None):
        async with self._lock:
            if vacancy_id in self._improvement_tasks:
                self._improvement_tasks[vacancy_id]["status"] = status
                if result is not None:
                    self._improvement_tasks[vacancy_id]["result"] = result

    async def get_improvement_task(self, vacancy_id: str) -> Optional[Dict[str, Any]]:
        async with self._lock:
            return self._improvement_tasks.get(vacancy_id)

    async def cleanup_improvement_tasks(self, max_completed: int = 100):
        async with self._lock:
            completed = [
                vid
                for vid, info in self._improvement_tasks.items()
                if info.get("status") in ("completed", "error")
            ]
            if len(completed) > max_completed:
                # Remove oldest completed tasks
                to_remove = completed[:-max_completed]
                for vid in to_remove:
                    del self._improvement_tasks[vid]
                log.info(f"Cleaned up {len(to_remove)} old improvement tasks")

    def track_stage2_task(self, task: asyncio.Task):
        self._stage2_tasks.add(task)
        task.add_done_callback(self._stage2_tasks.discard)

    async def cancel_all_stage2_tasks(self):
        tasks = list(self._stage2_tasks)
        for task in tasks:
            if not task.done():
                task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
            log.info(f"Cancelled {len(tasks)} Stage 2 tasks")

    @property
    def stage2_task_count(self) -> int:
        return len(self._stage2_tasks)

    @asynccontextmanager
    async def locked(self):
        async with self._lock:
            yield self


# Module-level singleton accessor
_state: Optional[AppState] = None


def get_state() -> AppState:
    """Get or create AppState singleton"""
    global _state
    if _state is None:
        _state = AppState()
    return _state


# Convenience functions for common operations
async def update_stats(**kwargs):
    await get_state().update_stats(**kwargs)


async def increment_stats(**kwargs):
    await get_state().increment_stats(**kwargs)


def get_stats() -> Dict[str, int]:
    return get_state().get_stats_dict()


def get_settings() -> Dict[str, Any]:
    return get_state().settings


def is_monitoring_active() -> bool:
    return get_state().monitoring_active
