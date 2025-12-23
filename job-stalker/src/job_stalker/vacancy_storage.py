import json
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional

log = logging.getLogger("vacancy_storage")

DATA_DIR = Path("data")
VACANCIES_FILE = DATA_DIR / "vacancies.json"
TRACKER_FILE = DATA_DIR / "tracker.json"


def ensure_data_dir():
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def save_vacancy(vacancy: Dict):
    ensure_data_dir()

    vacancies = load_all_vacancies()

    vacancy["added_at"] = datetime.now().isoformat()
    vacancy["is_new"] = True

    vacancies.append(vacancy)

    try:
        with open(VACANCIES_FILE, "w", encoding="utf-8") as f:
            json.dump(vacancies, f, ensure_ascii=False, indent=2)
    except Exception as e:
        log.error(f"Ошибка сохранения вакансии: {e}")


def load_all_vacancies() -> List[Dict]:
    ensure_data_dir()

    try:
        if VACANCIES_FILE.exists():
            with open(VACANCIES_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception as e:
        log.error(f"Ошибка загрузки вакансий: {e}")

    return []


def mark_all_as_old():
    vacancies = load_all_vacancies()

    for vac in vacancies:
        vac["is_new"] = False

    try:
        with open(VACANCIES_FILE, "w", encoding="utf-8") as f:
            json.dump(vacancies, f, ensure_ascii=False, indent=2)
    except Exception as e:
        log.error(f"Ошибка обновления вакансий: {e}")


def clear_all_vacancies():
    ensure_data_dir()

    try:
        if VACANCIES_FILE.exists():
            VACANCIES_FILE.unlink()
    except Exception as e:
        log.error(f"Ошибка очистки вакансий: {e}")


def get_vacancies_count() -> Dict[str, int]:
    vacancies = load_all_vacancies()

    new_count = sum(1 for v in vacancies if v.get("is_new", False))
    old_count = len(vacancies) - new_count

    return {"total": len(vacancies), "new": new_count, "old": old_count}


def update_vacancy(vacancy_id: str, updates: Dict) -> bool:
    ensure_data_dir()

    vacancies = load_all_vacancies()
    updated = False

    for vac in vacancies:
        if vac.get("id") == vacancy_id:
            vac.update(updates)
            updated = True
            break

    if updated:
        try:
            with open(VACANCIES_FILE, "w", encoding="utf-8") as f:
                json.dump(vacancies, f, ensure_ascii=False, indent=2)
            log.info(f"✅ Vacancy {vacancy_id[:8]} updated")
        except Exception as e:
            log.error(f"Ошибка обновления вакансии: {e}")
            return False

    return updated


def load_tracked_vacancies() -> List[Dict]:
    ensure_data_dir()
    try:
        if TRACKER_FILE.exists():
            with open(TRACKER_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception as e:
        log.error(f"Ошибка загрузки трекера: {e}")
    return []


def save_tracked_vacancies(vacancies: List[Dict]):
    ensure_data_dir()
    try:
        with open(TRACKER_FILE, "w", encoding="utf-8") as f:
            json.dump(vacancies, f, ensure_ascii=False, indent=2)
    except Exception as e:
        log.error(f"Ошибка сохранения трекера: {e}")


def add_to_tracker(vacancy: Dict) -> bool:
    vacancies = load_tracked_vacancies()

    if any(v.get("id") == vacancy.get("id") for v in vacancies):
        return False

    vacancy["tracked_at"] = datetime.now().isoformat()
    vacancy["tracker_status"] = vacancy.get("tracker_status", "wishlist")

    vacancies.insert(0, vacancy)
    save_tracked_vacancies(vacancies)
    log.info(f"✅ Added to tracker: {vacancy.get('id', '')[:8]}")
    return True


def remove_from_tracker(vacancy_id: str) -> bool:
    vacancies = load_tracked_vacancies()
    original_len = len(vacancies)

    vacancies = [v for v in vacancies if v.get("id") != vacancy_id]

    if len(vacancies) < original_len:
        save_tracked_vacancies(vacancies)
        log.info(f"✅ Removed from tracker: {vacancy_id[:8]}")
        return True
    return False


def update_tracker_status(vacancy_id: str, status: str) -> bool:
    vacancies = load_tracked_vacancies()

    for vac in vacancies:
        if vac.get("id") == vacancy_id:
            vac["tracker_status"] = status
            save_tracked_vacancies(vacancies)
            log.info(f"✅ Tracker status updated: {vacancy_id[:8]} -> {status}")
            return True
    return False


def update_tracked_vacancy(vacancy_id: str, updates: Dict) -> bool:
    vacancies = load_tracked_vacancies()

    for vac in vacancies:
        if vac.get("id") == vacancy_id:
            vac.update(updates)
            save_tracked_vacancies(vacancies)
            log.info(f"✅ Tracked vacancy updated: {vacancy_id[:8]}")
            return True
    return False


def get_tracked_vacancy(vacancy_id: str) -> Optional[Dict]:
    vacancies = load_tracked_vacancies()
    for vac in vacancies:
        if vac.get("id") == vacancy_id:
            return vac
    return None


def is_in_tracker(vacancy_id: str) -> bool:
    vacancies = load_tracked_vacancies()
    return any(v.get("id") == vacancy_id for v in vacancies)
