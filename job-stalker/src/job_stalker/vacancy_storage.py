import json
import os
from datetime import datetime
from typing import List, Dict

VACANCIES_FILE = "./data/vacancies.json"

def ensure_data_dir():
    """Создает папку data если её нет"""
    os.makedirs("./data", exist_ok=True)

def save_vacancy(vacancy: Dict):
    """Сохраняет вакансию в файл"""
    ensure_data_dir()

    vacancies = load_all_vacancies()

    vacancy['added_at'] = datetime.now().isoformat()
    vacancy['is_new'] = True

    vacancies.append(vacancy)

    try:
        with open(VACANCIES_FILE, 'w', encoding='utf-8') as f:
            json.dump(vacancies, f, ensure_ascii=False, indent=2)
    except Exception as e:
        safe_print(f"Ошибка сохранения вакансии: {e}")

def load_all_vacancies() -> List[Dict]:
    """Загружает все сохраненные вакансии"""
    ensure_data_dir()

    try:
        if os.path.exists(VACANCIES_FILE):
            with open(VACANCIES_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
    except Exception as e:
        safe_print(f"Ошибка загрузки вакансий: {e}")

    return []

def mark_all_as_old():
    """Помечает все вакансии как старые (не новые)"""
    vacancies = load_all_vacancies()

    for vac in vacancies:
        vac['is_new'] = False

    try:
        with open(VACANCIES_FILE, 'w', encoding='utf-8') as f:
            json.dump(vacancies, f, ensure_ascii=False, indent=2)
    except Exception as e:
        safe_print(f"Ошибка обновления вакансий: {e}")

def clear_all_vacancies():
    """Удаляет все вакансии"""
    ensure_data_dir()

    try:
        if os.path.exists(VACANCIES_FILE):
            os.remove(VACANCIES_FILE)
    except Exception as e:
        safe_print(f"Ошибка очистки вакансий: {e}")

def get_vacancies_count() -> Dict[str, int]:
    """Возвращает количество новых и старых вакансий"""
    vacancies = load_all_vacancies()

    new_count = sum(1 for v in vacancies if v.get('is_new', False))
    old_count = len(vacancies) - new_count

    return {
        'total': len(vacancies),
        'new': new_count,
        'old': old_count
    }
