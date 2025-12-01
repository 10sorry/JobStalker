import PyInstaller.__main__
import os
import platform

# Определяем путь к разделителю в зависимости от ОС
if platform.system() == "Windows":
    PATH_SEPARATOR = ";"
else:
    PATH_SEPARATOR = ":"

# Пути к данным, которые нужно включить в сборку
# Формат: "исходный_путь:путь_в_сборке"
# После реорганизации проекта UI лежит внутри пакета job_stalker
data_to_add = [
    f"job-stalker/src/job_stalker/templates-html{PATH_SEPARATOR}templates",
    f"job-stalker/src/job_stalker/static-files{PATH_SEPARATOR}static",
]

# Параметры для PyInstaller
# Точка входа теперь: job-stalker/src/job_stalker/run.py
pyinstaller_args = [
    "job-stalker/src/job_stalker/run.py",
    "--onefile",
    "--name", "telegram_filter_bot",
    "--clean",
    # Добавляем данные (UI)
]

# Добавляем данные в аргументы
for data in data_to_add:
    pyinstaller_args.extend(["--add-data", data])

# Запускаем сборку
if __name__ == "__main__":
    print(f"Запуск PyInstaller с аргументами: {' '.join(pyinstaller_args)}")
    PyInstaller.__main__.run(pyinstaller_args)
