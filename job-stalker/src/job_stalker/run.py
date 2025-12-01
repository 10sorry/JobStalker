import asyncio
from .web_ui import app
from uvicorn import Config, Server


async def run_web_only():
    """Запускаем только веб-сервер, бот запустится по кнопке START SCAN"""
    # Uvicorn запускаем через Server, чтобы не вызывать asyncio.run()
    config = Config(app, host="0.0.0.0", port=8000, log_level="info")
    server = Server(config)
    await server.serve()


def main():
    """Точка входа для консольного скрипта."""
    asyncio.run(run_web_only())


if __name__ == "__main__":
    main()

