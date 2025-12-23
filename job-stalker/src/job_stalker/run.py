import asyncio
import sys
from .web_ui import app
from uvicorn import Config, Server


async def run_web_only():
    config = Config(app, host="0.0.0.0", port=8000, log_level="info")
    server = Server(config)
    await server.serve()


def main():
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(run_web_only())


if __name__ == "__main__":
    main()
