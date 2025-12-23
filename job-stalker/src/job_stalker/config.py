import os
from dotenv import load_dotenv

load_dotenv()


def _csv(env_name):
    v = os.getenv(env_name, "").strip()
    return [x.strip() for x in v.split(",") if x.strip()]


API_ID = int(os.getenv("API_ID") or 0)
API_HASH = os.getenv("API_HASH") or ""
SESSION_NAME = os.getenv("SESSION_NAME") or "filter_session"
DEST_CHAT_ID = os.getenv("DEST_CHAT_ID")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or ""
GROQ_API_KEY = os.getenv("GROQ_API_KEY") or ""
KEYWORDS = _csv("KEYWORDS")
BLACKLIST = _csv("BLACKLIST")
REGEXES = _csv("REGEXES")
ALLOW_MEDIA = set(_csv("ALLOW_MEDIA") or ["text", "photo", "video", "document"])
CHANNELS = _csv("CHANNELS")
FORWARD_MODE = os.getenv("FORWARD_MODE", "copy")


def validate_config():
    if not API_ID or not API_HASH:
        raise RuntimeError("API_ID and API_HASH must be set in environment.")
    if not DEST_CHAT_ID:
        raise RuntimeError("DEST_CHAT_ID must be set in environment.")
