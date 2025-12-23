import re
from typing import List
from .config import KEYWORDS, BLACKLIST, REGEXES, ALLOW_MEDIA

compiled_regexes = [re.compile(r, re.IGNORECASE) for r in REGEXES if r]


def text_from_message(message) -> str:
    return (message.text or message.caption or "") or ""


def has_allowed_media(message) -> bool:
    if message.photo:
        t = "photo"
    elif message.video:
        t = "video"
    elif message.document:
        t = "document"
    elif message.sticker:
        t = "sticker"
    elif message.voice:
        t = "voice"
    elif message.audio:
        t = "audio"
    else:
        t = "text"
    return t in ALLOW_MEDIA


def matches_filters(message) -> bool:
    txt = text_from_message(message).lower()

    if not has_allowed_media(message):
        return False

    for b in BLACKLIST:
        if b.lower() in txt:
            return False

    if compiled_regexes:
        for r in compiled_regexes:
            if r.search(txt):
                return True
    if KEYWORDS:
        for kw in KEYWORDS:
            if kw.lower() in txt:
                return True
        return False

    return True
