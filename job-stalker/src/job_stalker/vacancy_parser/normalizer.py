from __future__ import annotations

import re
import unicodedata
from typing import List

from .rules import (
    BULLET_MARKERS,
    DECORATIVE_PATTERNS,
    EMOJI_PATTERN,
    NOISE_PREFIXES,
)


def normalize_text(text: str) -> str:
    if not text:
        return ""

    result = text

    # 1. Unicode normalization
    result = unicodedata.normalize("NFC", result)

    # 2. Normalize line endings
    result = result.replace("\r\n", "\n").replace("\r", "\n")

    # 3. Remove zero-width characters
    result = _remove_zero_width_chars(result)

    # 4. Remove emoji (keep text content)
    result = _remove_emoji(result)

    # 5. Unify bullet markers
    result = _unify_bullets(result)

    # 6. Remove decorative lines
    result = _remove_decorative_lines(result)

    # 7. Normalize whitespace
    result = _normalize_whitespace(result)

    # 8. Strip noise prefixes from first line
    result = _strip_noise_prefixes(result)

    return result.strip()


def _remove_zero_width_chars(text: str) -> str:
    invisible = "\u200b\u200c\u200d\ufeff\u00ad\u2060"
    for char in invisible:
        text = text.replace(char, "")
    return text


def _remove_emoji(text: str) -> str:
    return EMOJI_PATTERN.sub("", text)


def _unify_bullets(text: str) -> str:
    lines = text.split("\n")
    result_lines: List[str] = []

    for line in lines:
        stripped = line.lstrip()

        bullet_found = False
        for marker in BULLET_MARKERS:
            if stripped.startswith(marker):
                indent = len(line) - len(stripped)
                indent_str = " " * indent

                rest = stripped[len(marker) :].lstrip()
                result_lines.append(f"{indent_str}- {rest}")
                bullet_found = True
                break

        if not bullet_found:
            match = re.match(r"^(\s*)(\d+)[.):]\s*(.*)$", line)
            if match:
                indent, _num, rest = match.groups()
                result_lines.append(f"{indent}- {rest}")
            else:
                result_lines.append(line)

    return "\n".join(result_lines)


def _remove_decorative_lines(text: str) -> str:
    lines = text.split("\n")
    result_lines: List[str] = []

    for line in lines:
        is_decorative = False
        for pattern in DECORATIVE_PATTERNS:
            if pattern.match(line):
                is_decorative = True
                break

        if not is_decorative:
            result_lines.append(line)

    return "\n".join(result_lines)


def _normalize_whitespace(text: str) -> str:
    lines = text.split("\n")

    normalized_lines: List[str] = []
    for line in lines:
        indent_match = re.match(r"^(\s*)", line)
        indent = indent_match.group(1) if indent_match else ""
        content = line[len(indent) :]
        content = re.sub(r"  +", " ", content).strip()
        normalized_lines.append(f"{indent}{content}" if content else "")

    # Collapse excessive empty lines
    result_lines: List[str] = []
    empty_count = 0

    for line in normalized_lines:
        if not line.strip():
            empty_count += 1
            if empty_count <= 2:  # Keep max 2 consecutive empty lines
                result_lines.append("")
        else:
            empty_count = 0
            result_lines.append(line)

    return "\n".join(result_lines)


def _strip_noise_prefixes(text: str) -> str:
    lines = text.split("\n")
    if not lines:
        return text

    first_line = lines[0]
    for prefix in NOISE_PREFIXES:
        if first_line.lower().startswith(prefix.lower()):
            first_line = first_line[len(prefix) :].lstrip()
            break

    lines[0] = first_line
    return "\n".join(lines)


def get_lines(text: str) -> List[str]:
    return text.split("\n")


def is_empty_line(line: str) -> bool:
    return not line.strip()


def is_bullet_line(line: str) -> bool:
    return line.lstrip().startswith("- ")


def get_bullet_content(line: str) -> str:
    stripped = line.lstrip()
    if stripped.startswith("- "):
        return stripped[2:]
    return stripped


def get_indentation(line: str) -> int:
    return len(line) - len(line.lstrip())
