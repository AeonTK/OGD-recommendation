from __future__ import annotations

import html
import re
from typing import Any, List


def clean_text(text: str | None) -> str:
    """Clean title/description-style text.

    - Decode HTML entities (e.g. &agrave; -> à)
    - Strip HTML tags while keeping inner text
    - Simplify Markdown links/bold/italics
    - Remove simple ASCII decorations
    - Normalize whitespace
    """

    if not text:
        return ""

    # Decode HTML entities
    text = html.unescape(text)

    # Remove HTML tags (e.g. <a href="...">text</a> -> text)
    text = re.sub(r"<[^>]+>", "", text)

    # Markdown links: [Text](url) -> Text
    text = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", text)

    # Markdown bold/italics: **Text** / __Text__ / *Text* / _Text_ -> Text
    text = re.sub(r"[\*_]{2,}(.*?)[\*_]{2,}", r"\1", text)
    text = re.sub(r"[\*_](.*?)[\*_]", r"\1", text)

    # ASCII decorations like =====
    text = re.sub(r"={3,}", "", text)

    # Normalize whitespace and newlines
    text = text.replace("\r\n", " ").replace("\n", " ")
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def clean_keywords(keywords_value: Any) -> List[str]:
    """Clean keywords field.

    - Treat "N_A", "N/A", "NONE" or empty as no keywords
    - Split on semicolons, colons, or commas
    - Strip whitespace and drop empties

    Returns a list of keyword strings.
    """

    if not keywords_value:
        return []

    # Many records use a single string; tolerate lists as well.
    if isinstance(keywords_value, list):
        raw = ";".join(str(k) for k in keywords_value)
    else:
        raw = str(keywords_value)

    raw = raw.strip()
    if not raw or raw.upper() in {"N_A", "N/A", "NONE"}:
        return []

    parts = [p.strip() for p in re.split(r"[;:,]", raw)]
    return [p for p in parts if p]
