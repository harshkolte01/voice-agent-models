"""Turn LLM Markdown into text Kokoro can speak without naming symbols."""

from __future__ import annotations

import html
import re

_FENCE_RE = re.compile(r"```[\w+-]*\n?(.*?)```", re.DOTALL)
_INLINE_CODE_RE = re.compile(r"`([^`]+)`")
_IMAGE_RE = re.compile(r"!\[([^\]]*)\]\([^)]+\)")
_LINK_RE = re.compile(r"\[([^\]]+)\]\([^)]+\)")
_REF_LINK_RE = re.compile(r"\[([^\]]+)\]\[[^\]]*\]")
_REF_DEF_RE = re.compile(r"^\s*\[[^\]]+\]:\s+\S+.*$", re.MULTILINE)
_AUTOLINK_RE = re.compile(r"<https?://[^>\s]+>", re.IGNORECASE)
_URL_RE = re.compile(r"https?://[^\s<>\]]+", re.IGNORECASE)
_HTML_TAG_RE = re.compile(r"</?[^>]+>")
_HEADING_RE = re.compile(r"^\s{0,3}#{1,6}[ \t]*", re.MULTILINE)
_HEADING_ONLY_RE = re.compile(r"^\s{0,3}#{1,6}\s*$", re.MULTILINE)
_BLOCKQUOTE_RE = re.compile(r"^\s{0,3}>\s?", re.MULTILINE)
_HR_RE = re.compile(r"^\s*([-*_])(?:\s*\1){2,}\s*$", re.MULTILINE)
_TABLE_SEP_RE = re.compile(
    r"^\s*\|?(?:\s*:?-{3,}:?\s*\|)+\s*:?-{3,}:?\s*\|?\s*$",
    re.MULTILINE,
)
_LIST_ITEM_RE = re.compile(r"^[ \t]*(?:[-*+]|\d+[.)])[ \t]+(.+)$", re.MULTILINE)
_BOLD_RE = re.compile(r"(\*\*\*|___|\*\*|__)(.*?)(\1)")
_ITALIC_STAR_RE = re.compile(r"(?<!\w)\*(?!\s)(.+?)(?<!\s)\*(?!\w)")
_ITALIC_UNDERSCORE_RE = re.compile(r"(?<!\w)_(?!\s)(.+?)(?<!\s)_(?!\w)")
_STRIKE_RE = re.compile(r"~~(.+?)~~")
_ISSUE_HASH_RE = re.compile(r"#(\d+)")
_LEFTOVER_HASH_RE = re.compile(r"(?<![A-Za-z])#")
_SPACE_BEFORE_PUNCT_RE = re.compile(r"\s+([,.;:!?])")
_DUP_PUNCT_RE = re.compile(r"([.!?])(?:\s*[.!?])+")
_MULTI_SPACE_RE = re.compile(r"[ \t]+")


def prepare_speech_text(text: str) -> str:
    """Return speakable prose, or an empty string if nothing remains."""
    spoken = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    spoken = html.unescape(spoken)
    spoken = _FENCE_RE.sub(lambda match: f"\n{match.group(1).strip()}\n", spoken)
    spoken = _INLINE_CODE_RE.sub(r"\1", spoken)
    spoken = _IMAGE_RE.sub(r"\1", spoken)
    spoken = _LINK_RE.sub(r"\1", spoken)
    spoken = _REF_LINK_RE.sub(r"\1", spoken)
    spoken = _REF_DEF_RE.sub("", spoken)
    spoken = _AUTOLINK_RE.sub(" ", spoken)
    spoken = _URL_RE.sub(" ", spoken)
    spoken = _HTML_TAG_RE.sub(" ", spoken)
    spoken = _HEADING_ONLY_RE.sub("", spoken)
    spoken = _HEADING_RE.sub("", spoken)
    spoken = _BLOCKQUOTE_RE.sub("", spoken)
    spoken = _HR_RE.sub("", spoken)
    spoken = _TABLE_SEP_RE.sub("", spoken)
    spoken = spoken.replace("|", " ")
    spoken = _soften_list_items(spoken)
    spoken = _BOLD_RE.sub(r"\2", spoken)
    spoken = _STRIKE_RE.sub(r"\1", spoken)
    spoken = _ITALIC_STAR_RE.sub(r"\1", spoken)
    spoken = _ITALIC_UNDERSCORE_RE.sub(r"\1", spoken)
    spoken = spoken.replace("*", " ")
    spoken = spoken.replace("_", " ")
    spoken = spoken.replace("`", " ")
    spoken = spoken.replace("~", " ")
    spoken = _ISSUE_HASH_RE.sub(r"number \1", spoken)
    spoken = _LEFTOVER_HASH_RE.sub(" ", spoken)
    spoken = spoken.replace("\\", " ")
    return _collapse_for_speech(spoken)


def _soften_list_items(text: str) -> str:
    def replace(match: re.Match[str]) -> str:
        item = match.group(1).strip()
        if item and item[-1] not in ".!?;:":
            item = f"{item}."
        return item

    return _LIST_ITEM_RE.sub(replace, text)


def _collapse_for_speech(text: str) -> str:
    paragraphs = re.split(r"\n\s*\n", text)
    parts: list[str] = []
    for paragraph in paragraphs:
        line = _MULTI_SPACE_RE.sub(" ", paragraph.replace("\n", " ")).strip()
        if line:
            parts.append(line)
    if not parts:
        return ""
    spoken = parts[0]
    for part in parts[1:]:
        if spoken.endswith((".", "!", "?", ":", ";")):
            spoken = f"{spoken} {part}"
        else:
            spoken = f"{spoken}. {part}"
    spoken = _SPACE_BEFORE_PUNCT_RE.sub(r"\1", spoken)
    spoken = _DUP_PUNCT_RE.sub(r"\1", spoken)
    return _MULTI_SPACE_RE.sub(" ", spoken).strip()
