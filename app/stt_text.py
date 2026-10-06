"""Normalize STT transcripts so downstream parsers work for every engine.

Phonon-2 / Parakeet often emit ``p.m.`` and spoken hours (``four p.m.``).
Naive sentence splitters then cut on the abbreviation dots. Whisper may
already emit ``4 PM``. This pass makes those forms consistent without
changing meaning.
"""

from __future__ import annotations

import re

_HOUR_WORDS = {
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
    "eleven": "11",
    "twelve": "12",
}

_HOUR_ALT = "|".join(sorted(_HOUR_WORDS, key=len, reverse=True))

_LETTER_DIGIT = re.compile(r"([A-Za-z])(\d)")
_DIGIT_LETTER = re.compile(r"(\d)([A-Za-z])")
_SPACE_BEFORE_PUNCT = re.compile(r"\s+([,.;:!?])")
_MULTI_SPACE = re.compile(r"[ \t]+")
# Sentence-ending dotted am/pm: keep the clause boundary.
_AMPM_SENTENCE = re.compile(
    r"\b([ap])\s*\.\s*m\s*\.(?=\s+[A-Z])",
    re.IGNORECASE,
)
_AMPM_DOTTED = re.compile(
    r"\b([ap])\s*\.\s*m\.?",
    re.IGNORECASE,
)
_AMPM_AFTER_NUM = re.compile(
    r"(?<=\d)\s*([ap])m\b",
    re.IGNORECASE,
)
_HOUR_WORD_AMPM = re.compile(
    rf"\b({_HOUR_ALT})\s*(AM|PM)\b",
    re.IGNORECASE,
)
_HOUR_WORD_OCLOCK = re.compile(
    rf"\b({_HOUR_ALT})\s+o['’]?clock\b",
    re.IGNORECASE,
)


def normalize_stt_text(text: str) -> str:
    """Return planner-safe transcript text (all STT backends)."""
    spoken = (text or "").strip()
    if not spoken:
        return ""

    spoken = _LETTER_DIGIT.sub(r"\1 \2", spoken)
    spoken = _DIGIT_LETTER.sub(r"\1 \2", spoken)

    def _ampm_sentence(match: re.Match[str]) -> str:
        return "AM." if match.group(1).lower() == "a" else "PM."

    def _ampm(match: re.Match[str]) -> str:
        return "AM" if match.group(1).lower() == "a" else "PM"

    spoken = _AMPM_SENTENCE.sub(_ampm_sentence, spoken)
    spoken = _AMPM_DOTTED.sub(_ampm, spoken)
    spoken = _AMPM_AFTER_NUM.sub(lambda m: f" {_ampm(m)}", spoken)

    def _hour_ampm(match: re.Match[str]) -> str:
        hour = _HOUR_WORDS[match.group(1).lower()]
        return f"{hour} {match.group(2).upper()}"

    def _hour_oclock(match: re.Match[str]) -> str:
        return f"{_HOUR_WORDS[match.group(1).lower()]} o'clock"

    spoken = _HOUR_WORD_AMPM.sub(_hour_ampm, spoken)
    spoken = _HOUR_WORD_OCLOCK.sub(_hour_oclock, spoken)
    spoken = _SPACE_BEFORE_PUNCT.sub(r"\1", spoken)
    return _MULTI_SPACE.sub(" ", spoken).strip()
