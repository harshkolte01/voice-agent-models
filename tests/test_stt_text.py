from __future__ import annotations

from app.stt_text import normalize_stt_text


def test_meeting_four_pm_becomes_numeric_pm() -> None:
    raw = "We have a client meeting Friday at four p.m."
    assert (
        normalize_stt_text(raw)
        == "We have a client meeting Friday at 4 PM"
    )


def test_dotted_pm_does_not_break_next_sentence() -> None:
    raw = "We have a client meeting Friday at four p.m. Please bring notes."
    assert (
        normalize_stt_text(raw)
        == "We have a client meeting Friday at 4 PM. Please bring notes."
    )


def test_already_numeric_forms() -> None:
    assert normalize_stt_text("Friday at 4 p.m.") == "Friday at 4 PM"
    assert normalize_stt_text("Friday at 4 PM") == "Friday at 4 PM"
    assert normalize_stt_text("Friday at 4pm") == "Friday at 4 PM"


def test_am_forms() -> None:
    assert normalize_stt_text("starts at ten a.m.") == "starts at 10 AM"
    assert normalize_stt_text("starts at 10 a.m.") == "starts at 10 AM"


def test_does_not_rewrite_plain_am() -> None:
    assert normalize_stt_text("I am here at four.") == "I am here at four."


def test_phonon_glued_digits_and_space_before_period() -> None:
    raw = "starts at10 in conference room4 ."
    assert normalize_stt_text(raw) == "starts at 10 in conference room 4."


def test_oclock() -> None:
    assert normalize_stt_text("see you at four o'clock") == "see you at 4 o'clock"


def test_empty() -> None:
    assert normalize_stt_text("") == ""
    assert normalize_stt_text("   ") == ""
