from __future__ import annotations

from app.speech_text import prepare_speech_text


def test_plain_text_unchanged() -> None:
    assert prepare_speech_text("Hello, how are you?") == "Hello, how are you?"


def test_strips_bold_italic_and_headings() -> None:
    text = "# *Hello*\n\n**This** is _great_."
    assert prepare_speech_text(text) == "Hello. This is great."


def test_strips_unmatched_asterisks() -> None:
    assert prepare_speech_text("**Hello there") == "Hello there"
    assert prepare_speech_text("***") == ""


def test_lists_become_sentences() -> None:
    text = "Pick one:\n\n- **Milk**\n- Bread\n- Eggs"
    spoken = prepare_speech_text(text)
    assert spoken == "Pick one: Milk. Bread. Eggs."


def test_links_keep_label_and_drop_url() -> None:
    spoken = prepare_speech_text("See [the docs](https://example.com/path) please.")
    assert spoken == "See the docs please."
    assert "http" not in spoken
    assert "example" not in spoken


def test_code_fences_and_inline_code() -> None:
    text = "Use `af_heart`.\n\n```python\nprint('ok')\n```"
    spoken = prepare_speech_text(text)
    assert "backtick" not in spoken.lower()
    assert "af heart" in spoken
    assert "print('ok')" in spoken


def test_preserves_csharp_and_rewrites_issue_numbers() -> None:
    spoken = prepare_speech_text("Use C# for issue #42.")
    assert spoken == "Use C# for issue number 42."


def test_llm_whatsapp_reply() -> None:
    text = """
## Here's the plan

1. Open **WhatsApp**
2. Send the *voice* note
3. Wait

Visit https://notify-towers.trycloudflare.com for the API.
""".strip()
    spoken = prepare_speech_text(text)
    assert spoken == (
        "Here's the plan. Open WhatsApp. Send the voice note. Wait. Visit for the API."
    )


def test_html_entities_and_tags() -> None:
    assert prepare_speech_text("<b>Hello</b> &amp; hi") == "Hello & hi"


def test_blockquote_and_table() -> None:
    text = "> quoted line\n\n| Name | Role |\n| --- | --- |\n| Ira | voice |"
    spoken = prepare_speech_text(text)
    assert "quoted line" in spoken
    assert "vertical" not in spoken.lower()
    assert "Ira" in spoken
    assert "voice" in spoken


def test_empty_and_whitespace() -> None:
    assert prepare_speech_text("") == ""
    assert prepare_speech_text("   \n  ") == ""
    assert prepare_speech_text("###") == ""
