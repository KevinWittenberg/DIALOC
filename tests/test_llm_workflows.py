from __future__ import annotations

import pytest

from transcript_lok import llm
from transcript_lok.llm import SummaryConfig, correct_transcript, parse_correction_response, parse_correction_review_flags, run_llm_task
from transcript_lok.models import Segment, TranscriptDocument
from transcript_lok.ui import _follow_up_email, _run_correction_pass, _summarize


def test_llama_cpp_prompt_uses_single_user_message():
    config = SummaryConfig(provider="llama.cpp", model="gemma-4")

    messages = llm._task_messages_for("SPEAKER_00: hello", config, "meeting_summary", glossary_text="ACME")

    assert [message["role"] for message in messages] == ["user"]
    assert "<transcript>" in messages[0]["content"]
    assert "<glossary>" in messages[0]["content"]


def test_openai_compatible_prompt_uses_system_and_user_messages():
    config = SummaryConfig(provider="openai-compatible", model="local")

    messages = llm._task_messages_for("SPEAKER_00: hello", config, "action_points")

    assert [message["role"] for message in messages] == ["system", "user"]
    assert "Onbekend" in messages[1]["content"]


def test_artifact_prompts_support_dutch_and_english():
    dutch = SummaryConfig(provider="llama.cpp", model="gemma-4")
    english = SummaryConfig(provider="llama.cpp", model="gemma-4", artifact_language="en")

    dutch_action = llm._task_messages_for("SPEAKER_00: hello", dutch, "action_points")[0]["content"]
    english_action = llm._task_messages_for("SPEAKER_00: hello", english, "action_points")[0]["content"]
    dutch_summary = llm._task_messages_for("SPEAKER_00: hello", dutch, "meeting_summary")[0]["content"]
    english_summary = llm._task_messages_for("SPEAKER_00: hello", english, "meeting_summary")[0]["content"]

    assert "Wat, Wie, Deadline, Bewijs" in dutch_action
    assert "What, Who, Deadline, Evidence" in english_action
    assert "Samenvatting" in dutch_summary
    assert "Summary" in english_summary


def test_parse_correction_response_accepts_json_object():
    document = TranscriptDocument(segments=[Segment(0, 1, "Akme projekt starts", speaker="Kevin")])

    changes = parse_correction_response(
        '{"segments":[{"segment_index":1,"corrected_text":"ACME Project starts","reason":"glossary","confidence":0.9}]}',
        document,
    )

    assert len(changes) == 1
    assert changes[0].segment_index == 0
    assert changes[0].corrected_text == "ACME Project starts"


def test_parse_correction_response_accepts_review_flags():
    document = TranscriptDocument(segments=[Segment(0, 1, "Akme projekt starts", speaker="Kevin")])

    flags = parse_correction_review_flags(
        '{"segments":[],"review_flags":[{"segment_index":1,"text":"Akme","reason":"phonetic miss","confidence":0.6}]}',
        document,
    )

    assert len(flags) == 1
    assert flags[0].segment_index == 0
    assert flags[0].text == "Akme"
    assert flags[0].reason == "phonetic miss"


def test_correction_pass_auto_applies_text_only(monkeypatch):
    document = TranscriptDocument(segments=[Segment(0, 1, "Akme projekt starts", speaker="Kevin", confidence=0.7)])

    monkeypatch.setattr(
        llm,
        "_chat",
        lambda *_args, **_kwargs: '{"segments":[{"segment_index":1,"corrected_text":"ACME Project starts","reason":"glossary","confidence":0.9}]}',
    )

    result = correct_transcript(document, SummaryConfig(provider="llama.cpp", model="gemma-4"), "ACME Project")

    assert document.segments[0].text == "ACME Project starts"
    assert document.segments[0].start == 0
    assert document.segments[0].speaker == "Kevin"
    assert document.segments[0].confidence == 0.7
    assert len(result.changes) == 1
    assert result.review_flags == []
    assert document.metadata["llm_correction_runs"][0]["changed_segments"] == "1"


def test_correction_prompt_preserves_original_language():
    config = SummaryConfig(provider="llama.cpp", model="gemma-4", artifact_language="en")
    messages = llm._correction_messages_for("[1] [00:00:00.000] Kevin: Akme projekt", config, "ACME Project")
    prompt = messages[0]["content"]
    assert "translate" in prompt
    assert "Keep each segment in its original language" in prompt


def test_malformed_correction_response_applies_nothing(monkeypatch):
    document = TranscriptDocument(segments=[Segment(0, 1, "Akme projekt starts", speaker="Kevin")])
    monkeypatch.setattr(llm, "_chat", lambda *_args, **_kwargs: "not json")

    with pytest.raises(ValueError):
        correct_transcript(document, SummaryConfig(provider="llama.cpp", model="gemma-4"), "ACME Project")

    assert document.segments[0].text == "Akme projekt starts"
    assert "llm_correction_runs" not in document.metadata


def test_run_llm_task_uses_mocked_provider(monkeypatch):
    document = TranscriptDocument(segments=[Segment(0, 1, "Kevin: ship it", speaker="Kevin")])
    monkeypatch.setattr(llm, "_chat", lambda *_args, **_kwargs: "## Summary\nShip it.")

    text = run_llm_task(document, SummaryConfig(provider="ollama", model="local"), task="meeting_summary")

    assert "Ship it" in text


def test_ui_callbacks_store_llm_outputs(monkeypatch):
    rows = [[0, 1, "Kevin", "Akme projekt starts", 0.8]]
    monkeypatch.setattr(
        llm,
        "_chat",
        lambda *_args, **_kwargs: '{"segments":[{"segment_index":1,"corrected_text":"ACME Project starts","reason":"glossary"}]}',
    )

    state, table, preview, audit, latest, outputs, provenance, status = _run_correction_pass(
        rows,
        None,
        "llama.cpp",
        "gemma-4",
        "http://127.0.0.1:8080/v1",
        "",
        "ACME Project",
        "Dutch",
        {},
    )

    assert state["segments"][0]["text"] == "ACME Project starts"
    assert table.iloc[0]["text"] == "ACME Project starts"
    assert "Applied Corrections" in preview or "Applied Corrections" in audit
    assert latest == audit
    assert outputs["correction_pass"]["changes"][0]["corrected_text"] == "ACME Project starts"
    assert outputs["correction_pass"]["provenance"]["artifact_language"] == "nl"
    assert "LLM Provenance" in provenance
    assert "Applied 1" in status


def test_summary_and_follow_up_callbacks_store_outputs(monkeypatch):
    rows = [[0, 1, "Kevin", "Ship the report by Friday.", 0.8]]
    monkeypatch.setattr("transcript_lok.ui.run_llm_task", lambda *_args, **_kwargs: "Generated output")

    summary, latest, outputs, provenance, status = _summarize(
        rows,
        None,
        "ollama",
        "local",
        "",
        "",
        "Action points",
        "",
        "",
        "English",
        {},
    )
    assert summary == "Generated output"
    assert latest == "Generated output"
    assert "action_points" in outputs
    assert outputs["action_points"]["provenance"]["artifact_language"] == "en"
    assert "LLM Provenance" in provenance
    assert "action_points" in status

    email, _, outputs, provenance, status = _follow_up_email(rows, None, "ollama", "local", "", "", "", "Dutch", outputs)
    assert email == "Generated output"
    assert "follow_up_email" in outputs
    assert outputs["follow_up_email"]["provenance"]["artifact_language"] == "nl"
    assert "LLM Provenance" in provenance
    assert "Follow-up email" in status
