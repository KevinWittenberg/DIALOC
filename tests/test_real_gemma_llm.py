from __future__ import annotations

import os

import pytest

from transcript_lok.llm import SummaryConfig, correct_transcript, provider_health, run_llm_task
from transcript_lok.models import Segment, TranscriptDocument


@pytest.mark.skipif(os.getenv("LLM_REAL_TEST") != "1", reason="Set LLM_REAL_TEST=1 to run the local Gemma smoke test.")
def test_real_gemma_llama_cpp_workflow():
    config = SummaryConfig(
        provider="llama.cpp",
        model=os.getenv("LLM_REAL_MODEL", "gemma-4"),
        endpoint=os.getenv("LLM_REAL_ENDPOINT", "http://127.0.0.1:8080/v1"),
        max_chars_per_call=12_000,
        artifact_language="en",
    )
    ok, message = provider_health(config)
    if not ok:
        pytest.skip(message)

    glossary = "ACME Project\nTranscript LOK\nKevin\nFriday"
    document = TranscriptDocument(
        segments=[
            Segment(0, 2, "Kevn says the Akme projekt report must be finisht by Friday.", speaker="Kevin"),
            Segment(2, 4, "Anna will send the draft and ask legal for feed back.", speaker="Anna"),
        ]
    )

    result = correct_transcript(document, config, glossary)
    assert result.changes
    corrected = document.plain_text()
    assert "ACME" in corrected or "Project" in corrected

    outputs = {
        "meeting_summary": run_llm_task(document, config, task="meeting_summary", glossary_text=glossary),
        "action_points": run_llm_task(document, config, task="action_points", glossary_text=glossary),
        "decision_summary_internal_anonymous": run_llm_task(document, config, task="decision_summary_internal_anonymous", glossary_text=glossary),
        "decision_summary_external_anonymous": run_llm_task(document, config, task="decision_summary_external_anonymous", glossary_text=glossary),
        "follow_up_email": run_llm_task(document, config, task="follow_up_email", glossary_text=glossary),
    }
    for text in outputs.values():
        assert text.strip()
    assert "What" in outputs["action_points"] or "Action" in outputs["action_points"]
    assert "Subject" in outputs["follow_up_email"]
