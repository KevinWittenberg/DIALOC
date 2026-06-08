from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from transcript_lok.diagnostics import collect_diagnostics, collect_readiness, diagnostics_markdown, readiness_markdown
from transcript_lok.llm import CorrectionChange, CorrectionReviewFlag
from transcript_lok.models import Segment, TranscriptDocument, TRANSCRIPT_COLUMNS
from transcript_lok.sessions import load_session, save_session
from transcript_lok.ui import (
    _apply_workflow_profile,
    _auto_save_live,
    _ensure_local_correction_provider,
    _export_llm_artifacts,
    _filter_transcript,
    _mapping_from_table,
    _merge_next,
    _run_file_correction,
    _split_row,
    _stream_live,
    _clear_settings_search,
    _toggle_settings_panel,
    _toggle_live_pause,
)


def test_exports_include_csv_and_docx(tmp_path):
    document = TranscriptDocument(segments=[Segment(0, 1, "Hello", speaker="SPEAKER_00", confidence=0.9)])
    paths = document.write_exports(tmp_path, stem="sample")
    names = {Path(path).name for path in paths}
    assert {"sample.md", "sample.json", "sample.srt", "sample.vtt", "sample.csv", "sample.docx"} <= names
    assert (tmp_path / "sample.csv").read_text(encoding="utf-8-sig").splitlines()[0].split(",") == TRANSCRIPT_COLUMNS
    assert (tmp_path / "sample.docx").read_bytes().startswith(b"PK")


def test_session_save_load_roundtrip(tmp_path):
    document = TranscriptDocument(segments=[Segment(0, 1, "Hello", speaker="SPEAKER_00")])
    path = save_session(
        tmp_path,
        document,
        glossary="LOK",
        speaker_mapping={"SPEAKER_00": "Kevin"},
        summary="Summary",
        llm_outputs={"meeting_summary": {"text": "Summary"}},
    )
    payload = load_session(path)
    assert payload["document"].segments[0].text == "Hello"
    assert payload["glossary"] == "LOK"
    assert payload["speaker_mapping"]["SPEAKER_00"] == "Kevin"
    assert payload["summary"] == "Summary"
    assert payload["llm_outputs"]["meeting_summary"]["text"] == "Summary"


def test_diagnostics_markdown_has_runtime():
    diagnostics = collect_diagnostics("small", "auto", "auto")
    markdown = diagnostics_markdown(diagnostics)
    assert "Runtime Diagnostics" in markdown
    assert diagnostics.expected_runtime in markdown
    assert "CPU threads" in markdown
    assert "Platform" in markdown
    assert "Audio decode" in markdown


def test_readiness_markdown_has_profiles(monkeypatch):
    monkeypatch.setattr("transcript_lok.diagnostics._local_http_status", lambda url: (False, f"{url} unavailable"))
    readiness = collect_readiness(
        "small",
        "auto",
        "auto",
        "pyannote/speaker-diarization-community-1",
        "llama.cpp",
        "gemma-4",
        "http://127.0.0.1:8080/v1",
        "C:\\does-not-exist",
        "",
        disabled_features=["diarization"],
    )
    markdown = readiness_markdown(readiness)
    assert "Runtime Readiness" in markdown
    assert "Model Profiles" in markdown
    assert "llama.cpp API reachable" in markdown
    assert "Admin-disabled features" in markdown


def test_workflow_profile_applies_local_llm_defaults():
    model, device, compute, provider, llm_model, endpoint, status = _apply_workflow_profile("Fast live")
    assert model["value"] == "tiny"
    assert device["value"] == "auto"
    assert compute["value"] == "auto"
    assert provider["value"] == "llama.cpp"
    assert llm_model["value"] == "gemma-4"
    assert endpoint["value"].endswith("/v1")
    assert "Fast live" in status


def test_filter_merge_and_split_helpers():
    rows = [
        [0, 1, "A", "hello world", 0.8],
        [1, 2, "B", "next segment", 0.6],
    ]
    filtered, _ = _filter_transcript(rows, "hello", "", None, None)
    assert len(filtered.index) == 1

    state, merged, _, _ = _merge_next(rows, None, 1)
    assert len(merged.index) == 1
    assert "next segment" in state["segments"][0]["text"]

    state, split, _, _ = _split_row(rows, None, 1, 0.5)
    assert len(split.index) == 3


def test_speaker_mapping_from_table():
    assert _mapping_from_table([["SPEAKER_00", "Kevin"], ["SPEAKER_01", ""]]) == {"SPEAKER_00": "Kevin"}


def test_llm_artifact_export_writes_bundle(tmp_path, monkeypatch):
    monkeypatch.setattr("transcript_lok.ui.OUTPUT_DIR", tmp_path)
    document = TranscriptDocument(segments=[Segment(0, 1, "Hello", speaker="SPEAKER_00")])
    paths, status = _export_llm_artifacts(
        document.to_rows(),
        document.to_dict(),
        {"meeting_summary": {"text": "## Summary\nHello", "provenance": {"provider": "mock", "model": "test"}}},
    )
    names = {Path(path).name for path in paths}
    assert any(name.startswith("llm_artifacts_") and name.endswith(".md") for name in names)
    assert any(name.startswith("llm_artifacts_") and name.endswith(".json") for name in names)
    assert any(name.startswith("corrected_transcript_") for name in names)
    assert "Wrote" in status


def test_file_correction_stores_raw_and_corrected_variants(tmp_path, monkeypatch):
    monkeypatch.setattr("transcript_lok.ui.OUTPUT_DIR", tmp_path)

    def fake_correct(document, config, glossary):
        original = document.segments[0].text
        document.segments[0].text = "ACME Project starts"
        return SimpleNamespace(
            changes=[
                CorrectionChange(
                    segment_index=0,
                    start=0,
                    end=1,
                    speaker="Kevin",
                    original_text=original,
                    corrected_text="ACME Project starts",
                    reason="glossary",
                )
            ],
            review_flags=[
                CorrectionReviewFlag(
                    segment_index=0,
                    start=0,
                    end=1,
                    speaker="Kevin",
                    text="Akme",
                    reason="phonetic uncertainty",
                )
            ],
            provenance={"provider": config.provider, "model": config.model, "task": "correction_pass"},
        )

    monkeypatch.setattr("transcript_lok.ui.correct_transcript", fake_correct)
    raw = TranscriptDocument(segments=[Segment(0, 1, "Akme projekt starts", speaker="Kevin")])

    corrected, status = _run_file_correction(raw, "llama.cpp", "gemma-3-1b", "http://127.0.0.1:8080/v1", "", "ACME Project", "Dutch")

    variants = corrected.metadata["transcript_variants"]
    assert corrected.segments[0].text == "ACME Project starts"
    assert "flagged `1`" in status
    assert Path(variants["raw"]["json"]).read_text(encoding="utf-8").find("Akme projekt") >= 0
    assert Path(variants["corrected"]["json"]).read_text(encoding="utf-8").find("ACME Project") >= 0
    assert Path(variants["raw"]["markdown"]).exists()
    assert Path(variants["corrected"]["markdown"]).exists()


def test_automatic_correction_rejects_cloud_provider():
    with pytest.raises(Exception):
        _ensure_local_correction_provider("openai", "https://api.openai.com/v1")


def test_live_autosave_records_last_safe_path(tmp_path, monkeypatch):
    monkeypatch.setattr("transcript_lok.ui.OUTPUT_DIR", tmp_path)
    document = TranscriptDocument(segments=[Segment(0, 1, "Hello", speaker="LIVE")])
    state = {"committed_segments": 2, "last_autosave_segments": 0}
    _auto_save_live(document, 2, state)
    assert state["last_autosave_segments"] == 2
    assert state["last_safe_session_path"].endswith("session.json")
    assert Path(state["last_safe_session_path"]).exists()


def test_live_pause_button_toggles_state():
    paused, button, status = _toggle_live_pause(False)
    assert paused is True
    assert button["value"] == "Resume"
    assert "Live paused" in status

    paused, button, status = _toggle_live_pause(True)
    assert paused is False
    assert button["value"] == "Pause"
    assert "Live resumed" in status


def test_settings_panel_toggle_and_search_clear():
    opened, button = _toggle_settings_panel(False)
    assert opened is True
    assert button["value"] == "Close settings"

    opened, button = _toggle_settings_panel(True)
    assert opened is False
    assert button["value"] == "Settings"

    cleared = _clear_settings_search("Provider: LLM endpoint")
    assert cleared["value"] is None


def test_stream_live_does_not_commit_while_paused(monkeypatch):
    def fail_transcribe(*_args, **_kwargs):
        raise AssertionError("Paused live stream should not transcribe.")

    monkeypatch.setattr("transcript_lok.ui._transcribe_live_buffer", fail_transcribe)
    audio_state, doc_state, table, _preview, status = _stream_live(
        (16000, np.zeros(16000, dtype=np.float32)),
        None,
        None,
        "tiny",
        "English",
        "cpu",
        "int8",
        "",
        4,
        True,
        True,
        10,
    )
    assert audio_state is None
    assert doc_state["segments"] == []
    assert len(table.index) == 0
    assert "Live paused" in status
