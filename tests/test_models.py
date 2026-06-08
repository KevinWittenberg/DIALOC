from transcript_lok.glossary import build_asr_prompt, parse_glossary
from transcript_lok import asr
from transcript_lok.asr import ASRConfig, WhisperTranscriber
from transcript_lok.models import Segment, TranscriptDocument, format_timestamp, parse_speaker_mapping
import types


def test_timestamp_formatting():
    assert format_timestamp(65.4321) == "00:01:05,432"
    assert format_timestamp(65.4321, ".") == "00:01:05.432"


def test_glossary_parsing_and_prompt():
    terms = parse_glossary("LOK, Kevin\nOpenAI; Whisper\n# comment\nLOK")
    assert terms == ["LOK", "Kevin", "OpenAI", "Whisper"]
    assert "LOK" in build_asr_prompt("LOK", "en")


def test_speaker_mapping_and_exports():
    doc = TranscriptDocument(
        segments=[
            Segment(start=0, end=1.2, speaker="SPEAKER_00", text="Hello"),
            Segment(start=1.3, end=2.0, speaker="SPEAKER_01", text="Hi"),
        ]
    )
    mapping = parse_speaker_mapping("SPEAKER_00 = Kevin\nSPEAKER_01: Anna")
    doc.rename_speakers(mapping)
    assert doc.speaker_labels() == ["Anna", "Kevin"]
    assert "Kevin: Hello" in doc.to_srt()
    assert "Anna: Hi" in doc.plain_text()


def test_auto_device_skips_cuda_when_runtime_is_missing(monkeypatch):
    monkeypatch.setattr(asr, "_cuda_dependencies_available", lambda: False)
    assert asr._model_attempts("auto", "auto") == [("cpu", "int8")]


def test_default_asr_model_is_interactive_size():
    assert ASRConfig().model_size == "small"


def test_asr_cpu_thread_settings_reach_whisper_model(monkeypatch):
    calls = []

    class FakeWhisperModel:
        def __init__(self, model_size, **kwargs):
            calls.append((model_size, kwargs))

    monkeypatch.setitem(
        __import__("sys").modules,
        "faster_whisper",
        types.SimpleNamespace(WhisperModel=FakeWhisperModel),
    )
    asr._MODEL_CACHE.clear()

    WhisperTranscriber(ASRConfig(model_size="tiny", device="cpu", compute_type="int8", cpu_threads=2, num_workers=1))

    assert calls == [
        (
            "tiny",
            {
                "device": "cpu",
                "compute_type": "int8",
                "cpu_threads": 2,
                "num_workers": 1,
            },
        )
    ]
    assert ("tiny", "cpu", "int8", 2, 1) in asr.loaded_model_keys()
