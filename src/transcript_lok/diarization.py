from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import warnings

import soundfile as sf

from .audio import decode_file_to_wav16k
from .models import Segment, TranscriptDocument, Word


DEFAULT_DIARIZATION_MODEL = "pyannote/speaker-diarization-community-1"


@dataclass(frozen=True)
class DiarizationConfig:
    model_name_or_path: str = DEFAULT_DIARIZATION_MODEL
    hf_token: str | None = None
    device: str = "auto"
    min_speakers: int | None = None
    max_speakers: int | None = None


@dataclass(slots=True)
class DiarizationTurn:
    start: float
    end: float
    speaker: str


_PIPELINE_CACHE: dict[tuple[str, str | None, str], Any] = {}


class SpeakerDiarizer:
    def __init__(self, config: DiarizationConfig):
        self.config = config
        self.pipeline = self._load_pipeline(config)

    def diarize(self, audio_path: str) -> list[DiarizationTurn]:
        wav_path = decode_file_to_wav16k(audio_path)
        audio_input = _load_waveform_for_pyannote(wav_path)
        kwargs: dict[str, int] = {}
        if self.config.min_speakers:
            kwargs["min_speakers"] = int(self.config.min_speakers)
        if self.config.max_speakers:
            kwargs["max_speakers"] = int(self.config.max_speakers)

        output = self.pipeline(audio_input, **kwargs)
        annotation = getattr(output, "exclusive_speaker_diarization", None)
        if annotation is None:
            annotation = getattr(output, "speaker_diarization", output)
        return list(_iter_turns(annotation))

    def apply(self, audio_path: str, transcript: TranscriptDocument) -> TranscriptDocument:
        turns = self.diarize(audio_path)
        assign_speakers(transcript, turns)
        transcript.metadata["diarization_model"] = self.config.model_name_or_path
        transcript.metadata["speaker_turns"] = [turn.__dict__ for turn in turns]
        return transcript

    @classmethod
    def _load_pipeline(cls, config: DiarizationConfig) -> Any:
        key = (config.model_name_or_path, config.hf_token, config.device)
        if key in _PIPELINE_CACHE:
            return _PIPELINE_CACHE[key]

        suppress_pyannote_torchcodec_warning()
        try:
            from pyannote.audio import Pipeline
        except Exception as exc:
            raise RuntimeError("pyannote.audio is not installed. Install the project requirements first.") from exc

        kwargs = {}
        if config.hf_token:
            kwargs["token"] = config.hf_token

        try:
            pipeline = Pipeline.from_pretrained(config.model_name_or_path, **kwargs)
        except Exception as exc:
            if Path(config.model_name_or_path).exists():
                raise
            raise RuntimeError(
                "Could not load the diarization model. For pyannote Community-1, accept the model terms on "
                "Hugging Face and provide a token, or point the app at a previously downloaded local model path."
            ) from exc

        device = _resolve_torch_device(config.device)
        if device is not None:
            pipeline.to(device)

        _PIPELINE_CACHE[key] = pipeline
        return pipeline


def assign_speakers(transcript: TranscriptDocument, turns: list[DiarizationTurn]) -> None:
    for segment in transcript.segments:
        speaker = _best_speaker(segment.start, segment.end, turns)
        if speaker:
            segment.speaker = speaker
        for word in segment.words:
            word_speaker = _best_speaker(word.start, word.end, turns)
            if word_speaker:
                word.speaker = word_speaker


def _iter_turns(annotation: Any) -> list[DiarizationTurn]:
    turns: list[DiarizationTurn] = []
    if hasattr(annotation, "itertracks"):
        for turn, _, speaker in annotation.itertracks(yield_label=True):
            turns.append(DiarizationTurn(float(turn.start), float(turn.end), str(speaker)))
        return turns

    for item in annotation:
        if len(item) == 2:
            turn, speaker = item
        else:
            turn, _, speaker = item
        turns.append(DiarizationTurn(float(turn.start), float(turn.end), str(speaker)))
    return turns


def _best_speaker(start: float, end: float, turns: list[DiarizationTurn]) -> str | None:
    if not turns:
        return None
    midpoint = (start + end) / 2
    overlap_by_speaker: dict[str, float] = {}
    for turn in turns:
        overlap = max(0.0, min(end, turn.end) - max(start, turn.start))
        if overlap > 0:
            overlap_by_speaker[turn.speaker] = overlap_by_speaker.get(turn.speaker, 0.0) + overlap
    if overlap_by_speaker:
        return max(overlap_by_speaker.items(), key=lambda item: item[1])[0]

    for turn in turns:
        if turn.start <= midpoint <= turn.end:
            return turn.speaker
    return min(turns, key=lambda turn: min(abs(turn.start - midpoint), abs(turn.end - midpoint))).speaker


def _resolve_torch_device(device: str) -> Any | None:
    if device == "cpu":
        import torch

        return torch.device("cpu")
    try:
        import torch

        if device == "cuda" or (device == "auto" and torch.cuda.is_available()):
            return torch.device("cuda")
    except Exception:
        return None
    return None


def _load_waveform_for_pyannote(wav_path: str) -> dict[str, Any]:
    import torch

    samples, sample_rate = sf.read(wav_path, dtype="float32", always_2d=True)
    waveform = torch.from_numpy(samples.T.copy())
    return {"waveform": waveform, "sample_rate": int(sample_rate)}


def suppress_pyannote_torchcodec_warning() -> None:
    warnings.filterwarnings(
        "ignore",
        message=r"(?s).*torchcodec is not installed correctly.*",
        category=UserWarning,
    )
