from __future__ import annotations

from collections.abc import Iterator
import ctypes.util
import inspect
import shutil
from dataclasses import dataclass
from typing import Any

import numpy as np

from .audio import TARGET_SAMPLE_RATE, decode_file_to_array, resample_audio
from .glossary import build_asr_prompt, parse_glossary
from .models import Segment, TranscriptDocument, Word


@dataclass(frozen=True)
class ASRConfig:
    model_size: str = "small"
    language: str | None = None
    device: str = "auto"
    compute_type: str = "auto"
    beam_size: int = 5
    vad_filter: bool = True
    cpu_threads: int = 0
    num_workers: int = 1


@dataclass(slots=True)
class _ModelBundle:
    model: Any
    device: str
    compute_type: str


@dataclass(slots=True)
class TranscriptionUpdate:
    completed_seconds: float
    total_seconds: float | None
    document: TranscriptDocument | None = None
    is_final: bool = False


_MODEL_CACHE: dict[tuple[str, str, str, int, int], _ModelBundle] = {}


def loaded_model_keys() -> list[tuple[str, str, str, int, int]]:
    return sorted(_MODEL_CACHE.keys())


def is_model_loaded(model_size: str, device: str, compute_type: str, cpu_threads: int = 0, num_workers: int = 1) -> bool:
    return (model_size, device, compute_type, int(cpu_threads or 0), int(num_workers or 1)) in _MODEL_CACHE


class WhisperTranscriber:
    def __init__(self, config: ASRConfig):
        self.config = config
        self.bundle = self._load_model(config)

    def transcribe_file(self, audio_path: str, glossary_text: str | None = None) -> TranscriptDocument:
        for update in self.transcribe_file_iter(audio_path, glossary_text):
            if update.is_final and update.document is not None:
                return update.document
        raise RuntimeError("Transcription finished without a final transcript document.")

    def transcribe_file_iter(
        self,
        audio_path: str,
        glossary_text: str | None = None,
        *,
        cancel_check: Any | None = None,
    ) -> Iterator[TranscriptionUpdate]:
        yield from self._transcribe_iter(
            audio_path,
            glossary_text,
            offset_seconds=0.0,
            audio_path=audio_path,
            total_seconds=_audio_duration_seconds(audio_path),
            cancel_check=cancel_check,
        )

    def transcribe_file_chunked_iter(
        self,
        audio_path: str,
        glossary_text: str | None = None,
        *,
        chunk_seconds: float = 600.0,
        overlap_seconds: float = 5.0,
        cancel_check: Any | None = None,
    ) -> Iterator[TranscriptionUpdate]:
        audio = decode_file_to_array(audio_path, TARGET_SAMPLE_RATE)
        total_seconds = len(audio) / TARGET_SAMPLE_RATE
        if total_seconds <= 0:
            yield TranscriptionUpdate(0.0, total_seconds, document=TranscriptDocument(audio_path=audio_path), is_final=True)
            return

        chunk_seconds = max(float(chunk_seconds), 30.0)
        overlap_seconds = max(0.0, min(float(overlap_seconds), chunk_seconds / 3))
        step_seconds = max(1.0, chunk_seconds - overlap_seconds)
        segments: list[Segment] = []
        chunk_start = 0.0

        while chunk_start < total_seconds:
            if cancel_check and cancel_check():
                yield TranscriptionUpdate(
                    completed_seconds=chunk_start,
                    total_seconds=total_seconds,
                    document=_document_from_partial(
                        self,
                        segments,
                        audio_path,
                        total_seconds,
                        cancelled=True,
                    ),
                    is_final=True,
                )
                return

            chunk_end = min(total_seconds, chunk_start + chunk_seconds)
            start_sample = int(chunk_start * TARGET_SAMPLE_RATE)
            end_sample = int(chunk_end * TARGET_SAMPLE_RATE)
            chunk_audio = audio[start_sample:end_sample]
            keep_start = chunk_start if chunk_start == 0 else chunk_start + overlap_seconds / 2
            keep_end = chunk_end if chunk_end >= total_seconds else chunk_end - overlap_seconds / 2

            partial = self.transcribe_audio_array(
                chunk_audio,
                TARGET_SAMPLE_RATE,
                glossary_text,
                offset_seconds=chunk_start,
            )
            for segment in partial.segments:
                midpoint = (segment.start + segment.end) / 2
                if keep_start <= midpoint <= keep_end:
                    segments.append(segment)

            yield TranscriptionUpdate(completed_seconds=chunk_end, total_seconds=total_seconds)
            if chunk_end >= total_seconds:
                break
            chunk_start += step_seconds

        yield TranscriptionUpdate(
            completed_seconds=total_seconds,
            total_seconds=total_seconds,
            document=_document_from_partial(self, segments, audio_path, total_seconds, cancelled=False),
            is_final=True,
        )

    def transcribe_audio_array(
        self,
        samples: np.ndarray,
        sample_rate: int,
        glossary_text: str | None = None,
        *,
        offset_seconds: float = 0.0,
    ) -> TranscriptDocument:
        audio = resample_audio(samples, sample_rate, TARGET_SAMPLE_RATE)
        total_seconds = len(audio) / TARGET_SAMPLE_RATE
        for update in self._transcribe_iter(
            audio,
            glossary_text,
            offset_seconds=offset_seconds,
            total_seconds=total_seconds,
        ):
            if update.is_final and update.document is not None:
                return update.document
        raise RuntimeError("Transcription finished without a final transcript document.")

    def _transcribe_iter(
        self,
        audio: str | np.ndarray,
        glossary_text: str | None,
        *,
        offset_seconds: float,
        audio_path: str | None = None,
        total_seconds: float | None = None,
        cancel_check: Any | None = None,
    ) -> Iterator[TranscriptionUpdate]:
        language = self.config.language or None
        prompt = build_asr_prompt(glossary_text, language)
        glossary_terms = parse_glossary(glossary_text)
        kwargs: dict[str, Any] = {
            "language": language,
            "beam_size": self.config.beam_size,
            "vad_filter": self.config.vad_filter,
            "word_timestamps": True,
            "condition_on_previous_text": True,
        }
        if prompt:
            kwargs["initial_prompt"] = prompt
        if glossary_terms and _supports_kwarg(self.bundle.model.transcribe, "hotwords"):
            kwargs["hotwords"] = ", ".join(glossary_terms)

        try:
            if cancel_check and cancel_check():
                yield TranscriptionUpdate(0.0, total_seconds, document=TranscriptDocument(audio_path=audio_path), is_final=True)
                return
            segments_iter, info = self.bundle.model.transcribe(audio, **kwargs)
            yield from self._updates_from_segments(
                segments_iter,
                info,
                offset_seconds=offset_seconds,
                audio_path=audio_path,
                total_seconds=total_seconds,
                cancel_check=cancel_check,
            )
        except RuntimeError as exc:
            if self.config.device == "auto" and self.bundle.device == "cuda" and _is_cuda_runtime_error(exc):
                self.bundle = self._load_model(
                    ASRConfig(
                        model_size=self.config.model_size,
                        language=self.config.language,
                        device="cpu",
                        compute_type="int8" if self.config.compute_type == "auto" else self.config.compute_type,
                        beam_size=self.config.beam_size,
                        vad_filter=self.config.vad_filter,
                        cpu_threads=self.config.cpu_threads,
                        num_workers=self.config.num_workers,
                    )
                )
                segments_iter, info = self.bundle.model.transcribe(audio, **kwargs)
                yield from self._updates_from_segments(
                    segments_iter,
                    info,
                    offset_seconds=offset_seconds,
                    audio_path=audio_path,
                    total_seconds=total_seconds,
                    cancel_check=cancel_check,
                )
                return
            raise

    def _updates_from_segments(
        self,
        segments_iter: Any,
        info: Any,
        *,
        offset_seconds: float,
        audio_path: str | None,
        total_seconds: float | None,
        cancel_check: Any | None = None,
    ) -> Iterator[TranscriptionUpdate]:
        segments: list[Segment] = []
        resolved_total = total_seconds or getattr(info, "duration", None)
        completed_seconds = 0.0
        for raw_segment in segments_iter:
            if cancel_check and cancel_check():
                yield TranscriptionUpdate(
                    completed_seconds=completed_seconds,
                    total_seconds=resolved_total,
                    document=_document_from_partial(self, segments, audio_path, resolved_total, cancelled=True),
                    is_final=True,
                )
                return
            completed_seconds = max(completed_seconds, float(raw_segment.end or 0.0))
            words = [
                Word(
                    start=float(word.start or 0.0) + offset_seconds,
                    end=float(word.end or 0.0) + offset_seconds,
                    text=str(word.word or "").strip(),
                    probability=getattr(word, "probability", None),
                )
                for word in (raw_segment.words or [])
            ]
            text = str(raw_segment.text or "").strip()
            if not text:
                continue
            confidence = _segment_confidence(words)
            segments.append(
                Segment(
                    start=float(raw_segment.start) + offset_seconds,
                    end=float(raw_segment.end) + offset_seconds,
                    text=text,
                    confidence=confidence,
                    words=words,
                )
            )
            yield TranscriptionUpdate(
                completed_seconds=completed_seconds,
                total_seconds=resolved_total,
            )

        detected_language = self.config.language or getattr(info, "language", None)
        document = _document_from_partial(
            self,
            segments,
            audio_path,
            resolved_total,
            cancelled=False,
            language=detected_language,
            language_probability=getattr(info, "language_probability", None),
        )
        yield TranscriptionUpdate(
            completed_seconds=resolved_total or completed_seconds,
            total_seconds=resolved_total,
            document=document,
            is_final=True,
        )

    @classmethod
    def _load_model(cls, config: ASRConfig) -> _ModelBundle:
        requested_device = config.device
        requested_compute = config.compute_type
        attempts = _model_attempts(requested_device, requested_compute)
        last_error: Exception | None = None

        for device, compute_type in attempts:
            key = (config.model_size, device, compute_type, int(config.cpu_threads or 0), int(config.num_workers or 1))
            if key in _MODEL_CACHE:
                return _MODEL_CACHE[key]
            try:
                from faster_whisper import WhisperModel

                model = WhisperModel(
                    config.model_size,
                    device=device,
                    compute_type=compute_type,
                    cpu_threads=int(config.cpu_threads or 0),
                    num_workers=int(config.num_workers or 1),
                )
                bundle = _ModelBundle(model=model, device=device, compute_type=compute_type)
                _MODEL_CACHE[key] = bundle
                return bundle
            except Exception as exc:
                last_error = exc

        raise RuntimeError(f"Could not load faster-whisper model {config.model_size!r}: {last_error}") from last_error


def _model_attempts(device: str, compute_type: str) -> list[tuple[str, str]]:
    if device != "auto":
        return [(device, _compute_type_for(device, compute_type))]

    attempts = [("cpu", _compute_type_for("cpu", compute_type))]
    if _cuda_dependencies_available():
        attempts.insert(0, ("cuda", _compute_type_for("cuda", compute_type)))
    return attempts


def _compute_type_for(device: str, compute_type: str) -> str:
    if compute_type != "auto":
        return compute_type
    return "float16" if device == "cuda" else "int8"


def _supports_kwarg(fn: Any, name: str) -> bool:
    try:
        signature = inspect.signature(fn)
    except (TypeError, ValueError):
        return False
    return name in signature.parameters


def _cuda_dependencies_available() -> bool:
    try:
        import ctranslate2

        if ctranslate2.get_cuda_device_count() < 1:
            return False
    except Exception:
        return False

    # CTranslate2 can see the GPU while still failing at first inference when
    # the CUDA runtime libraries are not visible to the process.
    return bool(
        ctypes.util.find_library("cublas")
        or ctypes.util.find_library("cublas64_12")
        or shutil.which("cublas64_12.dll")
    )


def _is_cuda_runtime_error(exc: RuntimeError) -> bool:
    message = str(exc).lower()
    return "cublas" in message or "cudnn" in message or "cuda" in message


def _segment_confidence(words: list[Word]) -> float | None:
    probabilities = [float(word.probability) for word in words if word.probability is not None]
    if not probabilities:
        return None
    return sum(probabilities) / len(probabilities)


def _document_from_partial(
    transcriber: WhisperTranscriber,
    segments: list[Segment],
    audio_path: str | None,
    total_seconds: float | None,
    *,
    cancelled: bool,
    language: str | None = None,
    language_probability: float | None = None,
) -> TranscriptDocument:
    metadata = {
        "asr_model": transcriber.config.model_size,
        "asr_device": transcriber.bundle.device,
        "asr_compute_type": transcriber.bundle.compute_type,
        "asr_beam_size": transcriber.config.beam_size,
        "asr_cpu_threads": transcriber.config.cpu_threads,
        "asr_num_workers": transcriber.config.num_workers,
        "language_probability": language_probability,
        "cancelled": cancelled,
    }
    if total_seconds:
        metadata["audio_duration_seconds"] = total_seconds
    return TranscriptDocument(
        segments=segments,
        language=language or transcriber.config.language,
        audio_path=audio_path,
        metadata=metadata,
    )


def _audio_duration_seconds(audio_path: str) -> float | None:
    try:
        import av

        with av.open(str(audio_path)) as container:
            for stream in container.streams.audio:
                if stream.duration and stream.time_base:
                    duration = float(stream.duration * stream.time_base)
                    if duration > 0:
                        return duration
            if container.duration and container.duration > 0:
                return float(container.duration / av.time_base)
    except Exception:
        pass

    try:
        import soundfile as sf

        info = sf.info(str(audio_path))
        return float(info.duration)
    except Exception:
        return None
