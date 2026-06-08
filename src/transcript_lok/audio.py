from __future__ import annotations

from pathlib import Path
from tempfile import NamedTemporaryFile
from dataclasses import dataclass

import numpy as np
import soundfile as sf


TARGET_SAMPLE_RATE = 16_000


@dataclass(frozen=True)
class AudioMetadata:
    path: str
    duration_seconds: float | None = None
    sample_rate: int | None = None
    channels: int | None = None
    format_name: str | None = None
    size_bytes: int | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "path": self.path,
            "duration_seconds": self.duration_seconds,
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "format_name": self.format_name,
            "size_bytes": self.size_bytes,
        }


def to_mono_float32(samples: np.ndarray) -> np.ndarray:
    array = np.asarray(samples)
    if array.ndim == 2:
        array = array.mean(axis=1)
    if array.dtype.kind in {"i", "u"}:
        info = np.iinfo(array.dtype)
        scale = max(abs(info.min), info.max)
        array = array.astype(np.float32) / scale
    else:
        array = array.astype(np.float32, copy=False)
    array = np.nan_to_num(array, copy=False)
    return np.clip(array, -1.0, 1.0)


def resample_audio(samples: np.ndarray, source_rate: int, target_rate: int = TARGET_SAMPLE_RATE) -> np.ndarray:
    samples = to_mono_float32(samples)
    if int(source_rate) == int(target_rate):
        return samples

    try:
        from scipy.signal import resample_poly

        gcd = np.gcd(int(source_rate), int(target_rate))
        up = int(target_rate // gcd)
        down = int(source_rate // gcd)
        return resample_poly(samples, up, down).astype(np.float32)
    except Exception:
        duration = len(samples) / float(source_rate)
        old_positions = np.linspace(0, duration, num=len(samples), endpoint=False)
        new_length = int(round(duration * target_rate))
        new_positions = np.linspace(0, duration, num=new_length, endpoint=False)
        return np.interp(new_positions, old_positions, samples).astype(np.float32)


def write_wav(path: str | Path, samples: np.ndarray, sample_rate: int) -> str:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    sf.write(target, to_mono_float32(samples), int(sample_rate), subtype="PCM_16")
    return str(target.resolve())


def write_temp_wav(samples: np.ndarray, sample_rate: int, suffix: str = ".wav") -> str:
    temp = NamedTemporaryFile(delete=False, suffix=suffix)
    temp.close()
    return write_wav(temp.name, samples, sample_rate)


def decode_file_to_wav16k(audio_path: str | Path, output_path: str | Path | None = None) -> str:
    from faster_whisper.audio import decode_audio

    samples = decode_audio(str(audio_path), sampling_rate=TARGET_SAMPLE_RATE)
    if output_path is None:
        temp = NamedTemporaryFile(delete=False, suffix=".wav")
        temp.close()
        output_path = temp.name
    return write_wav(output_path, samples, TARGET_SAMPLE_RATE)


def decode_file_to_array(audio_path: str | Path, sample_rate: int = TARGET_SAMPLE_RATE) -> np.ndarray:
    from faster_whisper.audio import decode_audio

    return decode_audio(str(audio_path), sampling_rate=sample_rate).astype(np.float32)


def audio_metadata(audio_path: str | Path) -> AudioMetadata:
    path = Path(audio_path)
    size = path.stat().st_size if path.exists() else None

    try:
        import av

        with av.open(str(path)) as container:
            audio_stream = next(iter(container.streams.audio), None)
            duration = None
            sample_rate = None
            channels = None
            if audio_stream is not None:
                sample_rate = getattr(audio_stream, "rate", None)
                layout = getattr(audio_stream, "layout", None)
                channels = len(layout.channels) if layout is not None else None
                if audio_stream.duration and audio_stream.time_base:
                    duration = float(audio_stream.duration * audio_stream.time_base)
            if duration is None and container.duration:
                duration = float(container.duration / av.time_base)
            return AudioMetadata(
                path=str(path.resolve()),
                duration_seconds=duration,
                sample_rate=sample_rate,
                channels=channels,
                format_name=getattr(container.format, "name", None),
                size_bytes=size,
            )
    except Exception:
        pass

    try:
        info = sf.info(str(path))
        return AudioMetadata(
            path=str(path.resolve()),
            duration_seconds=float(info.duration),
            sample_rate=int(info.samplerate),
            channels=int(info.channels),
            format_name=str(info.format),
            size_bytes=size,
        )
    except Exception:
        return AudioMetadata(path=str(path.resolve()), size_bytes=size)
