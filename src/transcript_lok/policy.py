from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_POLICY_PATH = ROOT / "config" / "transcript_lok_policy.json"


FEATURE_KEYS = {
    "file_transcription",
    "live_transcription",
    "diarization",
    "correction",
    "meeting_summary",
    "action_points",
    "decision_summaries",
    "follow_up_email",
    "cloud_providers",
}


@dataclass(frozen=True)
class RuntimePolicy:
    production_cpu: bool = False
    default_model: str = "small"
    default_device: str = "auto"
    default_compute_type: str = "auto"
    cpu_threads: int = 0
    num_workers: int = 1
    beam_size: int = 5
    chunk_long_files: bool = False
    chunk_seconds: int = 600
    overlap_seconds: int = 5
    queue_max_threads: int = 40


@dataclass(frozen=True)
class TranscriptLokPolicy:
    features: dict[str, bool] = field(default_factory=lambda: {key: True for key in FEATURE_KEYS})
    runtime: RuntimePolicy = field(default_factory=RuntimePolicy)
    automatic_correction_local_only: bool = True
    source_path: str = ""

    def feature_enabled(self, name: str) -> bool:
        return bool(self.features.get(name, True))

    def disabled_features(self) -> list[str]:
        return sorted(name for name, enabled in self.features.items() if not enabled)


def load_policy(path: str | Path | None = None) -> TranscriptLokPolicy:
    policy_path = _policy_path(path)
    data = _read_policy_data(policy_path)
    policy = _policy_from_dict(data, source_path=str(policy_path) if policy_path and policy_path.exists() else "")
    return _policy_with_env_overrides(policy)


def _policy_path(path: str | Path | None) -> Path | None:
    explicit = path or os.getenv("TRANSCRIPT_LOK_POLICY", "").strip()
    if explicit:
        return Path(explicit).expanduser()
    return DEFAULT_POLICY_PATH if DEFAULT_POLICY_PATH.exists() else None


def _read_policy_data(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    if path.suffix.casefold() == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    if path.suffix.casefold() == ".toml":
        try:
            import tomllib  # type: ignore[import-not-found]
        except ModuleNotFoundError as exc:
            raise RuntimeError("TOML policies require Python 3.11+ or use a JSON policy file.") from exc
        return tomllib.loads(path.read_text(encoding="utf-8"))
    raise ValueError(f"Unsupported policy file extension: {path.suffix or '(none)'}")


def _policy_from_dict(data: dict[str, Any], *, source_path: str = "") -> TranscriptLokPolicy:
    raw_features = data.get("features") if isinstance(data.get("features"), dict) else {}
    features = {key: _bool(raw_features.get(key), True) for key in FEATURE_KEYS}

    raw_runtime = data.get("runtime") if isinstance(data.get("runtime"), dict) else {}
    production_cpu = _bool(raw_runtime.get("production_cpu"), False)
    runtime = RuntimePolicy(
        production_cpu=production_cpu,
        default_model=str(raw_runtime.get("default_model") or ("base" if production_cpu else "small")),
        default_device=str(raw_runtime.get("default_device") or ("cpu" if production_cpu else "auto")),
        default_compute_type=str(raw_runtime.get("default_compute_type") or ("int8" if production_cpu else "auto")),
        cpu_threads=_int(raw_runtime.get("cpu_threads"), 2 if production_cpu else 0),
        num_workers=_int(raw_runtime.get("num_workers"), 1),
        beam_size=_int(raw_runtime.get("beam_size"), 1 if production_cpu else 5),
        chunk_long_files=_bool(raw_runtime.get("chunk_long_files"), production_cpu),
        chunk_seconds=_int(raw_runtime.get("chunk_seconds"), 600),
        overlap_seconds=_int(raw_runtime.get("overlap_seconds"), 5),
        queue_max_threads=_int(raw_runtime.get("queue_max_threads"), 2 if production_cpu else 40),
    )

    return TranscriptLokPolicy(
        features=features,
        runtime=runtime,
        automatic_correction_local_only=_bool(data.get("automatic_correction_local_only"), True),
        source_path=source_path,
    )


def _policy_with_env_overrides(policy: TranscriptLokPolicy) -> TranscriptLokPolicy:
    production_cpu = _bool(os.getenv("TRANSCRIPT_LOK_PRODUCTION_CPU"), policy.runtime.production_cpu)
    runtime = RuntimePolicy(
        production_cpu=production_cpu,
        default_model=os.getenv("TRANSCRIPT_LOK_DEFAULT_MODEL", policy.runtime.default_model if not production_cpu else "base"),
        default_device=os.getenv("TRANSCRIPT_LOK_DEFAULT_DEVICE", policy.runtime.default_device if not production_cpu else "cpu"),
        default_compute_type=os.getenv(
            "TRANSCRIPT_LOK_DEFAULT_COMPUTE_TYPE",
            policy.runtime.default_compute_type if not production_cpu else "int8",
        ),
        cpu_threads=_int(os.getenv("TRANSCRIPT_LOK_CPU_THREADS"), policy.runtime.cpu_threads if not production_cpu else 2),
        num_workers=_int(os.getenv("TRANSCRIPT_LOK_NUM_WORKERS"), policy.runtime.num_workers),
        beam_size=_int(os.getenv("TRANSCRIPT_LOK_BEAM_SIZE"), policy.runtime.beam_size if not production_cpu else 1),
        chunk_long_files=_bool(os.getenv("TRANSCRIPT_LOK_CHUNK_LONG_FILES"), policy.runtime.chunk_long_files or production_cpu),
        chunk_seconds=_int(os.getenv("TRANSCRIPT_LOK_CHUNK_SECONDS"), policy.runtime.chunk_seconds),
        overlap_seconds=_int(os.getenv("TRANSCRIPT_LOK_OVERLAP_SECONDS"), policy.runtime.overlap_seconds),
        queue_max_threads=_int(os.getenv("TRANSCRIPT_LOK_QUEUE_MAX_THREADS"), policy.runtime.queue_max_threads if not production_cpu else 2),
    )
    return TranscriptLokPolicy(
        features=policy.features,
        runtime=runtime,
        automatic_correction_local_only=_bool(
            os.getenv("TRANSCRIPT_LOK_CORRECTION_LOCAL_ONLY"),
            policy.automatic_correction_local_only,
        ),
        source_path=policy.source_path,
    )


def _bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    text = str(value).strip().casefold()
    if text in {"1", "true", "yes", "on", "enabled"}:
        return True
    if text in {"0", "false", "no", "off", "disabled"}:
        return False
    return default


def _int(value: Any, default: int) -> int:
    if value is None or value == "":
        return default
    try:
        return max(0, int(float(value)))
    except (TypeError, ValueError):
        return default
