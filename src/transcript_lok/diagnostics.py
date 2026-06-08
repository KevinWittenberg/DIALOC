from __future__ import annotations

import ctypes.util
import importlib.metadata
import os
import platform as platform_module
import requests
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .asr import _model_attempts
from .llama_cpp import default_llama_cpp_dir, default_llama_cpp_endpoint, find_llama_server


HF_CACHE = Path.home() / ".cache" / "huggingface" / "hub"


@dataclass(frozen=True)
class RuntimeDiagnostics:
    python: str
    python_executable: str
    os_name: str
    platform: str
    machine: str
    cpu_logical_count: int | None
    effective_cpu_threads: int
    ram_total_bytes: int | None
    packages: dict[str, str]
    cuda_device_count: int | None
    torch_cuda_available: bool
    cublas_available: bool
    cudnn_available: bool
    cuda_runtime_usable: bool
    hf_token_present: bool
    requested_model: str
    requested_device: str
    requested_compute_type: str
    expected_runtime: str
    whisper_model_cached: bool
    diarization_model_cached: bool
    pyav_available: bool
    audio_decode_backend: str
    llama_cpp_dir: str
    llama_cpp_server: str
    llama_cpp_model_source: str
    llama_cpp_model_source_ok: bool
    build_tools: dict[str, str]

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


@dataclass(frozen=True)
class ReadinessItem:
    label: str
    ok: bool
    detail: str


@dataclass(frozen=True)
class RuntimeReadiness:
    diagnostics: RuntimeDiagnostics
    items: list[ReadinessItem]
    recommended_profile: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "diagnostics": self.diagnostics.to_dict(),
            "items": [item.__dict__.copy() for item in self.items],
            "recommended_profile": self.recommended_profile,
        }


def collect_diagnostics(
    model_name: str = "small",
    device: str = "auto",
    compute_type: str = "auto",
    diarization_model: str = "pyannote/speaker-diarization-community-1",
    cpu_threads: int | None = None,
    num_workers: int | None = None,
    llama_cpp_dir: str = "",
    llama_cpp_model_source: str = "",
) -> RuntimeDiagnostics:
    attempts = _model_attempts(device, compute_type)
    expected_device, expected_compute = attempts[0]
    resolved_llama_dir = (llama_cpp_dir or default_llama_cpp_dir()).strip()
    llama_server_path, _llama_server_error = _llama_server_status(resolved_llama_dir)
    model_source_ok, model_source_detail = _model_source_status(llama_cpp_model_source)
    cublas_available = _cublas_available()
    cudnn_available = _cudnn_available()
    pyav_available = _module_available("av")
    return RuntimeDiagnostics(
        python=sys.version.split()[0],
        python_executable=sys.executable,
        os_name=os.name,
        platform=platform_module.platform(),
        machine=platform_module.machine(),
        cpu_logical_count=os.cpu_count(),
        effective_cpu_threads=_effective_cpu_threads(cpu_threads),
        ram_total_bytes=_total_ram_bytes(),
        packages=_package_versions(["faster-whisper", "gradio", "pyannote.audio", "torch", "ctranslate2"]),
        cuda_device_count=_cuda_device_count(),
        torch_cuda_available=_torch_cuda_available(),
        cublas_available=cublas_available,
        cudnn_available=cudnn_available,
        cuda_runtime_usable=expected_device == "cuda",
        hf_token_present=bool(os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")),
        requested_model=model_name,
        requested_device=device,
        requested_compute_type=compute_type,
        expected_runtime=f"{expected_device}/{expected_compute}",
        whisper_model_cached=_hf_model_cached(f"Systran/faster-whisper-{model_name}"),
        diarization_model_cached=_hf_model_cached(diarization_model),
        pyav_available=pyav_available,
        audio_decode_backend="PyAV/faster-whisper" if pyav_available else "soundfile fallback only",
        llama_cpp_dir=resolved_llama_dir,
        llama_cpp_server=llama_server_path,
        llama_cpp_model_source=model_source_detail,
        llama_cpp_model_source_ok=model_source_ok,
        build_tools=_build_tool_status(),
    )


def collect_readiness(
    model_name: str = "small",
    device: str = "auto",
    compute_type: str = "auto",
    diarization_model: str = "pyannote/speaker-diarization-community-1",
    summary_provider: str = "ollama",
    summary_model: str = "llama3.1",
    summary_endpoint: str = "",
    llama_cpp_dir: str = "",
    llama_cpp_model_source: str = "",
    cpu_threads: int | None = None,
    disabled_features: list[str] | None = None,
) -> RuntimeReadiness:
    diagnostics = collect_diagnostics(
        model_name,
        device,
        compute_type,
        diarization_model,
        cpu_threads=cpu_threads,
        llama_cpp_dir=llama_cpp_dir,
        llama_cpp_model_source=llama_cpp_model_source,
    )
    llama_dir = (llama_cpp_dir or default_llama_cpp_dir()).strip()
    llama_server_path, llama_server_error = _llama_server_status(llama_dir)
    llama_endpoint = (summary_endpoint or default_llama_cpp_endpoint()).strip()
    llama_reachable, llama_detail = _local_http_status(f"{llama_endpoint.rstrip('/')}/models")
    ollama_reachable, ollama_detail = _local_http_status("http://localhost:11434/api/tags")
    model_source_ok, model_source_detail = _model_source_status(llama_cpp_model_source)

    items = [
        ReadinessItem("Python 3.10-3.12", diagnostics.python.startswith(("3.10", "3.11", "3.12")), diagnostics.python),
        ReadinessItem("Platform", True, diagnostics.platform),
        ReadinessItem("CPU thread cap", True, f"{diagnostics.effective_cpu_threads} of {diagnostics.cpu_logical_count or 'unknown'} logical CPUs"),
        ReadinessItem("System memory", diagnostics.ram_total_bytes is not None, _format_bytes(diagnostics.ram_total_bytes)),
        ReadinessItem("Audio decode backend", diagnostics.pyav_available, diagnostics.audio_decode_backend),
        ReadinessItem("Whisper runtime", diagnostics.packages.get("faster-whisper") != "not installed", diagnostics.packages.get("faster-whisper", "unknown")),
        ReadinessItem("Expected ASR runtime", True, diagnostics.expected_runtime),
        ReadinessItem("Whisper model cached", diagnostics.whisper_model_cached, f"Systran/faster-whisper-{model_name}"),
        ReadinessItem("CUDA runtime visible", diagnostics.cublas_available and diagnostics.cudnn_available, "cuBLAS and cuDNN visible" if diagnostics.cublas_available and diagnostics.cudnn_available else "CPU fallback expected"),
        ReadinessItem("Torch CUDA available", diagnostics.torch_cuda_available, "torch.cuda.is_available()"),
        ReadinessItem("Diarization token or cache", diagnostics.hf_token_present or diagnostics.diarization_model_cached, "token present or model cached" if diagnostics.hf_token_present or diagnostics.diarization_model_cached else "provide HF token for first download"),
        ReadinessItem("llama.cpp folder", bool(llama_dir and Path(llama_dir).exists()), llama_dir or "not configured"),
        ReadinessItem("llama-server executable", bool(llama_server_path), llama_server_path or llama_server_error),
        ReadinessItem("Gemma/GGUF model source", model_source_ok, model_source_detail),
        ReadinessItem("llama.cpp API reachable", llama_reachable, llama_detail),
        ReadinessItem("Ollama API reachable", ollama_reachable, ollama_detail),
        ReadinessItem("Selected LLM provider", summary_provider in {"ollama", "llama.cpp", "openai-compatible", "openai"}, f"{summary_provider}:{summary_model}"),
    ]
    for label, path in diagnostics.build_tools.items():
        items.append(ReadinessItem(f"{label} on PATH", bool(path), path or "not found"))
    if disabled_features:
        items.append(ReadinessItem("Admin-disabled features", False, ", ".join(sorted(disabled_features))))
    return RuntimeReadiness(diagnostics=diagnostics, items=items, recommended_profile=_recommended_profile(diagnostics))


def diagnostics_markdown(diagnostics: RuntimeDiagnostics) -> str:
    packages = ", ".join(f"`{name}` {version}" for name, version in diagnostics.packages.items())
    cuda_line = (
        f"CUDA devices: `{diagnostics.cuda_device_count}`; "
        f"Torch CUDA: `{_yes_no(diagnostics.torch_cuda_available)}`; "
        f"cuBLAS: `{_yes_no(diagnostics.cublas_available)}`; "
        f"cuDNN: `{_yes_no(diagnostics.cudnn_available)}`"
    )
    build_tools = ", ".join(f"{label}: `{path or 'not found'}`" for label, path in diagnostics.build_tools.items())
    checklist = [
        ("Python 3.10-3.12", diagnostics.python.startswith(("3.10", "3.11", "3.12"))),
        ("PyAV audio decode available", diagnostics.pyav_available),
        ("Whisper model cached", diagnostics.whisper_model_cached),
        ("Hugging Face token present", diagnostics.hf_token_present),
        ("Diarization model cached", diagnostics.diarization_model_cached),
        ("CUDA runtime usable", diagnostics.cuda_runtime_usable),
    ]
    checklist_lines = "\n".join(f"- [{'x' if ok else ' '}] {label}" for label, ok in checklist)
    return (
        "### Runtime Diagnostics\n"
        f"- Python: `{diagnostics.python}`\n"
        f"- Python executable: `{diagnostics.python_executable}`\n"
        f"- Platform: `{diagnostics.platform}` on `{diagnostics.machine or 'unknown'}`\n"
        f"- CPU threads: `{diagnostics.effective_cpu_threads}` effective / `{diagnostics.cpu_logical_count or 'unknown'}` logical\n"
        f"- System memory: `{_format_bytes(diagnostics.ram_total_bytes)}`\n"
        f"- Packages: {packages}\n"
        f"- Requested ASR: `{diagnostics.requested_model}` on `{diagnostics.requested_device}`/"
        f"`{diagnostics.requested_compute_type}`\n"
        f"- Expected runtime: `{diagnostics.expected_runtime}`\n"
        f"- {cuda_line}\n"
        f"- Audio decode: `{diagnostics.audio_decode_backend}`\n"
        f"- Hugging Face token: `{_yes_no(diagnostics.hf_token_present)}`\n"
        f"- llama.cpp folder: `{diagnostics.llama_cpp_dir or 'not configured'}`\n"
        f"- llama.cpp server: `{diagnostics.llama_cpp_server or 'not found'}`\n"
        f"- GGUF/model source: `{diagnostics.llama_cpp_model_source}` (`{'ready' if diagnostics.llama_cpp_model_source_ok else 'needs attention'}`)\n"
        f"- Build tools: {build_tools}\n"
        "\n### First-Run Checklist\n"
        f"{checklist_lines}"
    )


def readiness_markdown(readiness: RuntimeReadiness) -> str:
    lines = [
        "### Runtime Readiness",
        f"- Recommended workflow profile: `{readiness.recommended_profile}`",
        f"- Expected ASR runtime: `{readiness.diagnostics.expected_runtime}`",
        "",
        "### Checks",
    ]
    for item in readiness.items:
        status = "ready" if item.ok else "needs attention"
        lines.append(f"- **{item.label}**: `{status}` - `{_compact_detail(item.detail)}`")
    lines.extend(
        [
            "",
            "### Model Profiles",
            "- `Fast live`: tiny ASR on auto runtime for lowest live latency.",
            "- `Production CPU`: base ASR on cpu/int8 with a 2-thread cap.",
            "- `Balanced file`: small ASR on auto runtime for everyday file transcription.",
            "- `Accurate final`: medium ASR on CPU/int8 for final local passes when CUDA is not ready.",
            "- `Light LLM`: local `llama.cpp` Gemma profile with CPU-first settings.",
            "- `Best local LLM`: local `llama.cpp` Gemma profile with the same endpoint and a larger model alias when available.",
        ]
    )
    return "\n".join(lines)


def _package_versions(names: list[str]) -> dict[str, str]:
    versions: dict[str, str] = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not installed"
    return versions


def _module_available(name: str) -> bool:
    try:
        __import__(name)
        return True
    except Exception:
        return False


def _torch_cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _cuda_device_count() -> int | None:
    try:
        import ctranslate2

        return int(ctranslate2.get_cuda_device_count())
    except Exception:
        return None


def _cublas_available() -> bool:
    return bool(
        ctypes.util.find_library("cublas")
        or ctypes.util.find_library("cublas64_12")
        or shutil.which("cublas64_12.dll")
    )


def _cudnn_available() -> bool:
    return bool(
        ctypes.util.find_library("cudnn")
        or ctypes.util.find_library("cudnn64_9")
        or ctypes.util.find_library("cudnn_ops64_9")
        or shutil.which("cudnn64_9.dll")
        or shutil.which("cudnn_ops64_9.dll")
    )


def _effective_cpu_threads(cpu_threads: int | None) -> int:
    configured = cpu_threads or os.getenv("TRANSCRIPT_LOK_CPU_THREADS") or os.getenv("OMP_NUM_THREADS")
    try:
        value = int(configured) if configured else 0
    except (TypeError, ValueError):
        value = 0
    logical = os.cpu_count() or 1
    return max(1, min(value or logical, logical))


def _total_ram_bytes() -> int | None:
    if hasattr(os, "sysconf"):
        try:
            page_size = int(os.sysconf("SC_PAGE_SIZE"))
            pages = int(os.sysconf("SC_PHYS_PAGES"))
            if page_size > 0 and pages > 0:
                return page_size * pages
        except (OSError, ValueError):
            pass

    if os.name == "nt":
        try:
            class MemoryStatus(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            status = MemoryStatus()
            status.dwLength = ctypes.sizeof(MemoryStatus)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):  # type: ignore[attr-defined]
                return int(status.ullTotalPhys)
        except Exception:
            return None
    return None


def _build_tool_status() -> dict[str, str]:
    if os.name == "nt":
        return {
            "CMake": shutil.which("cmake") or "",
            "MSVC cl": shutil.which("cl") or "",
        }
    compiler = shutil.which("c++") or shutil.which("g++") or shutil.which("clang++") or shutil.which("gcc") or ""
    return {
        "CMake": shutil.which("cmake") or "",
        "C++ compiler": compiler,
    }


def _hf_model_cached(model_id: str) -> bool:
    if Path(model_id).exists():
        return True
    normalized = "models--" + model_id.replace("/", "--")
    return (HF_CACHE / normalized).exists()


def _yes_no(value: bool) -> str:
    return "yes" if value else "no"


def _llama_server_status(llama_cpp_dir: str) -> tuple[str, str]:
    try:
        return str(find_llama_server(llama_cpp_dir or None)), ""
    except Exception as exc:
        return "", str(exc)


def _local_http_status(url: str) -> tuple[bool, str]:
    try:
        response = requests.get(url, timeout=1.5)
        response.raise_for_status()
        return True, url
    except Exception as exc:
        return False, f"{url} ({exc})"


def _model_source_status(model_source: str) -> tuple[bool, str]:
    source = (model_source or os.getenv("LLAMA_CPP_MODEL_SOURCE") or "").strip().strip('"')
    if not source:
        return False, "not configured"
    if len(source) > 1 and source[1] == ":":
        return Path(source).exists(), source
    if "\\" in source or source.casefold().endswith(".gguf"):
        return Path(source).exists(), source
    return True, source


def _recommended_profile(diagnostics: RuntimeDiagnostics) -> str:
    if diagnostics.expected_runtime.startswith("cuda"):
        return "Accurate final"
    if diagnostics.effective_cpu_threads <= 2:
        return "Production CPU"
    if diagnostics.whisper_model_cached:
        return "Balanced file"
    return "Fast live"


def _escape_table(value: str) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def _compact_detail(value: str, limit: int = 140) -> str:
    text = str(value).replace("|", "\\|").replace("\n", " ").strip()
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "..."


def _format_bytes(value: int | None) -> str:
    if value is None:
        return "unknown"
    size = float(value)
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size < 1024 or unit == "TB":
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"
