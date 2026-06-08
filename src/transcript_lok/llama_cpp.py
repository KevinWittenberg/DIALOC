from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path
import shutil
import subprocess
import time
from collections.abc import Sequence


DEFAULT_LLAMA_CPP_PORT = 8080
DEFAULT_LLAMA_CPP_HOST = "127.0.0.1"
DEFAULT_LLAMA_CPP_MODEL = "gemma-4"
DEFAULT_LLAMA_CPP_CHAT_TEMPLATE = "gemma"
DEFAULT_LLAMA_CPP_CONTEXT_SIZE = 8192
DEFAULT_LLAMA_CPP_GPU_LAYERS = "0"

_LLAMA_SERVER_PROCESS: subprocess.Popen | None = None
_LLAMA_SERVER_LOG_PATH: Path | None = None


@dataclass(frozen=True)
class LlamaCppServerConfig:
    llama_cpp_dir: str | None = None
    server_path: str | None = None
    model_source: str | None = None
    model_alias: str = DEFAULT_LLAMA_CPP_MODEL
    host: str = DEFAULT_LLAMA_CPP_HOST
    port: int = DEFAULT_LLAMA_CPP_PORT
    chat_template: str = DEFAULT_LLAMA_CPP_CHAT_TEMPLATE
    gpu_layers: str | None = DEFAULT_LLAMA_CPP_GPU_LAYERS
    context_size: int = DEFAULT_LLAMA_CPP_CONTEXT_SIZE
    extra_args: Sequence[str] = field(default_factory=tuple)


def default_llama_cpp_dir() -> str:
    configured = os.getenv("LLAMA_CPP_DIR", "").strip()
    if configured:
        return configured

    home = Path.home()
    candidates = [
        home / "local_llms" / "llama_cpp",
        home / "llama.cpp",
        home / "Documents" / "Codex" / "llama.cpp",
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)

    resolved = shutil.which("llama-server") or shutil.which("llama-server.exe")
    return str(Path(resolved).parent) if resolved else ""


def default_llama_cpp_endpoint(port: int = DEFAULT_LLAMA_CPP_PORT, host: str = DEFAULT_LLAMA_CPP_HOST) -> str:
    return f"http://{host}:{int(port)}/v1"


def find_llama_server(llama_cpp_dir: str | None = None, server_path: str | None = None) -> Path:
    if server_path:
        explicit = Path(server_path).expanduser()
        if explicit.exists():
            return explicit
        raise FileNotFoundError(f"llama-server was not found at `{explicit}`.")

    candidates: list[Path] = []
    if llama_cpp_dir:
        root = Path(llama_cpp_dir).expanduser()
        candidates.extend(
            [
                root / "llama-server.exe",
                root / "llama-server",
                root / "build" / "bin" / "Release" / "llama-server.exe",
                root / "build" / "bin" / "Debug" / "llama-server.exe",
                root / "build" / "bin" / "llama-server.exe",
                root / "build" / "bin" / "llama-server",
                root / "build" / "Release" / "llama-server.exe",
                root / "build" / "Debug" / "llama-server.exe",
                root / "bin" / "llama-server.exe",
                root / "bin" / "llama-server",
            ]
        )

    for candidate in candidates:
        if candidate.exists():
            return candidate

    for name in ("llama-server", "llama-server.exe"):
        resolved = shutil.which(name)
        if resolved:
            return Path(resolved)

    hint = f" under `{llama_cpp_dir}`" if llama_cpp_dir else ""
    raise FileNotFoundError(f"llama-server executable was not found{hint}. Build llama.cpp or add llama-server to PATH.")


def build_llama_cpp_command(config: LlamaCppServerConfig) -> list[str]:
    server = find_llama_server(config.llama_cpp_dir, config.server_path)
    model_source = (config.model_source or "").strip().strip('"')
    if not model_source:
        raise ValueError("Set a GGUF model path or Hugging Face GGUF repo before starting llama.cpp.")

    command = [
        str(server),
        *_model_source_args(model_source),
        "--host",
        config.host,
        "--port",
        str(int(config.port)),
        "--chat-template",
        config.chat_template or DEFAULT_LLAMA_CPP_CHAT_TEMPLATE,
        "-c",
        str(int(config.context_size)),
        "-a",
        config.model_alias or DEFAULT_LLAMA_CPP_MODEL,
    ]

    gpu_layers = (config.gpu_layers or "").strip()
    if gpu_layers and gpu_layers.casefold() not in {"none", "cpu"}:
        command.extend(["-ngl", gpu_layers])

    command.extend(str(arg) for arg in config.extra_args if str(arg).strip())
    return command


def start_llama_cpp_server(config: LlamaCppServerConfig, log_dir: Path) -> tuple[str, Path]:
    global _LLAMA_SERVER_LOG_PATH, _LLAMA_SERVER_PROCESS

    if _LLAMA_SERVER_PROCESS and _LLAMA_SERVER_PROCESS.poll() is None:
        endpoint = default_llama_cpp_endpoint(config.port, config.host)
        return f"llama.cpp server is already running at `{endpoint}`.", _LLAMA_SERVER_LOG_PATH or log_dir

    command = build_llama_cpp_command(config)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "llama_cpp_server.log"
    creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    cwd_path = Path(config.llama_cpp_dir).expanduser() if config.llama_cpp_dir else None
    cwd = str(cwd_path) if cwd_path and cwd_path.exists() else None

    log_file = log_path.open("a", encoding="utf-8", errors="replace")
    try:
        log_file.write(f"\n\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] {' '.join(command)}\n")
        log_file.flush()
        _LLAMA_SERVER_PROCESS = subprocess.Popen(
            command,
            cwd=cwd,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            creationflags=creationflags,
        )
    finally:
        log_file.close()

    _LLAMA_SERVER_LOG_PATH = log_path
    time.sleep(0.5)
    if _LLAMA_SERVER_PROCESS.poll() is not None:
        tail = _read_log_tail(log_path)
        raise RuntimeError(f"llama.cpp server exited during startup. See `{log_path}`.\n{tail}")

    endpoint = default_llama_cpp_endpoint(config.port, config.host)
    return f"Started llama.cpp at `{endpoint}`. Model loading can take a moment; run Health check after it settles.", log_path


def stop_llama_cpp_server(timeout_seconds: float = 10.0) -> str:
    global _LLAMA_SERVER_PROCESS

    if not _LLAMA_SERVER_PROCESS:
        return "No app-managed llama.cpp server is running."

    process = _LLAMA_SERVER_PROCESS
    if process.poll() is not None:
        _LLAMA_SERVER_PROCESS = None
        return f"llama.cpp server already exited with code `{process.returncode}`."

    process.terminate()
    try:
        process.wait(timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)
        _LLAMA_SERVER_PROCESS = None
        return "llama.cpp server did not stop gracefully, so it was killed."

    _LLAMA_SERVER_PROCESS = None
    return "Stopped llama.cpp server."


def _model_source_args(model_source: str) -> list[str]:
    if _looks_like_local_model_path(model_source):
        return ["-m", model_source]
    return ["-hf", model_source]


def _looks_like_local_model_path(model_source: str) -> bool:
    if len(model_source) > 1 and model_source[1] == ":":
        return True
    if "\\" in model_source:
        return True
    if model_source.casefold().endswith(".gguf"):
        return True
    return Path(model_source).expanduser().exists()


def _read_log_tail(path: Path, max_chars: int = 2000) -> str:
    if not path.exists():
        return ""
    text = path.read_text(encoding="utf-8", errors="replace")
    return text[-max_chars:]
