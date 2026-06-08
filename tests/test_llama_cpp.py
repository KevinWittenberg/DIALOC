from transcript_lok.llama_cpp import LlamaCppServerConfig, build_llama_cpp_command, default_llama_cpp_endpoint
from transcript_lok.ui import _summary_config


def test_build_llama_cpp_command_for_local_gguf(tmp_path):
    server = tmp_path / "llama-server.exe"
    server.write_text("", encoding="utf-8")
    model = tmp_path / "gemma-4-4b-it-Q4_K_M.gguf"
    model.write_text("", encoding="utf-8")

    command = build_llama_cpp_command(
        LlamaCppServerConfig(
            server_path=str(server),
            model_source=str(model),
            model_alias="gemma-4",
            port=8090,
            context_size=4096,
            gpu_layers="0",
        )
    )

    assert command[:3] == [str(server), "-m", str(model)]
    assert command[command.index("--chat-template") + 1] == "gemma"
    assert command[command.index("--port") + 1] == "8090"
    assert command[command.index("-c") + 1] == "4096"
    assert command[command.index("-a") + 1] == "gemma-4"
    assert command[command.index("-ngl") + 1] == "0"


def test_build_llama_cpp_command_for_hugging_face_repo(tmp_path):
    server = tmp_path / "llama-server.exe"
    server.write_text("", encoding="utf-8")

    command = build_llama_cpp_command(
        LlamaCppServerConfig(
            server_path=str(server),
            model_source="bartowski/gemma-4-4b-it-GGUF:Q4_K_M",
            model_alias="gemma-4",
            gpu_layers="cpu",
        )
    )

    assert command[:3] == [str(server), "-hf", "bartowski/gemma-4-4b-it-GGUF:Q4_K_M"]
    assert "-ngl" not in command


def test_summary_config_defaults_for_llama_cpp():
    config = _summary_config("llama.cpp", "", "", "", "Action points", "")

    assert config.provider == "llama.cpp"
    assert config.model == "gemma-4"
    assert config.endpoint == default_llama_cpp_endpoint()
    assert config.template == "Action points"
