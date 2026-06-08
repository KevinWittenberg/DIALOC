# AGENTS.md

Guidance for AI agents working on Transcript LOK.

## Project Overview

Transcript LOK is a local-first Python transcription app. It uses:

- Gradio for the local UI.
- `faster-whisper` / CTranslate2 for Dutch and English ASR.
- `pyannote.audio` for speaker diarization.
- Optional Ollama, `llama.cpp`/Gemma, OpenAI, or OpenAI-compatible endpoints for LLM workflows.

The app supports file transcription, live microphone transcription, runtime readiness checks, workflow model profiles, diarization, speaker renaming, glossary prompting, transcript correction, LLM auto-correction, summaries, action lists, anonymous decision summaries, follow-up emails, session save/load, and exports.

## Repository Map

- `src/transcript_lok/ui.py`: Gradio UI and callback wiring.
- `src/transcript_lok/asr.py`: Whisper model loading, transcription, progress, chunking, CUDA fallback.
- `src/transcript_lok/diarization.py`: pyannote diarization and speaker assignment.
- `src/transcript_lok/models.py`: transcript data structures and export formats.
- `src/transcript_lok/audio.py`: audio conversion, metadata, WAV helpers.
- `src/transcript_lok/llm.py`: LLM providers, task prompts, correction parsing, health checks.
- `src/transcript_lok/llama_cpp.py`: `llama.cpp` server discovery and lifecycle helpers.
- `src/transcript_lok/diagnostics.py`: runtime diagnostics and first-run checklist.
- `src/transcript_lok/sessions.py`: session save/load.
- `scripts/doctor.py`: environment diagnostics.
- `scripts/benchmark.py`: repeatable model benchmark.
- `scripts/smoke_transcribe.py`: quick ASR smoke test.
- `scripts/real_llm_smoke.py`: gated real local LLM/Gemma smoke test.
- `tests/`: unit and workflow tests.
- `outputs/`: generated files and fixtures. Most outputs are ignored; `outputs/short_speech_10s.wav` is a useful smoke-test fixture.

## Setup And Commands

Use Python 3.10.

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\python -m pip install --upgrade pip
.\.venv\Scripts\python -m pip install -r requirements.txt
```

Run the app:

```powershell
.\.venv\Scripts\python -m transcript_lok.ui
```

Open:

```text
http://127.0.0.1:7860
```

Core verification:

```powershell
.\.venv\Scripts\python -m pytest
.\.venv\Scripts\python -m compileall -q src scripts tests
.\.venv\Scripts\python scripts\doctor.py
.\.venv\Scripts\python scripts\smoke_transcribe.py outputs\short_speech_10s.wav --model small --language en
```

Benchmark:

```powershell
.\.venv\Scripts\python scripts\benchmark.py --models tiny,base,small
.\.venv\Scripts\python scripts\benchmark.py --models tiny --with-llm --llm-task meeting_summary
```

## Working Conventions

- Keep changes local-first. Do not make cloud calls unless the user explicitly configures and triggers a cloud summary provider.
- Do not run real LLM/Gemma tests unless the user explicitly asks or sets `LLM_REAL_TEST=1`; mocked LLM tests should remain the default.
- Prefer small, testable changes over rewrites.
- Preserve Gradio unless the user explicitly asks for a frontend migration.
- Use `apply_patch` for manual edits.
- Use `rg` / `rg --files` for searches when available.
- Do not delete generated user outputs unless they are clearly temporary files created during the current task.
- Avoid committing model files, caches, logs, or generated transcripts.
- Keep UI dense and utilitarian. This is an operational tool, not a landing page.

## Important Runtime Notes

- On this machine, CTranslate2 may detect a CUDA device while CUDA runtime DLLs such as `cublas64_12.dll` are not on `PATH`. The app intentionally falls back to CPU/int8 in `auto` mode.
- pyannote may warn about TorchCodec/FFmpeg on Windows. The app passes audio to pyannote as an in-memory waveform, so that warning is not usually a blocker.
- pyannote Community-1 requires accepting Hugging Face model terms and providing a token for first download.
- `large-v3` can be very slow on CPU. The UI defaults to `small` for responsiveness.
- File transcription progress is based on emitted Whisper segment timestamps, so progress is intentionally coarse.
- Cancellation is graceful at segment or chunk boundaries, not instantaneous.
- `llama.cpp`/Gemma uses an OpenAI-compatible local server at `http://127.0.0.1:8080/v1` by default. Gemma prompts should not rely on a separate system role; fold system-level instructions into the user prompt for `llama.cpp`.
- LLM correction auto-applies only validated segment text changes. It must not change timestamps, speakers, confidence, audio path, or diarization metadata. If correction JSON is malformed, apply no transcript changes.
- The glossary doubles as the custom LLM wordlist for correction passes and should be treated as canonical spelling for names, acronyms, products, and domain terms.

## Testing Guidance

When changing ASR or audio behavior:

- Run the smoke transcription command with `outputs\short_speech_10s.wav`.
- Check that timestamps remain monotonic and absolute.
- Check CPU/int8 fallback still works when CUDA DLLs are unavailable.

When changing UI callbacks:

- Import and build the app:

```powershell
@'
from transcript_lok.ui import build_app
app = build_app()
print(type(app).__name__, len(app.blocks))
'@ | .\.venv\Scripts\python -
```

- Run callback-level smoke tests when possible before browser checks.
- Restart the local Gradio server after code changes before browser verification.

When changing exports or sessions:

- Verify JSON round trips through `TranscriptDocument`.
- Verify CSV includes `start,end,speaker,text,confidence`.
- Verify DOCX starts with ZIP bytes (`PK`) and opens in Word-compatible tools.
- Verify `llm_outputs` round trips while preserving the legacy `summary` field.

When changing LLM workflows:

- Run mocked LLM workflow tests with regular `pytest`; do not require a real local model for normal verification.
- Check that correction failures do not mutate transcript text.
- Check provider-aware prompt construction, especially the `llama.cpp`/Gemma single-user-message path.
- For explicit real local model verification:

```powershell
$env:LLM_REAL_TEST = "1"
$env:LLM_REAL_ENDPOINT = "http://127.0.0.1:8080/v1"
$env:LLM_REAL_MODEL = "gemma-4"
.\.venv\Scripts\python -m pytest tests\test_real_gemma_llm.py
.\.venv\Scripts\python scripts\real_llm_smoke.py --force
```

## Common Pitfalls

- Do not assume `auto` means CUDA will be usable. Check diagnostics.
- Do not use a raw file path for pyannote input unless TorchCodec/FFmpeg is known-good; prefer the existing waveform path.
- Do not remove `outputs/short_speech_10s.wav`; tests and benchmarks rely on it.
- Do not add heavy dependencies for simple document/export features unless there is a clear need.
- Gradio APIs shift between versions. Smoke-test `build_app()` after UI changes.
- Runtime readiness and workflow profiles live in the Runtime tab. Keep them aligned with README setup guidance and release checklist checks.
- Settings are folded into the right-edge sidebar by default. Keep searchable settings targets and scroll behavior working when moving controls.
- LLM artifact outputs are launched from Artifacts tab tile buttons; do not reintroduce a settings-side output template selector.
- Live transcription stores commit history and last autosave path in document metadata. Preserve those recovery fields when changing live callbacks.
- The benchmark script should stay usable without a real LLM by default. Optional LLM and diarization timings must remain opt-in.
- LLM artifacts default to Dutch. Keep the `Artifact language` selector, prompts, provenance, and tests in sync when changing artifact workflows.
- Live transcription is intentionally literal. Do not add LLM rewriting to the live path; use the explicit post-meeting correction pass.

## Release Checklist

Before handing off a significant change:

1. Run tests and compile checks.
2. Run `scripts\doctor.py`.
3. Run `scripts\smoke_transcribe.py` on the short fixture.
4. Launch the app and verify the main UI renders.
5. For UI changes, use the browser to check the relevant screen.
6. Update `README.md` or `docs\RELEASE_CHECKLIST.md` if user-facing behavior changed.
