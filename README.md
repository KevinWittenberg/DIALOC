# Transcript LOK

Local Dutch and English transcription pipeline with:

- live microphone transcription
- audio file transcription
- runtime diagnostics and first-run checklist
- runtime readiness view with local model profiles
- audio metadata preflight
- optional chunked transcription for long files
- speaker diarization
- timestamped and speaker-annotated transcripts
- custom glossary prompting
- editable transcript corrections
- search/filter, merge/split, confidence indicators, and speaker-name table
- speaker label replacement
- session save/load
- local/API LLM workflows via Ollama, `llama.cpp`/Gemma, OpenAI, or OpenAI-compatible endpoints
- LLM artifact exports with provenance and correction audit data
- Markdown, JSON, SRT, VTT, CSV, and DOCX exports

## Setup

Use Python 3.10.

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\python -m pip install --upgrade pip
.\.venv\Scripts\python -m pip install -r requirements.txt
```

## Run

```powershell
.\.venv\Scripts\python -m transcript_lok.ui
```

Then open http://127.0.0.1:7860.

The selected Whisper model downloads on first use and is then cached locally.
The UI defaults to `small` because it is responsive on CPU. Use `large-v3` for final accuracy when you have time or a working CUDA runtime.

If `Device` is set to `auto`, the app only uses CUDA when the required CTranslate2 CUDA DLLs are visible. On this machine it falls back to CPU/int8 because `cublas64_12.dll` is not currently on `PATH`.

During file transcription, the status field shows a lightweight progress bar based on Whisper segment timestamps. Updates are throttled to coarse progress buckets so the UI stays responsive without adding meaningful compute overhead.

For long recordings, enable `Chunk long files` in the ASR panel. Chunking uses overlapping windows and preserves absolute timestamps.

## Diarization

The default diarization model is `pyannote/speaker-diarization-community-1`.

For first use, accept the model terms on Hugging Face and provide a token in the UI or set:

```powershell
$env:HUGGINGFACE_TOKEN = "hf_..."
```

To pre-download models:

```powershell
.\.venv\Scripts\python scripts\download_models.py --whisper-model large-v3
```

To pre-download Whisper only:

```powershell
.\.venv\Scripts\python scripts\download_models.py --whisper-model large-v3 --skip-diarization
```

After the model files are cached, transcription and diarization inference run locally.

On Windows, pyannote may warn that TorchCodec/FFmpeg is unavailable. This app passes audio to pyannote as an in-memory waveform, so that optional file-decoder warning is not a blocker for diarization.

## Runtime Readiness And UI

Open the Runtime tab before a first run or after changing local models. The app uses a meeting workbench with live meeting controls on the left, Runtime/File/Live/Transcript/Artifacts work in the center, and a folded searchable settings sidebar opened from the right-edge settings control. Runtime checks cover the selected Whisper runtime, Hugging Face/diarization readiness, `llama.cpp` paths, Gemma/GGUF model source, local API reachability, and build tools. The workflow profiles set sensible ASR and LLM defaults:

- `Fast live`: lowest-latency live capture.
- `Production CPU`: two-thread CPU profile for Linux/OpenOnDemand deployments.
- `Balanced file`: everyday local file transcription and artifacts.
- `Accurate final`: slower final pass on CPU/int8 when CUDA is not ready.
- `Light LLM`: local Gemma/GGUF artifact generation.
- `Best local LLM`: prefer the strongest local Gemma alias available on `llama.cpp`.

Admins can hide UI actions and set production CPU defaults with a local policy file instead of editing source. Copy `config/transcript_lok_policy.example.json`, set `TRANSCRIPT_LOK_POLICY` to that path, and adjust feature flags or thread limits. See `docs/OOD_DEPLOYMENT.md` for Linux/OpenOnDemand launcher settings.

## Meeting Artifacts

Ollama is the default local summary provider. Start Ollama separately and choose a local model already installed on your machine.

`llama.cpp` is also available as a local summary provider for GGUF models such as Gemma 4. The Summary panel can either talk to an already running `llama-server` at `http://127.0.0.1:8080/v1`, or start one for you from a local llama.cpp checkout.

Build `llama-server` on Windows from your llama.cpp folder:

```powershell
cd C:\Users\Kevin\Documents\Codex\llama.cpp
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release --target llama-server
```

This requires CMake and a C++ build toolchain such as Visual Studio Build Tools on `PATH`.

Download a Gemma 4 GGUF, for example the Q4_K_M 4B instruct model:

```powershell
.\.venv\Scripts\python -m pip install huggingface_hub
.\.venv\Scripts\huggingface-cli download bartowski/gemma-4-4b-it-GGUF --include "*Q4_K_M*" --local-dir C:\Users\Kevin\Documents\Codex\llama.cpp\models\gemma-4-4b-it-GGUF
```

In the Summary panel, choose provider `llama.cpp`, set model `gemma-4`, enter the GGUF path or an explicit Hugging Face repo such as `bartowski/gemma-4-4b-it-GGUF:Q4_K_M`, then press `Start llama.cpp`. The app starts `llama-server` with `--chat-template gemma`, because Gemma 4 needs the Gemma chat template. GPU layers default to `0` for CPU-first startup; set `99` or `all` when you have a working GPU build and enough VRAM.

Manual server startup works too:

```powershell
C:\Users\Kevin\Documents\Codex\llama.cpp\build\bin\Release\llama-server.exe `
  -m C:\Users\Kevin\Documents\Codex\llama.cpp\models\gemma-4-4b-it-GGUF\gemma-4-4b-it-Q4_K_M.gguf `
  --chat-template gemma `
  --port 8080 `
  -c 8192 `
  -ngl 0 `
  -a gemma-4
```

OpenAI and OpenAI-compatible endpoints are optional. The app only sends transcript text to a cloud endpoint when that provider is selected and the summary button is pressed.

For OpenAI:

```powershell
$env:OPENAI_API_KEY = "sk-..."
```

For a local OpenAI-compatible server such as LM Studio:

```powershell
$env:OPENAI_BASE_URL = "http://localhost:1234/v1"
```

The Artifacts tab exposes each LLM workflow as its own tile-style action:

- auto-applying an LLM correction pass over likely ASR typos while preserving an audit trail
- meeting summaries
- action-point tables with what, who, deadline, and evidence
- internal and external anonymous decision summaries
- follow-up emails

Artifact output defaults to Dutch. Use `Artifact language` in the folded settings sidebar to switch summaries, action lists, anonymous decision summaries, and follow-up emails to English. The correction pass does not translate; it only fixes likely ASR mistakes while preserving the transcript language and meaning.

The correction pass uses the glossary as the canonical wordlist for names, acronyms, products, and domain terms. If the LLM response cannot be parsed as structured corrections, no transcript changes are applied.
Generated LLM outputs are stored in session JSON under `llm_outputs`, shown with provenance, and can be exported as a Markdown/JSON bundle plus output-specific Markdown files.

The File tab supports `Raw` and `Corrected` transcript modes. `Corrected` first stores the raw ASR transcript, then applies a local-only cleanup pass through `llama.cpp`, Ollama, or a localhost OpenAI-compatible endpoint. Successful runs store both raw and corrected JSON/Markdown variants with hashes in transcript metadata.

Real local Gemma testing is gated so ordinary tests stay fast:

```powershell
$env:LLM_REAL_TEST = "1"
$env:LLM_REAL_ENDPOINT = "http://127.0.0.1:8080/v1"
$env:LLM_REAL_MODEL = "gemma-4"
.\.venv\Scripts\python -m pytest tests\test_real_gemma_llm.py
.\.venv\Scripts\python scripts\real_llm_smoke.py --force
```

To include the ASR fixture in the smoke flow:

```powershell
.\.venv\Scripts\python scripts\real_llm_smoke.py --force --fixture outputs\short_speech_10s.wav
```

## Diagnostics and Benchmarks

```powershell
.\.venv\Scripts\python scripts\doctor.py
.\.venv\Scripts\python scripts\benchmark.py --models tiny,base,small
```

Optional broader timing loops:

```powershell
.\.venv\Scripts\python scripts\benchmark.py --models tiny --with-llm --llm-task meeting_summary
.\.venv\Scripts\python scripts\benchmark.py --models small --with-diarization --jsonl outputs\benchmarks.jsonl
```

See `docs\RELEASE_CHECKLIST.md` for the smoke-test flow.

## Glossary

Put names, domain terms, product names, and acronyms in the glossary box. Use one term per line, or separate terms with commas/semicolons.

## Speaker Names

After diarization, replace labels with names in the corrections tab:

```text
SPEAKER_00 = Kevin
SPEAKER_01 = Anna
```
