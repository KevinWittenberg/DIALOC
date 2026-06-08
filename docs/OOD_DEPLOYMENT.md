# Linux/OpenOnDemand Deployment Notes

Transcript LOK remains a local-first Gradio app. For OpenOnDemand, launch it inside the allocated VM or job session and let OOD proxy the local Gradio port.

## Production CPU Policy

Use a local policy file rather than editing UI code:

```bash
export TRANSCRIPT_LOK_POLICY="$PWD/config/transcript_lok_policy.json"
cp config/transcript_lok_policy.example.json "$TRANSCRIPT_LOK_POLICY"
```

The example policy is conservative for a 2-thread CPU VM:

- ASR: `base`, `cpu/int8`, `cpu_threads=2`, `num_workers=1`, `beam_size=1`.
- Chunking enabled for long files.
- Diarization, live capture, external decisions, follow-up email, and cloud providers disabled by default.
- Automatic correction remains local-only.

## OOD Launcher Environment

Set these in the OOD app launcher or job script:

```bash
export TRANSCRIPT_LOK_HOST="${TRANSCRIPT_LOK_HOST:-127.0.0.1}"
export TRANSCRIPT_LOK_PORT="${TRANSCRIPT_LOK_PORT:-7860}"
export TRANSCRIPT_LOK_ROOT_PATH="${TRANSCRIPT_LOK_ROOT_PATH:-}"
export TRANSCRIPT_LOK_OUTPUT_DIR="${TRANSCRIPT_LOK_OUTPUT_DIR:-$HOME/transcript_lok_outputs}"
export TRANSCRIPT_LOK_PRODUCTION_CPU=1
export TRANSCRIPT_LOK_CPU_THREADS=2
export TRANSCRIPT_LOK_QUEUE_MAX_THREADS=2
export OMP_NUM_THREADS=2
export CT2_USE_EXPERIMENTAL_PACKED_GEMM=1
```

Then launch:

```bash
python -m transcript_lok.ui
```

## Local LLM Correction

For lightweight cleanup, prefer a tiny local `llama.cpp` server before enabling larger models:

```bash
export LLAMA_CPP_DIR="$HOME/local_llms/llama_cpp"
export LLAMA_CPP_MODEL_SOURCE="ggml-org/gemma-3-1b-it-GGUF"
export LLAMA_CPP_BASE_URL="http://127.0.0.1:8080/v1"
```

The app also checks `~/local_llms/llama_cpp`, `~/llama.cpp`, Windows defaults, and `PATH` for `llama-server`.

## Verification

Run these before handoff:

```bash
python -m pytest
python -m compileall -q src scripts tests
python scripts/doctor.py --model base --device cpu --compute-type int8
python scripts/smoke_transcribe.py outputs/short_speech_10s.wav --model base --language en --device cpu --compute-type int8
```

Do not run real Gemma/LLM tests unless a local model is approved and `LLM_REAL_TEST=1` is set.
