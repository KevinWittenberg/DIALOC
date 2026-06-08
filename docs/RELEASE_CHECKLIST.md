# Release Checklist

- Create a fresh Python 3.10 virtual environment.
- Install with `.\.venv\Scripts\python -m pip install -r requirements.txt`.
- Run `.\.venv\Scripts\python scripts\doctor.py`.
- Run `.\.venv\Scripts\python scripts\smoke_transcribe.py outputs\short_speech_10s.wav --model small --language en`.
- Run `.\.venv\Scripts\python scripts\benchmark.py --models tiny,base,small`.
- For artifact timing, run `.\.venv\Scripts\python scripts\benchmark.py --models tiny --with-llm --llm-task meeting_summary` when a local LLM endpoint is available.
- For real Gemma, run `LLM_REAL_TEST=1` tests only when `llama-server` and a Gemma GGUF are available.
- Launch `.\.venv\Scripts\python -m transcript_lok.ui`.
- Verify the Runtime tab readiness panel, compact runtime cards, folded searchable settings sidebar, and workflow profiles.
- Verify file transcription, progress, cancellation status, correction edits, speaker renaming, export, session save/load, and LLM provider health check.
- Verify Artifacts: Dutch default output, English override, tile buttons for correction/summary/action/decision/follow-up outputs, correction pass audit, provenance table, and LLM artifact export.
- Verify live transcription status: pause/resume button, recording time, committed segments, autosave path, discard/flush behavior, and recovery by loading the live autosave session.
- For UI changes, open the app in a browser and check Runtime, File, Live, Transcript, and Artifacts screens.
