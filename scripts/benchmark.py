from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any

from transcript_lok.audio import audio_metadata
from transcript_lok.asr import ASRConfig, WhisperTranscriber
from transcript_lok.diarization import DEFAULT_DIARIZATION_MODEL, DiarizationConfig, SpeakerDiarizer
from transcript_lok.llm import SummaryConfig, correct_transcript, run_llm_task


LLM_TASKS = [
    "correction_pass",
    "meeting_summary",
    "action_points",
    "decision_summary_internal_anonymous",
    "decision_summary_external_anonymous",
    "follow_up_email",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark local ASR, diarization, and LLM artifact workflows.")
    parser.add_argument("--audio", default="outputs/short_speech_10s.wav")
    parser.add_argument("--models", default="tiny,base,small")
    parser.add_argument("--language", default="en")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--compute-type", default="auto")
    parser.add_argument("--with-diarization", action="store_true", help="Include pyannote diarization timing.")
    parser.add_argument("--diarization-model", default=DEFAULT_DIARIZATION_MODEL)
    parser.add_argument("--hf-token", default=os.getenv("HUGGINGFACE_TOKEN") or os.getenv("HF_TOKEN") or "")
    parser.add_argument("--with-llm", action="store_true", help="Include one LLM workflow timing.")
    parser.add_argument("--llm-provider", default=os.getenv("LLM_REAL_PROVIDER", "llama.cpp"))
    parser.add_argument("--llm-model", default=os.getenv("LLM_REAL_MODEL", "gemma-4"))
    parser.add_argument("--llm-endpoint", default=os.getenv("LLM_REAL_ENDPOINT", "http://127.0.0.1:8080/v1"))
    parser.add_argument("--llm-api-key", default=os.getenv("OPENAI_API_KEY", ""))
    parser.add_argument("--llm-task", default="meeting_summary", choices=LLM_TASKS)
    parser.add_argument("--glossary", default="")
    parser.add_argument("--jsonl", default="", help="Optional path to append run metadata as JSON Lines.")
    args = parser.parse_args()

    audio_path = Path(args.audio).resolve()
    if not audio_path.exists():
        raise SystemExit(f"Audio fixture does not exist: {audio_path}")

    metadata = audio_metadata(str(audio_path))
    rows = []
    for model in [item.strip() for item in args.models.split(",") if item.strip()]:
        rows.append(_benchmark_model(model, audio_path, metadata.duration_seconds, args))

    writer = csv.DictWriter(sys.stdout, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)

    if args.jsonl:
        jsonl_path = Path(args.jsonl)
        jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        with jsonl_path.open("a", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _benchmark_model(model: str, audio_path: Path, audio_seconds: float | None, args: argparse.Namespace) -> dict[str, Any]:
    row: dict[str, Any] = {
        "run_id": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "audio": str(audio_path),
        "audio_seconds": _round(audio_seconds),
        "model": model,
        "segments": 0,
        "load_seconds": "",
        "transcribe_seconds": "",
        "diarize_seconds": "",
        "llm_seconds": "",
        "total_seconds": "",
        "runtime": "",
        "chunking": "single",
        "llm_provider": args.llm_provider if args.with_llm else "",
        "llm_model": args.llm_model if args.with_llm else "",
        "llm_task": args.llm_task if args.with_llm else "",
        "llm_output_chars": "",
        "correction_count": "",
        "status": "ok",
    }
    total_start = perf_counter()
    try:
        load_start = perf_counter()
        transcriber = WhisperTranscriber(
            ASRConfig(
                model_size=model,
                language=args.language or None,
                device=args.device,
                compute_type=args.compute_type,
            )
        )
        loaded = perf_counter()
        document = transcriber.transcribe_file(str(audio_path), args.glossary)
        transcribed = perf_counter()

        row.update(
            {
                "segments": len(document.segments),
                "load_seconds": _round(loaded - load_start),
                "transcribe_seconds": _round(transcribed - loaded),
                "runtime": f"{document.metadata.get('asr_device')}/{document.metadata.get('asr_compute_type')}",
            }
        )

        if args.with_diarization:
            diarize_start = perf_counter()
            diarizer = SpeakerDiarizer(
                DiarizationConfig(
                    model_name_or_path=args.diarization_model,
                    hf_token=args.hf_token or None,
                    device=args.device,
                )
            )
            document = diarizer.apply(str(audio_path), document)
            row["diarize_seconds"] = _round(perf_counter() - diarize_start)

        if args.with_llm:
            llm_start = perf_counter()
            config = SummaryConfig(
                provider=args.llm_provider,
                model=args.llm_model,
                endpoint=args.llm_endpoint or None,
                api_key=args.llm_api_key or None,
                max_chars_per_call=12_000,
            )
            if args.llm_task == "correction_pass":
                result = correct_transcript(document, config, args.glossary)
                row["llm_output_chars"] = len(result.raw_response)
                row["correction_count"] = len(result.changes)
            else:
                output = run_llm_task(document, config, task=args.llm_task, glossary_text=args.glossary)
                row["llm_output_chars"] = len(output)
                row["correction_count"] = 0
            row["llm_seconds"] = _round(perf_counter() - llm_start)
    except Exception as exc:
        row["status"] = f"failed: {exc}"
    finally:
        row["total_seconds"] = _round(perf_counter() - total_start)
    return row


def _round(value: float | None) -> str:
    if value is None:
        return ""
    return f"{float(value):.3f}"


if __name__ == "__main__":
    main()
