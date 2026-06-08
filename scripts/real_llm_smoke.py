from __future__ import annotations

import argparse
import os
import sys

from transcript_lok.asr import ASRConfig, WhisperTranscriber
from transcript_lok.llm import SummaryConfig, correct_transcript, provider_health, run_llm_task
from transcript_lok.models import Segment, TranscriptDocument


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a gated real LLM smoke test against a local llama.cpp/Gemma server.")
    parser.add_argument("--endpoint", default=os.getenv("LLM_REAL_ENDPOINT", "http://127.0.0.1:8080/v1"))
    parser.add_argument("--model", default=os.getenv("LLM_REAL_MODEL", "gemma-4"))
    parser.add_argument("--provider", default=os.getenv("LLM_REAL_PROVIDER", "llama.cpp"))
    parser.add_argument("--fixture", default="")
    parser.add_argument("--force", action="store_true", help="Run even when LLM_REAL_TEST is not set to 1.")
    args = parser.parse_args()

    if not args.force and os.getenv("LLM_REAL_TEST") != "1":
        print("SKIP: set LLM_REAL_TEST=1 or pass --force to run the real LLM smoke test.")
        return 0

    config = SummaryConfig(provider=args.provider, model=args.model, endpoint=args.endpoint, max_chars_per_call=12_000)
    ok, message = provider_health(config)
    if not ok:
        print(f"SKIP: {message}")
        return 0

    glossary = "ACME Project\nTranscript LOK\nKevin\nFriday"
    if args.fixture:
        document = WhisperTranscriber(ASRConfig(model_size="tiny", language="en", device="cpu", compute_type="int8")).transcribe_file(args.fixture, glossary)
    else:
        document = TranscriptDocument(
            segments=[
                Segment(0, 2, "Kevn says the Akme projekt report must be finisht by Friday.", speaker="Kevin"),
                Segment(2, 4, "Anna will send the draft and ask legal for feed back.", speaker="Anna"),
            ]
        )

    correction = correct_transcript(document, config, glossary)
    print(f"Correction changes: {len(correction.changes)}")
    if not correction.changes:
        return _fail("Expected at least one correction from the synthetic transcript.")

    tasks = [
        "meeting_summary",
        "action_points",
        "decision_summary_internal_anonymous",
        "decision_summary_external_anonymous",
        "follow_up_email",
    ]
    for task in tasks:
        output = run_llm_task(document, config, task=task, glossary_text=glossary)
        print(f"\n## {task}\n{output[:1200]}")
        if not output.strip():
            return _fail(f"{task} returned empty output.")

    return 0


def _fail(message: str) -> int:
    print(f"FAIL: {message}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
