from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter

from transcript_lok.asr import ASRConfig, WhisperTranscriber


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a timed local ASR smoke test.")
    parser.add_argument("audio", help="Path to a short audio file.")
    parser.add_argument("--model", default="small")
    parser.add_argument("--language", default="en")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--compute-type", default="auto")
    args = parser.parse_args()

    audio_path = Path(args.audio).resolve()
    if not audio_path.exists():
        raise SystemExit(f"Audio file does not exist: {audio_path}")

    start = perf_counter()
    transcriber = WhisperTranscriber(
        ASRConfig(
            model_size=args.model,
            language=args.language or None,
            device=args.device,
            compute_type=args.compute_type,
        )
    )
    loaded = perf_counter()
    document = transcriber.transcribe_file(str(audio_path))
    finished = perf_counter()

    print(f"Audio: {audio_path}")
    print(f"Segments: {len(document.segments)}")
    print(f"Model load seconds: {loaded - start:.2f}")
    print(f"Transcribe seconds: {finished - loaded:.2f}")
    print(f"Total seconds: {finished - start:.2f}")
    print(f"Runtime: {document.metadata.get('asr_device')}/{document.metadata.get('asr_compute_type')}")
    print()
    print(document.plain_text())


if __name__ == "__main__":
    main()
