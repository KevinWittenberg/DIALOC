from __future__ import annotations

import argparse

from transcript_lok.diagnostics import collect_diagnostics, diagnostics_markdown


def main() -> None:
    parser = argparse.ArgumentParser(description="Check Transcript LOK runtime health.")
    parser.add_argument("--model", default="small")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--compute-type", default="auto")
    parser.add_argument("--diarization-model", default="pyannote/speaker-diarization-community-1")
    args = parser.parse_args()

    diagnostics = collect_diagnostics(args.model, args.device, args.compute_type, args.diarization_model)
    print(diagnostics_markdown(diagnostics))


if __name__ == "__main__":
    main()
