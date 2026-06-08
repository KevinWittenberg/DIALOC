from __future__ import annotations

import argparse
import os
from pathlib import Path

from transcript_lok.asr import ASRConfig, WhisperTranscriber
from transcript_lok.diarization import DEFAULT_DIARIZATION_MODEL, suppress_pyannote_torchcodec_warning


def main() -> None:
    parser = argparse.ArgumentParser(description="Pre-download local transcription models.")
    parser.add_argument("--whisper-model", default="large-v3")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--compute-type", default="auto")
    parser.add_argument("--diarization-model", default=DEFAULT_DIARIZATION_MODEL)
    parser.add_argument("--hf-token", default=os.getenv("HUGGINGFACE_TOKEN") or os.getenv("HF_TOKEN"))
    parser.add_argument("--skip-diarization", action="store_true")
    args = parser.parse_args()

    if args.hf_token:
        os.environ.setdefault("HF_TOKEN", args.hf_token)
        os.environ.setdefault("HUGGINGFACE_TOKEN", args.hf_token)

    WhisperTranscriber(ASRConfig(model_size=args.whisper_model, device=args.device, compute_type=args.compute_type))
    print(f"Downloaded faster-whisper model: {args.whisper_model}")

    if not args.skip_diarization:
        if not args.hf_token and not Path(args.diarization_model).exists():
            print(
                "Skipped diarization model download: provide --hf-token or set HUGGINGFACE_TOKEN after "
                "accepting the pyannote model terms on Hugging Face. Use --skip-diarization when you only "
                "want to pre-download Whisper."
            )
            return

        suppress_pyannote_torchcodec_warning()
        from pyannote.audio import Pipeline

        kwargs = {"token": args.hf_token} if args.hf_token else {}
        Pipeline.from_pretrained(args.diarization_model, **kwargs)
        print(f"Downloaded diarization model: {args.diarization_model}")


if __name__ == "__main__":
    main()
