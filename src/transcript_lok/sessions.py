from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from .models import TranscriptDocument


def save_session(
    output_root: str | Path,
    document: TranscriptDocument,
    *,
    settings: dict[str, Any] | None = None,
    glossary: str = "",
    speaker_mapping: Any = None,
    summary: str = "",
    exports: list[str] | None = None,
    llm_outputs: dict[str, Any] | None = None,
    name: str | None = None,
) -> str:
    root = Path(output_root)
    session_name = name or datetime.now().strftime("session_%Y%m%d_%H%M%S")
    target = root / "sessions" / session_name
    target.mkdir(parents=True, exist_ok=True)
    payload = {
        "created_at": datetime.now().isoformat(),
        "document": document.to_dict(),
        "settings": settings or {},
        "glossary": glossary,
        "speaker_mapping": speaker_mapping,
        "summary": summary,
        "llm_outputs": llm_outputs or {},
        "exports": exports or [],
    }
    session_path = target / "session.json"
    session_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(session_path.resolve())


def load_session(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    payload["document"] = TranscriptDocument.from_dict(payload.get("document"))
    return payload
