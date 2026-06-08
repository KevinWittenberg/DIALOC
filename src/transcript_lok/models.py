from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
import csv
import html
import zipfile


TRANSCRIPT_COLUMNS = ["start", "end", "speaker", "text", "confidence"]


def _to_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text:
        return default
    if ":" not in text:
        return float(text.replace(",", "."))

    parts = text.replace(",", ".").split(":")
    seconds = float(parts[-1])
    minutes = int(parts[-2]) if len(parts) >= 2 else 0
    hours = int(parts[-3]) if len(parts) >= 3 else 0
    return hours * 3600 + minutes * 60 + seconds


def format_timestamp(seconds: float, separator: str = ",") -> str:
    seconds = max(float(seconds), 0.0)
    milliseconds = int(round((seconds - int(seconds)) * 1000))
    total_seconds = int(seconds)
    if milliseconds == 1000:
        total_seconds += 1
        milliseconds = 0
    hours, remainder = divmod(total_seconds, 3600)
    minutes, whole_seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{whole_seconds:02d}{separator}{milliseconds:03d}"


@dataclass(slots=True)
class Word:
    start: float
    end: float
    text: str
    probability: float | None = None
    speaker: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "start": self.start,
            "end": self.end,
            "text": self.text,
            "probability": self.probability,
            "speaker": self.speaker,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Word":
        return cls(
            start=_to_float(data.get("start")),
            end=_to_float(data.get("end")),
            text=str(data.get("text") or ""),
            probability=data.get("probability"),
            speaker=data.get("speaker"),
        )


@dataclass(slots=True)
class Segment:
    start: float
    end: float
    text: str
    speaker: str | None = None
    confidence: float | None = None
    words: list[Word] = field(default_factory=list)

    def to_row(self) -> list[Any]:
        confidence = "" if self.confidence is None else round(float(self.confidence), 3)
        return [round(self.start, 3), round(self.end, 3), self.speaker or "", self.text, confidence]

    def to_dict(self) -> dict[str, Any]:
        return {
            "start": self.start,
            "end": self.end,
            "text": self.text,
            "speaker": self.speaker,
            "confidence": self.confidence,
            "words": [word.to_dict() for word in self.words],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Segment":
        return cls(
            start=_to_float(data.get("start")),
            end=_to_float(data.get("end")),
            text=str(data.get("text") or ""),
            speaker=(str(data.get("speaker")).strip() or None) if data.get("speaker") is not None else None,
            confidence=_optional_float(data.get("confidence")),
            words=[Word.from_dict(word) for word in data.get("words") or []],
        )


@dataclass(slots=True)
class TranscriptDocument:
    segments: list[Segment] = field(default_factory=list)
    language: str | None = None
    audio_path: str | None = None
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_rows(self) -> list[list[Any]]:
        return [segment.to_row() for segment in self.segments]

    def speaker_labels(self) -> list[str]:
        labels = {segment.speaker for segment in self.segments if segment.speaker}
        return sorted(labels)

    def rename_speakers(self, mapping: dict[str, str]) -> None:
        clean_mapping = {key.strip(): value.strip() for key, value in mapping.items() if key.strip() and value.strip()}
        if not clean_mapping:
            return
        for segment in self.segments:
            if segment.speaker in clean_mapping:
                segment.speaker = clean_mapping[segment.speaker]
            for word in segment.words:
                if word.speaker in clean_mapping:
                    word.speaker = clean_mapping[word.speaker]

    def plain_text(self, include_timestamps: bool = True) -> str:
        lines: list[str] = []
        for segment in self.segments:
            speaker = segment.speaker or "UNKNOWN"
            if include_timestamps:
                lines.append(f"[{format_timestamp(segment.start, '.')}] {speaker}: {segment.text}")
            else:
                lines.append(f"{speaker}: {segment.text}")
        return "\n".join(lines)

    def to_markdown(self) -> str:
        if not self.segments:
            return ""
        lines = ["| Start | End | Speaker | Text |", "|---:|---:|---|---|"]
        for segment in self.segments:
            start = format_timestamp(segment.start, ".")
            end = format_timestamp(segment.end, ".")
            speaker = (segment.speaker or "").replace("|", "\\|")
            text = segment.text.replace("\n", " ").replace("|", "\\|")
            lines.append(f"| {start} | {end} | {speaker} | {text} |")
        return "\n".join(lines)

    def to_srt(self) -> str:
        blocks: list[str] = []
        for index, segment in enumerate(self.segments, start=1):
            speaker = f"{segment.speaker}: " if segment.speaker else ""
            blocks.append(
                "\n".join(
                    [
                        str(index),
                        f"{format_timestamp(segment.start)} --> {format_timestamp(segment.end)}",
                        f"{speaker}{segment.text}",
                    ]
                )
            )
        return "\n\n".join(blocks) + ("\n" if blocks else "")

    def to_vtt(self) -> str:
        blocks = ["WEBVTT", ""]
        for segment in self.segments:
            speaker = f"{segment.speaker}: " if segment.speaker else ""
            blocks.extend(
                [
                    f"{format_timestamp(segment.start, '.')} --> {format_timestamp(segment.end, '.')}",
                    f"{speaker}{segment.text}",
                    "",
                ]
            )
        return "\n".join(blocks)

    def to_dict(self) -> dict[str, Any]:
        return {
            "language": self.language,
            "audio_path": self.audio_path,
            "created_at": self.created_at,
            "metadata": self.metadata,
            "segments": [segment.to_dict() for segment in self.segments],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "TranscriptDocument":
        if not data:
            return cls()
        return cls(
            language=data.get("language"),
            audio_path=data.get("audio_path"),
            created_at=str(data.get("created_at") or datetime.now(timezone.utc).isoformat()),
            metadata=dict(data.get("metadata") or {}),
            segments=[Segment.from_dict(segment) for segment in data.get("segments") or []],
        )

    @classmethod
    def from_rows(
        cls,
        rows: Any,
        *,
        language: str | None = None,
        audio_path: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> "TranscriptDocument":
        normalized_rows = dataframe_to_rows(rows)
        segments: list[Segment] = []
        for row in normalized_rows:
            if len(row) < 4:
                continue
            text = str(row[3] or "").strip()
            if not text:
                continue
            speaker = str(row[2] or "").strip() or None
            start = _to_float(row[0])
            end = max(_to_float(row[1], start), start)
            confidence = _optional_float(row[4]) if len(row) > 4 else None
            segments.append(Segment(start=start, end=end, speaker=speaker, text=text, confidence=confidence))
        return cls(segments=segments, language=language, audio_path=audio_path, metadata=metadata or {})

    def write_exports(self, output_dir: str | Path, stem: str | None = None) -> list[str]:
        target = Path(output_dir)
        target.mkdir(parents=True, exist_ok=True)
        safe_stem = stem or datetime.now().strftime("transcript_%Y%m%d_%H%M%S")
        paths = [
            target / f"{safe_stem}.md",
            target / f"{safe_stem}.json",
            target / f"{safe_stem}.srt",
            target / f"{safe_stem}.vtt",
            target / f"{safe_stem}.csv",
            target / f"{safe_stem}.docx",
        ]
        paths[0].write_text(self.to_markdown(), encoding="utf-8")
        paths[1].write_text(_json_dump(self.to_dict()), encoding="utf-8")
        paths[2].write_text(self.to_srt(), encoding="utf-8")
        paths[3].write_text(self.to_vtt(), encoding="utf-8")
        self.write_csv(paths[4])
        self.write_docx(paths[5])
        return [str(path.resolve()) for path in paths]

    def write_csv(self, path: str | Path) -> None:
        with Path(path).open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.writer(handle)
            writer.writerow(TRANSCRIPT_COLUMNS)
            writer.writerows(self.to_rows())

    def write_docx(self, path: str | Path) -> None:
        paragraphs = []
        for segment in self.segments:
            speaker = f"{segment.speaker}: " if segment.speaker else ""
            timestamp = f"[{format_timestamp(segment.start, '.')}] "
            paragraphs.append(_docx_paragraph(timestamp + speaker + segment.text))
        document_xml = (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
            f"<w:body>{''.join(paragraphs)}<w:sectPr><w:pgSz w:w=\"12240\" w:h=\"15840\"/>"
            '<w:pgMar w:top="1440" w:right="1440" w:bottom="1440" w:left="1440"/>'
            "</w:sectPr></w:body></w:document>"
        )
        content_types = (
            '<?xml version="1.0" encoding="UTF-8"?>'
            '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
            '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
            '<Default Extension="xml" ContentType="application/xml"/>'
            '<Override PartName="/word/document.xml" '
            'ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>'
            "</Types>"
        )
        rels = (
            '<?xml version="1.0" encoding="UTF-8"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rId1" '
            'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" '
            'Target="word/document.xml"/>'
            "</Relationships>"
        )
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("[Content_Types].xml", content_types)
            archive.writestr("_rels/.rels", rels)
            archive.writestr("word/document.xml", document_xml)


def dataframe_to_rows(rows: Any) -> list[list[Any]]:
    if rows is None:
        return []
    if hasattr(rows, "values") and hasattr(rows, "columns"):
        return rows[TRANSCRIPT_COLUMNS].values.tolist() if set(TRANSCRIPT_COLUMNS).issubset(rows.columns) else rows.values.tolist()
    if isinstance(rows, dict) and "data" in rows:
        return list(rows["data"] or [])
    if isinstance(rows, Iterable) and not isinstance(rows, (str, bytes)):
        return [list(row.values()) if isinstance(row, dict) else list(row) for row in rows]
    return []


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text.replace(",", "."))
    except ValueError:
        return None


def parse_speaker_mapping(text: str) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for raw_line in (text or "").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        separator = "=" if "=" in line else ":"
        if separator not in line:
            continue
        key, value = line.split(separator, 1)
        mapping[key.strip()] = value.strip()
    return mapping


def _json_dump(data: dict[str, Any]) -> str:
    import json

    return json.dumps(data, ensure_ascii=False, indent=2)


def _docx_paragraph(text: str) -> str:
    escaped = html.escape(text)
    return f"<w:p><w:r><w:t>{escaped}</w:t></w:r></w:p>"
