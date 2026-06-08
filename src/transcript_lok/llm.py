from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
import hashlib
import json
import re
from typing import Any, Literal

import requests

from .llama_cpp import default_llama_cpp_endpoint
from .models import TranscriptDocument, format_timestamp


Provider = Literal["ollama", "llama.cpp", "openai", "openai-compatible"]
ArtifactLanguage = Literal["nl", "en"]
LLMTask = Literal[
    "correction_pass",
    "meeting_summary",
    "action_points",
    "decision_summary_internal_anonymous",
    "decision_summary_external_anonymous",
    "follow_up_email",
]


@dataclass(frozen=True)
class SummaryConfig:
    provider: Provider = "ollama"
    model: str = "llama3.1"
    endpoint: str | None = None
    api_key: str | None = None
    max_chars_per_call: int = 30_000
    template: str = "Meeting summary"
    custom_prompt: str | None = None
    artifact_language: ArtifactLanguage = "nl"


@dataclass(frozen=True)
class CorrectionChange:
    segment_index: int
    start: float
    end: float
    speaker: str
    original_text: str
    corrected_text: str
    reason: str = ""
    confidence: float | None = None


@dataclass(frozen=True)
class CorrectionReviewFlag:
    segment_index: int
    start: float
    end: float
    speaker: str
    text: str
    reason: str = ""
    confidence: float | None = None


@dataclass(frozen=True)
class CorrectionResult:
    document: TranscriptDocument
    changes: list[CorrectionChange]
    review_flags: list[CorrectionReviewFlag]
    raw_response: str
    provenance: dict[str, str]


SYSTEM_PROMPT = (
    "You are an expert meeting transcript assistant. Ground every output in the supplied transcript. "
    "Do not invent facts, owners, deadlines, or decisions. Use `Unknown` when the transcript does not state a value."
)

LANGUAGE_NAMES: dict[str, str] = {"nl": "Dutch", "en": "English"}

TASK_INSTRUCTIONS: dict[str, dict[LLMTask, str]] = {
    "en": {
        "meeting_summary": (
            "Create a concise meeting summary. Return Markdown with these headings: "
            "Summary, Decisions, Action Points, Open Questions. Use speaker names when available. "
            "If no speaker names are available, avoid invented attribution."
        ),
        "action_points": (
            "Extract concrete action points. Return a Markdown table with columns: What, Who, Deadline, Evidence. "
            "Evidence must include a timestamp or short quote. Use `Unknown` for missing who or deadline."
        ),
        "decision_summary_internal_anonymous": (
            "Create an anonymous decision summary for internal use. Return Markdown with headings: "
            "General Context, Decisions, Rationale, Open Risks. Remove people names and speaker labels. "
            "Keep project, product, and domain terms if needed for internal usefulness."
        ),
        "decision_summary_external_anonymous": (
            "Create an anonymous decision summary for external sharing. Return Markdown with headings: "
            "General Context, Decisions, Rationale, Open Risks. Remove names, speaker labels, organizations, "
            "products, locations, and unique project identifiers where possible. Use general terms only."
        ),
        "follow_up_email": (
            "Draft an attendee-facing follow-up email. Return Markdown with headings: Subject, Email. "
            "The email must include a short recap, action items with owners/deadlines when stated, and open questions. "
            "Use `Unknown` for missing owners or deadlines."
        ),
        "correction_pass": "",
    },
    "nl": {
        "meeting_summary": (
            "Maak een beknopte Nederlandstalige vergadersamenvatting. Geef Markdown terug met deze koppen: "
            "Samenvatting, Besluiten, Actiepunten, Open vragen. Gebruik sprekernamen wanneer die beschikbaar zijn. "
            "Verzin geen toeschrijving als sprekers ontbreken."
        ),
        "action_points": (
            "Haal concrete actiepunten uit het transcript. Geef een Markdown-tabel terug met kolommen: "
            "Wat, Wie, Deadline, Bewijs. Bewijs moet een tijdstempel of kort citaat bevatten. "
            "Gebruik `Onbekend` wanneer eigenaar of deadline niet genoemd wordt."
        ),
        "decision_summary_internal_anonymous": (
            "Maak een anonieme besluitensamenvatting voor intern gebruik. Geef Markdown terug met koppen: "
            "Algemene context, Besluiten, Redenatie, Open risico's. Verwijder persoonsnamen en sprekerlabels. "
            "Behoud project-, product- en domeintermen wanneer die intern nodig zijn."
        ),
        "decision_summary_external_anonymous": (
            "Maak een anonieme besluitensamenvatting voor extern delen. Geef Markdown terug met koppen: "
            "Algemene context, Besluiten, Redenatie, Open risico's. Verwijder namen, sprekerlabels, organisaties, "
            "producten, locaties en unieke projectnamen waar mogelijk. Gebruik algemene termen."
        ),
        "follow_up_email": (
            "Schrijf een Nederlandstalige opvolgmail voor deelnemers. Geef Markdown terug met koppen: Onderwerp, E-mail. "
            "De e-mail bevat een korte terugblik, actiepunten met eigenaar/deadline wanneer genoemd, en open vragen. "
            "Gebruik `Onbekend` wanneer eigenaar of deadline ontbreekt."
        ),
        "correction_pass": "",
    },
}

TEMPLATE_TASKS: dict[str, LLMTask] = {
    "Meeting summary": "meeting_summary",
    "Action points": "action_points",
    "Decision summary - internal anonymous": "decision_summary_internal_anonymous",
    "Decision summary - external anonymous": "decision_summary_external_anonymous",
    "Follow-up email": "follow_up_email",
}


def summarize_transcript(document: TranscriptDocument, config: SummaryConfig) -> str:
    return run_llm_task(document, config, task=TEMPLATE_TASKS.get(config.template, "meeting_summary"))


def run_llm_task(
    document: TranscriptDocument,
    config: SummaryConfig,
    *,
    task: LLMTask,
    glossary_text: str = "",
) -> str:
    transcript = document.plain_text(include_timestamps=True)
    if not transcript.strip():
        return "No transcript text is available."

    chunks = _chunk_text(transcript, config.max_chars_per_call)
    if len(chunks) == 1:
        return _chat(config, _task_messages_for(chunks[0], config, task, glossary_text=glossary_text))

    partials: list[str] = []
    for index, chunk in enumerate(chunks, start=1):
        partials.append(
            _chat(
                config,
                _task_messages_for(
                    chunk,
                    config,
                    task,
                    glossary_text=glossary_text,
                    chunk_note=f"Process transcript part {index} of {len(chunks)}.",
                ),
            )
        )

    combine_prompt = (
        f"{_task_instruction(task, config)}\n\n"
        "Combine these partial outputs into one final result. Keep the required output format. "
        "Do not add facts that are not present in the partial outputs.\n\n"
        "<partials>\n"
        + "\n\n---\n\n".join(partials)
        + "\n</partials>"
    )
    return _chat(config, _messages_for(config, SYSTEM_PROMPT, combine_prompt))


def correct_transcript(document: TranscriptDocument, config: SummaryConfig, glossary_text: str = "") -> CorrectionResult:
    if not document.segments:
        return CorrectionResult(
            document=document,
            changes=[],
            review_flags=[],
            raw_response="",
            provenance=llm_provenance(document, config, "correction_pass", glossary_text),
        )

    raw_parts: list[str] = []
    all_changes: list[CorrectionChange] = []
    all_review_flags: list[CorrectionReviewFlag] = []
    for chunk in _segment_chunks(document, config.max_chars_per_call):
        response = _chat(config, _correction_messages_for(chunk, config, glossary_text), temperature=0.0)
        raw_parts.append(response)
        changes, review_flags = _parse_correction_payload(response, document)
        all_changes.extend(changes)
        all_review_flags.extend(review_flags)

    changes = _dedupe_changes(all_changes)
    review_flags = _dedupe_review_flags(all_review_flags)
    for change in changes:
        document.segments[change.segment_index].text = change.corrected_text

    provenance = llm_provenance(document, config, "correction_pass", glossary_text)
    audit = {
        **provenance,
        "changed_segments": str(len(changes)),
        "changes": [asdict(change) for change in changes],
        "review_flags": [asdict(flag) for flag in review_flags],
    }
    document.metadata.setdefault("llm_correction_runs", []).append(audit)
    document.metadata.setdefault("llm_outputs", {})["correction_pass"] = audit
    if review_flags:
        document.metadata.setdefault("review_flags", []).extend(asdict(flag) for flag in review_flags)
    return CorrectionResult(document=document, changes=changes, review_flags=review_flags, raw_response="\n\n".join(raw_parts), provenance=provenance)


def parse_correction_response(response_text: str, document: TranscriptDocument) -> list[CorrectionChange]:
    changes, _review_flags = _parse_correction_payload(response_text, document)
    return changes


def parse_correction_review_flags(response_text: str, document: TranscriptDocument) -> list[CorrectionReviewFlag]:
    _changes, review_flags = _parse_correction_payload(response_text, document)
    return review_flags


def _parse_correction_payload(response_text: str, document: TranscriptDocument) -> tuple[list[CorrectionChange], list[CorrectionReviewFlag]]:
    data = _json_loads_lenient(response_text)
    if isinstance(data, dict):
        items = data.get("segments") or data.get("corrections") or data.get("changes") or []
        flag_items = data.get("review_flags") or data.get("flags") or []
    elif isinstance(data, list):
        items = data
        flag_items = []
    else:
        raise ValueError("Correction response must be a JSON object or array.")

    changes: list[CorrectionChange] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        index = _correction_index(item)
        if index is None or index < 0 or index >= len(document.segments):
            continue
        corrected_text = str(item.get("corrected_text") or item.get("text") or "").strip()
        if not corrected_text:
            continue
        original = document.segments[index].text
        if corrected_text == original:
            continue
        segment = document.segments[index]
        changes.append(
            CorrectionChange(
                segment_index=index,
                start=segment.start,
                end=segment.end,
                speaker=segment.speaker or "UNKNOWN",
                original_text=original,
                corrected_text=corrected_text,
                reason=str(item.get("reason") or "").strip(),
                confidence=_optional_float(item.get("confidence")),
            )
        )
    review_flags = _parse_review_flags(flag_items, document)
    return changes, review_flags


def provider_health(config: SummaryConfig) -> tuple[bool, str]:
    try:
        if config.provider == "ollama":
            endpoint = (config.endpoint or "http://localhost:11434").rstrip("/")
            response = requests.get(f"{endpoint}/api/tags", timeout=5)
            response.raise_for_status()
            return True, f"Ollama reachable at {endpoint}."

        endpoint = _openai_compatible_endpoint(config)
        headers = {"Authorization": f"Bearer {config.api_key}"} if config.api_key else {}
        response = requests.get(f"{endpoint}/models", headers=headers, timeout=8)
        response.raise_for_status()
        return True, f"{config.provider} reachable at {endpoint}."
    except Exception as exc:
        return False, f"{config.provider} health check failed: {exc}"


def summary_provenance(document: TranscriptDocument, config: SummaryConfig) -> dict[str, str]:
    return llm_provenance(document, config, TEMPLATE_TASKS.get(config.template, "meeting_summary"))


def llm_provenance(
    document: TranscriptDocument,
    config: SummaryConfig,
    task: str,
    glossary_text: str = "",
) -> dict[str, str]:
    transcript = document.plain_text(include_timestamps=True)
    return {
        "provider": config.provider,
        "model": config.model,
        "task": task,
        "template": config.template,
        "artifact_language": _artifact_language(config),
        "artifact_language_name": LANGUAGE_NAMES[_artifact_language(config)],
        "created_at": datetime.now().isoformat(),
        "transcript_hash": hashlib.sha256(transcript.encode("utf-8")).hexdigest(),
        "glossary_hash": hashlib.sha256((glossary_text or "").encode("utf-8")).hexdigest(),
    }


def _summary_messages(transcript: str) -> list[dict[str, str]]:
    return _task_messages_for(transcript, SummaryConfig(), "meeting_summary")


def _summary_messages_for(transcript: str, config: SummaryConfig) -> list[dict[str, str]]:
    return _task_messages_for(transcript, config, TEMPLATE_TASKS.get(config.template, "meeting_summary"))


def _task_messages_for(
    transcript: str,
    config: SummaryConfig,
    task: LLMTask,
    *,
    glossary_text: str = "",
    chunk_note: str = "",
) -> list[dict[str, str]]:
    instruction = config.custom_prompt or _task_instruction(task, config)
    prompt = (
        f"<instructions>\n{instruction}\n{chunk_note}\n</instructions>\n\n"
        f"<glossary>\n{glossary_text or 'No glossary terms provided.'}\n</glossary>\n\n"
        f"<transcript>\n{transcript}\n</transcript>"
    )
    return _messages_for(config, SYSTEM_PROMPT, prompt)


def _correction_messages_for(chunk: str, config: SummaryConfig, glossary_text: str) -> list[dict[str, str]]:
    instruction = (
        "Run an ASR correction pass over the transcript segment lines. Fix likely transcription typos, casing, "
        "punctuation, and glossary spelling. Preserve the original meaning. Do not summarize, reorder, merge, split, "
        "translate, or change speaker labels or timestamps. Keep each segment in its original language. "
        "Use glossary terms as canonical spellings for names, jargon, acronyms, product names, and domain terms. "
        "Be conservative: if a span is suspicious but not safe to rewrite, add it to `review_flags` instead of changing it. "
        "Return ONLY valid JSON in this shape: "
        "{\"segments\":[{\"segment_index\":1,\"corrected_text\":\"...\",\"reason\":\"...\",\"confidence\":0.0}],"
        "\"review_flags\":[{\"segment_index\":1,\"text\":\"...\",\"reason\":\"...\",\"confidence\":0.0}]}. "
        "Only include changed segments. `segment_index` is the 1-based index shown in the transcript."
    )
    prompt = (
        f"<instructions>\n{instruction}\n</instructions>\n\n"
        f"<glossary>\n{glossary_text or 'No glossary terms provided.'}\n</glossary>\n\n"
        f"<transcript>\n{chunk}\n</transcript>"
    )
    return _messages_for(config, SYSTEM_PROMPT, prompt)


def _messages_for(config: SummaryConfig, system_prompt: str, user_prompt: str) -> list[dict[str, str]]:
    if config.provider == "llama.cpp":
        return [{"role": "user", "content": f"{system_prompt}\n\n{user_prompt}"}]
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def _task_instruction(task: LLMTask, config: SummaryConfig) -> str:
    language = _artifact_language(config)
    if task in TASK_INSTRUCTIONS[language]:
        instruction = TASK_INSTRUCTIONS[language][task]
        return f"Output language: {LANGUAGE_NAMES[language]}.\n{instruction}"
    raise ValueError(f"Unsupported LLM task: {task}")


def _artifact_language(config: SummaryConfig) -> ArtifactLanguage:
    return "en" if config.artifact_language == "en" else "nl"


def _chat(config: SummaryConfig, messages: list[dict[str, str]], *, temperature: float = 0.2) -> str:
    if config.provider == "ollama":
        endpoint = (config.endpoint or "http://localhost:11434").rstrip("/")
        response = requests.post(
            f"{endpoint}/api/chat",
            json={"model": config.model, "messages": messages, "stream": False, "options": {"temperature": temperature}},
            timeout=180,
        )
        response.raise_for_status()
        data = response.json()
        return data.get("message", {}).get("content", "").strip()

    endpoint = _openai_compatible_endpoint(config)
    headers = {"Content-Type": "application/json"}
    if config.api_key:
        headers["Authorization"] = f"Bearer {config.api_key}"

    response = requests.post(
        f"{endpoint}/chat/completions",
        headers=headers,
        json={"model": config.model, "messages": messages, "temperature": temperature},
        timeout=180,
    )
    response.raise_for_status()
    data = response.json()
    return data["choices"][0]["message"]["content"].strip()


def _openai_compatible_endpoint(config: SummaryConfig) -> str:
    if config.provider == "llama.cpp":
        return (config.endpoint or default_llama_cpp_endpoint()).rstrip("/")
    return (config.endpoint or "https://api.openai.com/v1").rstrip("/")


def _segment_chunks(document: TranscriptDocument, max_chars: int) -> list[str]:
    chunks: list[str] = []
    current: list[str] = []
    current_size = 0
    for index, segment in enumerate(document.segments, start=1):
        line = f"[{index}] [{format_timestamp(segment.start, '.')}] {segment.speaker or 'UNKNOWN'}: {segment.text}"
        extra = len(line) + 1
        if current and current_size + extra > max_chars:
            chunks.append("\n".join(current))
            current = []
            current_size = 0
        current.append(line)
        current_size += extra
    if current:
        chunks.append("\n".join(current))
    return chunks


def _chunk_text(text: str, max_chars: int) -> list[str]:
    if len(text) <= max_chars:
        return [text]

    chunks: list[str] = []
    current: list[str] = []
    current_size = 0
    for line in text.splitlines():
        extra = len(line) + 1
        if current and current_size + extra > max_chars:
            chunks.append("\n".join(current))
            current = []
            current_size = 0
        current.append(line)
        current_size += extra
    if current:
        chunks.append("\n".join(current))
    return chunks


def _json_loads_lenient(text: str) -> Any:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped, flags=re.IGNORECASE)
        stripped = re.sub(r"\s*```$", "", stripped)
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start >= 0 and end > start:
            return json.loads(stripped[start : end + 1])
        raise


def _correction_index(item: dict[str, Any]) -> int | None:
    if "zero_based_index" in item:
        try:
            return int(item["zero_based_index"])
        except (TypeError, ValueError):
            return None
    for key in ("segment_index", "index", "row"):
        if key in item:
            try:
                return int(item[key]) - 1
            except (TypeError, ValueError):
                return None
    return None


def _parse_review_flags(items: Any, document: TranscriptDocument) -> list[CorrectionReviewFlag]:
    if not isinstance(items, list):
        return []
    flags: list[CorrectionReviewFlag] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        index = _correction_index(item)
        if index is None or index < 0 or index >= len(document.segments):
            continue
        segment = document.segments[index]
        text = str(item.get("text") or item.get("span") or segment.text or "").strip()
        reason = str(item.get("reason") or item.get("issue") or "").strip()
        if not text and not reason:
            continue
        flags.append(
            CorrectionReviewFlag(
                segment_index=index,
                start=segment.start,
                end=segment.end,
                speaker=segment.speaker or "UNKNOWN",
                text=text,
                reason=reason,
                confidence=_optional_float(item.get("confidence")),
            )
        )
    return flags


def _dedupe_changes(changes: list[CorrectionChange]) -> list[CorrectionChange]:
    by_index: dict[int, CorrectionChange] = {}
    for change in changes:
        by_index[change.segment_index] = change
    return [by_index[index] for index in sorted(by_index)]


def _dedupe_review_flags(flags: list[CorrectionReviewFlag]) -> list[CorrectionReviewFlag]:
    by_key: dict[tuple[int, str, str], CorrectionReviewFlag] = {}
    for flag in flags:
        by_key[(flag.segment_index, flag.text, flag.reason)] = flag
    return [by_key[key] for key in sorted(by_key)]


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(str(value).replace(",", "."))
    except ValueError:
        return None
