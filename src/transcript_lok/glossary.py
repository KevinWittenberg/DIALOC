from __future__ import annotations

import re


def parse_glossary(text: str | None) -> list[str]:
    if not text:
        return []
    terms: list[str] = []
    seen: set[str] = set()
    for line in text.splitlines():
        clean_line = line.split("#", 1)[0].strip()
        if not clean_line:
            continue
        for term in re.split(r"[,;]", clean_line):
            normalized = " ".join(term.strip().split())
            key = normalized.casefold()
            if normalized and key not in seen:
                seen.add(key)
                terms.append(normalized)
    return terms


def build_asr_prompt(glossary_text: str | None, language: str | None) -> str | None:
    terms = parse_glossary(glossary_text)
    if not terms:
        return None

    if language == "nl":
        prefix = "Let op deze namen, technische termen en afkortingen in de transcriptie:"
    else:
        prefix = "Pay attention to these names, technical terms, and acronyms in the transcript:"
    return f"{prefix} {', '.join(terms)}."
