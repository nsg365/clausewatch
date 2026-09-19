"""PDF -> pages -> section-aware chunks that remember their page span and section heading."""

from __future__ import annotations

import bisect
import re
from dataclasses import dataclass, field
from pathlib import Path

from pypdf import PdfReader

_HEADING_PATTERNS = [
    # "ARTICLE 5 - TERMINATION", "Section 12. Governing Law"
    re.compile(r"^\s*((?:ARTICLE|Article|SECTION|Section)\s+[0-9IVXLC]+(?:\.\d+)*[.:\-\s]*[A-Za-z][^\n]{0,80})$"),
    # "12. Termination", "12.1 Termination for Cause. Either party may ..."
    re.compile(r"^\s*(\d{1,2}(?:\.\d{1,2}){0,2}\.?\s*[A-Z][A-Za-z,;&/' \-]{2,70}?)(?:\.\s|:\s|\s*$)"),
    # "TERMINATION", "LIMITATION OF LIABILITY"
    re.compile(r"^\s*([A-Z][A-Z0-9 ,;&/'()\-]{3,70})\s*$"),
]


@dataclass
class Chunk:
    text: str
    section: str
    page_start: int
    page_end: int
    char_start: int
    metadata: dict = field(default_factory=dict)


# EDGAR footers ("Source: TUBE MEDIA CORP., 8-K, 3/10/2006") and bare page numbers.
_NOISE = re.compile(r"^\s*(Source: .*|-?\s*\d{1,3}\s*-?|Page \d+( of \d+)?)\s*$", re.M)


_UNICODE_FIXES = str.maketrans({"\xa0": " ", " ": " ", "​": "", "’": "'", "‘": "'",
                                "“": '"', "”": '"', "–": "-", "—": "-"})


def read_pdf_pages(path: str | Path) -> list[str]:
    reader = PdfReader(str(path))
    return [_NOISE.sub("", (p.extract_text() or "").translate(_UNICODE_FIXES)) for p in reader.pages]


def _align_start(body: str, i: int, limit: int) -> int:
    """Move a chunk start forward to a sentence or word boundary (never past `limit`)."""
    window = body[i:limit]
    for sep in (". ", ".\n", "\n"):
        k = window.find(sep)
        if 0 <= k < len(window) - 1:
            return i + k + len(sep)
    k = window.find(" ")
    return i + k + 1 if k >= 0 else i


def _heading(line: str) -> str | None:
    if len(line.strip()) < 4:
        return None
    for pat in _HEADING_PATTERNS:
        m = pat.match(line)
        if m:
            h = re.sub(r"\s+", " ", m.group(1)).strip(" .:-")
            # ALL-CAPS pattern: skip lines that are mostly numbers/signature noise
            if sum(ch.isalpha() for ch in h) >= 4:
                return h[:90]
    return None


def chunk_pages(pages: list[str], chunk_size: int = 1200, overlap: int = 150) -> list[Chunk]:
    full, page_offsets = "", []
    for p in pages:
        page_offsets.append(len(full))
        full += p.rstrip() + "\n"

    def page_of(offset: int) -> int:
        return bisect.bisect_right(page_offsets, offset)  # 1-indexed

    # 1) Split into sections at heading lines.
    sections: list[tuple[str, int, int]] = []  # (heading, start, end)
    current, start, pos = "Preamble", 0, 0
    for line in full.splitlines(keepends=True):
        h = _heading(line)
        if h and pos - start > 0:
            sections.append((current, start, pos))
            current, start = h, pos
        elif h:
            current = h
        pos += len(line)
    sections.append((current, start, len(full)))

    # 2) Pack paragraphs within each section; hard-split oversized ones with overlap.
    chunks: list[Chunk] = []
    for heading, s, e in sections:
        body = full[s:e]
        if not body.strip():
            continue
        i = 0
        while i < len(body):
            j = min(len(body), i + chunk_size)
            if j < len(body):
                # Prefer to cut on a paragraph / sentence boundary.
                window = body[i:j]
                cut = max(window.rfind("\n\n"), window.rfind(". "), window.rfind(".\n"))
                if cut > chunk_size * 0.5:
                    j = i + cut + 1
            text = body[i:j].strip()
            if len(text) >= 40:
                abs_start = s + i
                chunks.append(
                    Chunk(
                        text=text,
                        section=heading,
                        page_start=page_of(abs_start),
                        page_end=page_of(s + j - 1),
                        char_start=abs_start,
                    )
                )
            if j >= len(body):
                break
            i = _align_start(body, max(j - overlap, i + 1), j)
    return chunks
