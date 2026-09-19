from __future__ import annotations

import hashlib
import re

_NON_ALNUM = re.compile(r"[^a-z0-9]+")
_WS = re.compile(r"\s+")


def norm(text: str) -> str:
    """Aggressive normalization for span matching: PDF extraction mangles whitespace,
    hyphenation and quotes, so compare on lowercase alphanumerics only."""
    return _NON_ALNUM.sub("", text.lower())


def squash(text: str) -> str:
    return _WS.sub(" ", text).strip()


def quote_in_text(quote: str, text: str, min_len: int = 12) -> bool:
    """True if `quote` appears verbatim (modulo whitespace/punctuation) in `text`.
    Ellipses in the quote are treated as gaps: every fragment must appear in order."""
    fragments = [norm(f) for f in re.split(r"\.\.\.|…", quote)]
    fragments = [f for f in fragments if f]
    if not fragments or sum(map(len, fragments)) < min_len:
        return False
    hay = norm(text)
    pos = 0
    for frag in fragments:
        idx = hay.find(frag, pos)
        if idx < 0:
            return False
        pos = idx + len(frag)
    return True


def slugify(title: str, max_len: int = 48) -> str:
    base = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")[:max_len].strip("-")
    digest = hashlib.sha1(title.encode()).hexdigest()[:6]
    return f"{base}-{digest}"
