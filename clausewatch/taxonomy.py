"""Clause taxonomy.

Every chunk is tagged with clause categories at ingest time by a deterministic keyword
tagger (the same path is used for CUAD contracts and user uploads). CUAD's expert labels
are mapped onto the same taxonomy and stored separately as `gold_clause_types`; they are
used only to *measure* retrieval, never to retrieve.
"""

from __future__ import annotations

import re

CLAUSE_CATEGORIES: dict[str, str] = {
    "parties": "Parties, recitals and document identity",
    "term": "Effective date, term and expiration",
    "renewal": "Renewal and auto-renewal",
    "termination": "Termination rights (cause, convenience, insolvency)",
    "payment": "Fees, pricing, payment terms, revenue sharing",
    "minimum_commitment": "Minimum purchase / volume commitments",
    "exclusivity": "Exclusivity and non-compete",
    "non_solicit": "Non-solicitation of employees or customers",
    "ip": "Intellectual property ownership, assignment and licensing",
    "license": "License grants and restrictions",
    "confidentiality": "Confidentiality and non-disclosure",
    "indemnification": "Indemnification and hold harmless",
    "liability": "Limitation or exclusion of liability, damages caps",
    "warranty": "Warranties and disclaimers",
    "insurance": "Insurance requirements",
    "assignment": "Assignment and change of control",
    "audit": "Audit and inspection rights",
    "governing_law": "Governing law, venue, dispute resolution",
    "mfn": "Most favored nation / price protection",
    "amendment": "Amendments, waivers, entire agreement",
    "liquidated_damages": "Liquidated damages and penalties",
}

_KEYWORDS: dict[str, list[str]] = {
    "parties": [r"\bby and between\b", r"\bentered into\b.*\bbetween\b", r"\bwhereas\b"],
    "term": [r"\bterm of this agreement\b", r"\beffective date\b", r"\bshall (?:commence|expire)\b", r"\binitial term\b"],
    "renewal": [r"\brenew", r"\bsuccessive (?:one|two|\d)", r"\bautomatically extend", r"\bevergreen\b"],
    "termination": [r"\bterminat", r"\bexpiration or termination\b"],
    "payment": [r"\bpayment", r"\bfees?\b", r"\binvoice", r"\broyalt", r"\bprice[sd]?\b", r"\bcommission", r"\brevenue shar"],
    "minimum_commitment": [r"\bminimum (?:purchase|order|quantity|volume|annual|commitment|royalt)", r"\btake[- ]or[- ]pay\b", r"\bshortfall\b"],
    "exclusivity": [r"\bexclusiv", r"\bnon-?compet", r"\bshall not (?:directly or indirectly )?(?:engage|compete)", r"\bcompetitive product"],
    "non_solicit": [r"\bsolicit", r"\bhire any (?:employee|person)"],
    "ip": [r"\bintellectual property\b", r"\bpatent", r"\btrademark", r"\bcopyright", r"\bwork(?:s)? made for hire\b", r"\bhereby assigns?\b", r"\bproprietary rights\b"],
    "license": [r"\blicen[cs]e", r"\bsublicens"],
    "confidentiality": [r"\bconfidential", r"\bnon-?disclosure\b", r"\bproprietary information\b"],
    "indemnification": [r"\bindemnif", r"\bhold (?:\w+ )?harmless\b", r"\bdefend\b"],
    "liability": [r"\blimitation of liability\b", r"\bliab(?:le|ility)\b", r"\bconsequential\b", r"\bincidental\b", r"\bin no event\b", r"\baggregate\b.*\bexceed"],
    "warranty": [r"\bwarrant", r"\bas is\b", r"\bmerchantability\b", r"\bfitness for a particular purpose\b"],
    "insurance": [r"\binsurance\b", r"\binsured\b", r"\bpolicy limits?\b"],
    "assignment": [r"\bassign", r"\bchange (?:of|in) control\b", r"\bmerger\b", r"\bsuccessors and assigns\b"],
    "audit": [r"\baudit", r"\binspect", r"\bbooks and records\b"],
    "governing_law": [r"\bgoverning law\b", r"\bgoverned by\b", r"\bjurisdiction\b", r"\barbitrat", r"\bvenue\b"],
    "mfn": [r"\bmost favou?red\b", r"\bno less favou?rable\b", r"\bbest price\b"],
    "amendment": [r"\bamend", r"\bentire agreement\b", r"\bwaiver\b", r"\bmodif(?:y|ied|ication)\b"],
    "liquidated_damages": [r"\bliquidated damages\b", r"\bpenalt(?:y|ies)\b"],
}
_COMPILED = {k: [re.compile(p, re.I) for p in v] for k, v in _KEYWORDS.items()}

# Heading words are a stronger signal than body mentions.
_HEADING_HINTS = {
    "termination": ["termination", "term and termination"],
    "indemnification": ["indemn"],
    "liability": ["liabilit", "damages"],
    "confidentiality": ["confidential"],
    "ip": ["intellectual property", "ownership", "proprietary"],
    "governing_law": ["governing law", "dispute", "arbitration"],
    "warranty": ["warrant"],
    "insurance": ["insurance"],
    "assignment": ["assignment"],
    "exclusivity": ["exclusiv", "non-compet", "noncompet"],
    "renewal": ["renewal"],
    "payment": ["payment", "compensation", "fees", "royalt", "price"],
    "audit": ["audit", "records"],
    "license": ["license", "grant"],
}

# CUAD category name -> taxonomy category (used for gold labels / retrieval eval only).
CUAD_LABEL_MAP: dict[str, str] = {
    "Document Name": "parties",
    "Parties": "parties",
    "Agreement Date": "term",
    "Effective Date": "term",
    "Expiration Date": "term",
    "Renewal Term": "renewal",
    "Notice Period To Terminate Renewal": "renewal",
    "Governing Law": "governing_law",
    "Most Favored Nation": "mfn",
    "Competitive Restriction Exception": "exclusivity",
    "Non-Compete": "exclusivity",
    "Exclusivity": "exclusivity",
    "No-Solicit Of Customers": "non_solicit",
    "No-Solicit Of Employees": "non_solicit",
    "Non-Disparagement": "non_solicit",
    "Termination For Convenience": "termination",
    "Rofr/Rofo/Rofn": "assignment",
    "Change Of Control": "assignment",
    "Anti-Assignment": "assignment",
    "Revenue/Profit Sharing": "payment",
    "Price Restrictions": "payment",
    "Minimum Commitment": "minimum_commitment",
    "Volume Restriction": "minimum_commitment",
    "Ip Ownership Assignment": "ip",
    "Joint Ip Ownership": "ip",
    "License Grant": "license",
    "Non-Transferable License": "license",
    "Affiliate License-Licensor": "license",
    "Affiliate License-Licensee": "license",
    "Unlimited/All-You-Can-Eat-License": "license",
    "Irrevocable Or Perpetual License": "license",
    "Source Code Escrow": "ip",
    "Post-Termination Services": "termination",
    "Audit Rights": "audit",
    "Uncapped Liability": "liability",
    "Cap On Liability": "liability",
    "Liquidated Damages": "liquidated_damages",
    "Warranty Duration": "warranty",
    "Insurance": "insurance",
    "Covenant Not To Sue": "ip",
    "Third Party Beneficiary": "amendment",
}


def tag_clauses(text: str, section: str = "") -> list[str]:
    """Return taxonomy categories whose keyword patterns fire on the chunk."""
    tags: set[str] = set()
    sec = section.lower()
    for cat, hints in _HEADING_HINTS.items():
        if any(h in sec for h in hints):
            tags.add(cat)
    for cat, patterns in _COMPILED.items():
        hits = sum(len(p.findall(text)) for p in patterns)
        # Very common words (e.g. "terminate", "fees") need >1 hit to count.
        threshold = 2 if cat in {"termination", "payment", "liability", "assignment", "amendment", "license"} else 1
        if hits >= threshold:
            tags.add(cat)
    return sorted(tags)
