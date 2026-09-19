"""Catalog of contract risk patterns scanned by the risk-flagging node."""

from __future__ import annotations

from dataclasses import dataclass, field

SEVERITY_WEIGHT = {"high": 3, "medium": 2, "low": 1}


@dataclass(frozen=True)
class RiskPattern:
    id: str
    name: str
    severity: str
    categories: list[str]
    query: str
    criteria: str
    examples: list[str] = field(default_factory=list)


RISK_PATTERNS: list[RiskPattern] = [
    RiskPattern(
        id="uncapped_liability",
        name="Unlimited / uncapped liability",
        severity="high",
        categories=["liability", "indemnification"],
        query="limitation of liability cap on damages aggregate liability shall not exceed; liability unlimited",
        criteria=(
            "Flag if a party's liability is expressly unlimited, if there is no cap on direct damages, "
            "or if carve-outs (e.g. for indemnification, breach of confidentiality, IP infringement) are "
            "broad enough to make liability effectively uncapped for a material category of claims."
        ),
    ),
    RiskPattern(
        id="auto_renewal",
        name="Auto-renewal without adequate notice",
        severity="medium",
        categories=["renewal", "term"],
        query="agreement shall automatically renew for successive terms unless notice of non-renewal",
        criteria=(
            "Flag if the agreement renews automatically (evergreen) AND either no opt-out exists, the "
            "non-renewal notice window is long (60+ days) or easy to miss, or only one party can prevent renewal."
        ),
    ),
    RiskPattern(
        id="one_sided_indemnification",
        name="One-sided indemnification",
        severity="high",
        categories=["indemnification"],
        query="shall indemnify defend and hold harmless from any and all claims losses damages",
        criteria=(
            "Flag if only one party gives an indemnity (no reciprocal indemnity from the other party), or the "
            "indemnity is materially broader for one side (e.g. covers the indemnitee's own negligence)."
        ),
    ),
    RiskPattern(
        id="one_sided_termination",
        name="One-sided termination for convenience",
        severity="medium",
        categories=["termination"],
        query="may terminate this agreement at any time for any reason or no reason upon written notice for convenience",
        criteria=(
            "Flag if one party may terminate for convenience / without cause and the other party has no "
            "equivalent right, or if termination can occur on very short notice without compensation."
        ),
    ),
    RiskPattern(
        id="broad_ip_assignment",
        name="Broad IP assignment",
        severity="high",
        categories=["ip"],
        query="hereby assigns all right title and interest in intellectual property work product inventions",
        criteria=(
            "Flag if a party assigns all IP (especially pre-existing/background IP, improvements, or "
            "IP developed outside the scope of the agreement) to the other party, or IP is work-made-for-hire "
            "without a license-back."
        ),
    ),
    RiskPattern(
        id="restrictive_non_compete",
        name="Restrictive non-compete / exclusivity",
        severity="medium",
        categories=["exclusivity", "non_solicit"],
        query="shall not directly or indirectly compete exclusive distributor non-competition restricted territory",
        criteria=(
            "Flag if a party is barred from competing, dealing with competitors, or selling competing "
            "products, especially with broad territory, long duration, or survival after termination."
        ),
    ),
    RiskPattern(
        id="change_of_control",
        name="Change-of-control trigger",
        severity="medium",
        categories=["assignment", "termination"],
        query="change of control merger acquisition consent terminate assignment by operation of law",
        criteria=(
            "Flag if a merger, acquisition or change of control of a party requires consent, is deemed an "
            "assignment, or gives the counterparty a termination right."
        ),
    ),
    RiskPattern(
        id="minimum_commitment",
        name="Minimum purchase / take-or-pay commitment",
        severity="medium",
        categories=["minimum_commitment", "payment"],
        query="minimum purchase requirement minimum annual quantity shortfall payment failure to meet minimum",
        criteria=(
            "Flag if a party must buy/sell/pay a minimum amount, with penalties, loss of exclusivity, or "
            "termination if the minimum is missed."
        ),
    ),
    RiskPattern(
        id="liquidated_damages",
        name="Liquidated damages / penalties",
        severity="medium",
        categories=["liquidated_damages", "payment"],
        query="liquidated damages penalty shall pay as damages and not as a penalty",
        criteria="Flag if the contract imposes liquidated damages, penalties, or fixed damages amounts.",
    ),
    RiskPattern(
        id="unilateral_amendment",
        name="Unilateral changes to terms or pricing",
        severity="medium",
        categories=["amendment", "payment"],
        query="may change modify prices terms at any time in its sole discretion upon notice",
        criteria=(
            "Flag if one party may unilaterally change prices, fees, specifications, or contract terms "
            "without the other party's written agreement."
        ),
    ),
]

PATTERNS_BY_ID = {p.id: p for p in RISK_PATTERNS}
