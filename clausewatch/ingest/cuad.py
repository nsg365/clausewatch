"""CUAD download + deterministic subset selection.

CUAD v1 (Hendrycks et al., 2021; CC BY 4.0) is pulled from the Hugging Face mirror
`theatticusproject/cuad`: 510 EDGAR contracts as PDFs plus expert span labels for
41 clause categories in SQuAD format.
"""

from __future__ import annotations

import json
import os
import re
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

from clausewatch.config import get_settings

REPO = "theatticusproject/cuad"
LABELS_FILE = "CUAD_v1/CUAD_v1.json"


@dataclass
class CuadContract:
    title: str
    pdf_repo_path: str
    contract_type: str
    n_chars: int
    labels: dict[str, list[str]] = field(default_factory=dict)  # CUAD category -> answer spans

    @property
    def n_labeled(self) -> int:
        return sum(1 for v in self.labels.values() if v)


def raw_dir() -> Path:
    d = get_settings().data_dir / "raw"
    d.mkdir(parents=True, exist_ok=True)
    return d


def load_labels() -> dict:
    path = raw_dir() / "CUAD_v1.json"
    if not path.exists():
        hf_hub_download(REPO, LABELS_FILE, repo_type="dataset", local_dir=raw_dir().parent / "_hf")
        (raw_dir().parent / "_hf" / LABELS_FILE).rename(path)
    return json.loads(path.read_text())


def _normalize_type(folder: str) -> str:
    t = re.sub(r"[_\s]+", " ", folder).strip()
    if t.startswith("Commercial Contracts"):
        return "Commercial"
    t = re.sub(r"\s*(Agreements?|Filing)\s*$", "", t, flags=re.I).strip()
    return t.replace("Non Compete Non Solicit", "Non-Compete/Non-Solicit")


def _category(question_id: str) -> str:
    return question_id.split("__", 1)[1]


def list_contracts() -> list[CuadContract]:
    index_path = raw_dir() / "pdf_index.json"
    if index_path.exists():
        pdfs = json.loads(index_path.read_text())
    else:
        pdfs = [
            f.path
            for f in HfApi().list_repo_tree(
                REPO, repo_type="dataset", path_in_repo="CUAD_v1/full_contract_pdf", recursive=True
            )
            if f.path.lower().endswith(".pdf")
        ]
        index_path.write_text(json.dumps(pdfs))
    by_stem = {os.path.splitext(os.path.basename(p))[0]: p for p in pdfs}

    out = []
    for entry in load_labels()["data"]:
        pdf = by_stem.get(entry["title"])
        if not pdf:
            continue
        para = entry["paragraphs"][0]
        labels = {
            _category(qa["id"]): [a["text"] for a in qa["answers"]] for qa in para["qas"]
        }
        ctype = _normalize_type(pdf.split("/")[-2])
        out.append(CuadContract(entry["title"], pdf, ctype, len(para["context"]), labels))
    return out


def select_subset(n: int = 30, n_holdout: int = 3, min_chars: int = 12_000, max_chars: int = 90_000):
    """Pick `n` contracts round-robin across contract types (richest labels first),
    plus `n_holdout` unseen contracts for the 'new contract' risk demo."""
    pool = [c for c in list_contracts() if min_chars <= c.n_chars <= max_chars]
    by_type: dict[str, list[CuadContract]] = defaultdict(list)
    for c in sorted(pool, key=lambda c: (-c.n_labeled, c.title)):
        by_type[c.contract_type].append(c)
    types = sorted(by_type)
    picked: list[CuadContract] = []
    while len(picked) < n + n_holdout and any(by_type.values()):
        for t in types:
            if by_type[t] and len(picked) < n + n_holdout:
                picked.append(by_type[t].pop(0))
    # Holdout = every k-th pick so it spans types too.
    step = max(1, len(picked) // max(n_holdout, 1))
    holdout = picked[step - 1 :: step][:n_holdout]
    corpus = [c for c in picked if c not in holdout]
    return corpus, holdout


def download_pdf(c: CuadContract) -> Path:
    dest = raw_dir() / "pdfs" / f"{c.title}.pdf"
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        cached = hf_hub_download(REPO, c.pdf_repo_path, repo_type="dataset")
        dest.write_bytes(Path(cached).read_bytes())
    return dest


def save_manifest(corpus: list[CuadContract], holdout: list[CuadContract]) -> Path:
    path = raw_dir() / "subset.json"
    path.write_text(
        json.dumps(
            {
                "corpus": [asdict(c) for c in corpus],
                "holdout": [asdict(c) for c in holdout],
            },
            indent=1,
        )
    )
    return path


def load_manifest() -> tuple[list[CuadContract], list[CuadContract]]:
    data = json.loads((raw_dir() / "subset.json").read_text())
    return [CuadContract(**c) for c in data["corpus"]], [CuadContract(**c) for c in data["holdout"]]
