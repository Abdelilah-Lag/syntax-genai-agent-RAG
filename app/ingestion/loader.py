"""Load the markdown/text corpus and build a small drug catalog."""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

_FILE_RE = re.compile(r"^(?P<kind>dailymed|medlineplus)-(?P<drug>[a-z0-9-]+)\.(?:md|txt)$", re.I)
_SOURCE_RE = re.compile(r"^Source:\s*(\S+)", re.M)

# Brand names / alternative names, used to recognise drugs in free-text questions.
DRUG_ALIASES: dict[str, tuple[str, ...]] = {
    "acetaminophen": ("paracetamol", "tylenol"),
    "albuterol": ("salbutamol", "proair", "ventolin"),
    "amoxicillin": ("amoxil",),
    "apixaban": ("eliquis",),
    "aspirin": ("acetylsalicylic acid",),
    "atorvastatin": ("lipitor",),
    "ibuprofen": ("advil", "motrin"),
    "metformin": ("glucophage",),
    "sertraline": ("zoloft",),
    "warfarin": ("coumadin", "jantoven"),
}

# Class-level terms that labels use instead of naming the other drug (e.g. "NSAIDs", "SSRIs").
CLASS_TERMS: dict[str, tuple[str, ...]] = {
    "warfarin": ("anticoagulant", "anticoagulants", "vitamin k antagonist"),
    "apixaban": ("anticoagulant", "anticoagulants", "doac", "factor xa"),
    "sertraline": ("ssri", "ssris", "selective serotonin reuptake", "serotonergic"),
    "ibuprofen": ("nsaid", "nsaids", "nonsteroidal anti-inflammatory"),
    "aspirin": ("nsaid", "nsaids", "salicylate", "salicylates", "antiplatelet"),
    "atorvastatin": ("statin", "statins", "hmg-coa"),
    "amoxicillin": ("antibiotic", "antibiotics", "penicillin"),
    "albuterol": ("beta-agonist", "beta-adrenergic", "beta agonist"),
}


@dataclass(frozen=True)
class Document:
    doc_id: str  # file name, used as the citation key
    source_type: str  # dailymed | medlineplus | glossary | other
    drug: str | None
    title: str
    url: str | None
    text: str


def load_corpus(corpus_dir: Path) -> list[Document]:
    paths = sorted(p for p in Path(corpus_dir).glob("*") if p.suffix.lower() in {".md", ".txt"})
    if not paths:
        raise FileNotFoundError(f"No .md/.txt files found in {corpus_dir}")
    docs: list[Document] = []
    for path in paths:
        text = path.read_text(encoding="utf-8", errors="replace").replace("\r\n", "\n").replace("\r", "\n")
        match = _FILE_RE.match(path.name)
        if match:
            source_type, drug = match["kind"].lower(), match["drug"].lower()
        elif path.stem.lower() == "glossary":
            source_type, drug = "glossary", None
        else:
            source_type, drug = "other", None
        title_match = re.search(r"^#\s+(.+)$", text, re.M)
        url_match = _SOURCE_RE.search(text)
        docs.append(
            Document(
                doc_id=path.name,
                source_type=source_type,
                drug=drug,
                title=title_match.group(1).strip() if title_match else path.stem,
                url=url_match.group(1) if url_match else None,
                text=text,
            )
        )
    return docs


class DrugCatalog:
    """Maps names / brand names found in text to canonical drug names present in the corpus."""

    def __init__(self, drugs: Iterable[str]):
        self.drugs: list[str] = sorted({d.lower() for d in drugs if d})
        self._patterns: list[tuple[str, re.Pattern[str]]] = []
        for drug in self.drugs:
            names = (drug, *DRUG_ALIASES.get(drug, ()))
            pattern = re.compile(r"\b(?:" + "|".join(re.escape(n) for n in names) + r")\b", re.I)
            self._patterns.append((drug, pattern))

    def canonical(self, name: str | None) -> str | None:
        if not name:
            return None
        hits = self.detect(str(name))
        return hits[0] if hits else None

    def detect(self, text: str) -> list[str]:
        found: list[tuple[int, str]] = []
        for drug, pattern in self._patterns:
            m = pattern.search(text)
            if m:
                found.append((m.start(), drug))
        return [d for _, d in sorted(found)]

    def search_terms(self, drug: str) -> list[str]:
        return [drug, *DRUG_ALIASES.get(drug, ())]

    def class_terms(self, drug: str) -> list[str]:
        return list(CLASS_TERMS.get(drug, ()))
