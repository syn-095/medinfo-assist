"""Load approved documents, chunk them by section and retrieve with BM25.

Each `## <section> <heading>` block in an approved markdown file is one chunk.
The chunk ID ("VLX-SMPC-001 §4.2") is also the citation unit, so the citation
check is a set comparison between cited IDs and retrieved IDs.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from rank_bm25 import BM25Okapi

# Minimum BM25 score of the best chunk for retrieval to count as "confident".
# Tuned on the 30 test enquiries (see NOTES.md). Below this we do not draft.
MIN_CONFIDENT_SCORE = 3.0
# Chunks scoring below this fraction of the top score are not passed on.
RELATIVE_CUTOFF = 0.35

STOPWORDS = set("""
a an the and or but if of to in on for with without at by from as is are was were be been being am
do does did can could should would will shall may might must i me my we our you your he she they them
his her their it its this that these those there here what which who whom whose when where why how
any some all about into than then so not no yes also just very much more most such only own same
too out up down over under again further once have has had having get got please tell know need want
like ok okay is it hi hello dear thanks thank regards one would anyone something
""".split())

# Small domain lexicon so patient wording finds label wording.
SYNONYMS = {
    "pregnant": ["pregnancy"], "pregnancy": ["pregnant"],
    "breastfeed": ["breast", "milk", "lactation"], "breastfeeding": ["breast", "milk", "lactation"],
    "store": ["storage"], "storage": ["store"], "condition": ["store"], "storing": ["storage", "store"], "kept": ["store", "storage"], "keep": ["store"],
    "fridge": ["refrigerator"], "travel": ["travelling"], "travelling": ["travel"], "flying": ["flight", "travelling"],
    "side": ["undesirable"], "effect": ["undesirable", "common"], "adverse": ["undesirable", "reaction"],
    "dose": ["dosing", "posology"], "dosage": ["dose", "posology"], "dosing": ["dose", "posology"],
    "forgot": ["missed", "forgotten"], "forget": ["missed"], "miss": ["missed"],
    "kidney": ["renal"], "liver": ["hepatic"],
    "interaction": ["interact"], "interact": ["interaction"], "together": ["interaction", "concomitant"],
    "half": ["split", "halves"], "cut": ["split"], "swallow": ["swallowed", "swallowing"],
    "coeliac": ["gluten"], "celiac": ["gluten"], "vegan": ["animal", "vegetarian"],
    "contraindicated": ["contraindications"], "contraindication": ["contraindications"],
    "throw": ["disposal", "disposed"], "dispose": ["disposal", "disposed"], "bin": ["disposal", "sharps"],
    "blood": ["hypertension"], "pressure": ["hypertension"],
    "use": ["indicated"], "used": ["indicated"], "treat": ["indicated", "treatment"], "prevention": ["indicated"],
    "accidentally": ["overdose"], "many": ["overdose"], "overdosed": ["overdose"],
}


def stem(tok: str) -> str:
    if len(tok) > 4 and tok.endswith("ies"):
        return tok[:-3] + "y"
    if len(tok) > 3 and tok.endswith("s") and not tok.endswith("ss"):
        return tok[:-1]
    return tok


def tokenize(text: str, expand: bool = False) -> list[str]:
    raw = re.findall(r"[a-z0-9]+(?:\.[0-9]+)?", text.lower())
    toks = [t for t in raw if t not in STOPWORDS]
    if expand:
        extra = [s for t in toks for s in SYNONYMS.get(t, [])]
        toks += extra
    return [stem(t) for t in toks]


@dataclass
class Chunk:
    chunk_id: str        # "VLX-SMPC-001 §4.2"
    doc_id: str
    section: str         # "4.2" or "Q1"
    heading: str
    text: str
    product: str
    version: str
    approval_date: str
    title: str
    substance: str = ""  # active substance, filled in from §1 of the product's SmPC


@dataclass
class Retrieved:
    chunk: Chunk
    score: float


@dataclass
class RetrievalResult:
    query: str
    products: list[str]                 # products named in the enquiry
    hits: list[Retrieved] = field(default_factory=list)
    top_score: float = 0.0

    @property
    def confident(self) -> bool:
        return bool(self.hits) and self.top_score >= MIN_CONFIDENT_SCORE

    @property
    def ids(self) -> set[str]:
        return {h.chunk.chunk_id for h in self.hits}

    @property
    def texts(self) -> dict[str, str]:
        return {h.chunk.chunk_id: h.chunk.text for h in self.hits}


def _parse_doc(path: Path) -> list[Chunk]:
    text = path.read_text(encoding="utf-8")
    m = re.match(r"---\s*\n(.*?)\n---\s*\n(.*)", text, re.S)
    if not m:
        raise ValueError(f"{path.name}: missing front matter")
    meta = dict(
        (k.strip(), v.strip()) for k, v in
        (line.split(":", 1) for line in m.group(1).splitlines() if ":" in line)
    )
    if meta.get("status", "").lower() != "approved":
        return []  # only approved documents are ever retrievable
    chunks = []
    for sec in re.split(r"\n(?=## )", "\n" + m.group(2)):
        sec = sec.strip()
        if not sec.startswith("## "):
            continue
        header, _, body = sec.partition("\n")
        header = header[3:].strip()
        section, _, heading = header.partition(" ")
        chunks.append(Chunk(
            chunk_id=f"{meta['doc_id']} §{section}", doc_id=meta["doc_id"], section=section,
            heading=heading, text=body.strip(), product=meta["product"], version=meta["version"],
            approval_date=meta["approval_date"], title=meta["title"],
        ))
    return chunks


class Retriever:
    def __init__(self, doc_dir: str | Path):
        self.chunks: list[Chunk] = []
        for p in sorted(Path(doc_dir).glob("*.md")):
            self.chunks.extend(_parse_doc(p))
        self.by_id = {c.chunk_id: c for c in self.chunks}
        self.products = sorted({c.product for c in self.chunks})
        # Aliases: product name + active substance ("The active substance is X").
        self.aliases: dict[str, set[str]] = {p: {p.lower()} for p in self.products}
        for c in self.chunks:
            m = re.search(r"active substance is (\w+)", c.text)
            if m:
                self.aliases[c.product].add(m.group(1).lower())
        substance = {}
        for c in self.chunks:
            m = re.search(r"active substance is (\w+)", c.text)
            if m:
                substance[c.product] = m.group(1)
        for c in self.chunks:
            c.substance = substance.get(c.product, "")
        self._alias_tokens = {stem(a) for s in self.aliases.values() for a in s}
        # Headings weighted x2: they summarise the section in label vocabulary.
        self._corpus = [tokenize(f"{c.heading} {c.heading} {c.text}") for c in self.chunks]
        self.bm25 = BM25Okapi(self._corpus)

    def detect_products(self, text: str) -> list[str]:
        low = text.lower()
        return [p for p, al in self.aliases.items() if any(re.search(rf"\b{a}\b", low) for a in al)]

    def product_text(self, product: str) -> str:
        return " ".join(c.text for c in self.chunks if c.product == product).lower()

    def search(self, query: str, k: int = 4) -> RetrievalResult:
        products = self.detect_products(query)
        # Product names decide *which* documents are searched; they are removed
        # from the scoring query so that "Veloxitab" alone cannot make a chunk relevant.
        q = [t for t in tokenize(query, expand=True) if t not in self._alias_tokens]
        scores = self.bm25.get_scores(q) if q else [0.0] * len(self.chunks)
        ranked = sorted(
            (Retrieved(c, float(s)) for c, s in zip(self.chunks, scores)
             if not products or c.product in products),
            key=lambda r: r.score, reverse=True,
        )
        top = ranked[0].score if ranked else 0.0
        hits = [r for r in ranked[:k] if r.score > 0 and r.score >= RELATIVE_CUTOFF * top]
        return RetrievalResult(query=query, products=products, hits=hits, top_score=top)
