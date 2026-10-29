"""Markdown chunking and BM25 search. Pure Python: no API key, no model download."""

import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

STOPWORDS = set(
    """a an and are as at be by can could do does did for from has have how i if in into is it its
    me my of on or our so than that the their them then there these they this to up us was we what
    when where which who why will with would you your yours get got am also any about should
    too much many happen happens need want just like long""".split()
)


def tokenize(text: str, extra_stopwords: frozenset[str] = frozenset()) -> list[str]:
    """Lowercase words, minus stopwords, crudely stemmed.

    Suffix stripping plus truncating to 6 characters is a cheap stemmer that groups
    share/shared/sharing, sync/syncing, cancel/cancellation and encrypt/encryption
    without any language library.
    """
    tokens = []
    for word in re.findall(r"[a-z0-9]+", text.lower()):
        if word in STOPWORDS or word in extra_stopwords or word.isdigit():
            continue  # bare numbers ("2 weeks" vs "14 days") rarely match docs literally
        tokens.append(stem(word))
    return tokens


def stem(word: str) -> str:
    if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
        word = word[:-1]
    if len(word) > 5 and word.endswith("ing"):
        word = word[:-3]
    elif len(word) > 4 and word.endswith("ed"):
        word = word[:-2]
    if len(word) > 3 and word.endswith("e"):
        word = word[:-1]
    return word[:6]


def expand_synonyms(text: str, synonyms: dict[str, str]) -> str:
    """Rewrite customer wording into the docs' wording ("money back" -> "refund")."""
    text = text.lower()
    for phrase, replacement in synonyms.items():
        text = re.sub("(?<![a-z0-9])" + re.escape(phrase) + "(?![a-z0-9])", replacement, text)
    return text


@dataclass(frozen=True)
class Chunk:
    id: int
    file: str
    title: str  # document H1
    section: str  # H2
    text: str

    @property
    def label(self) -> str:
        return f"{self.title} › {self.section}"


def load_chunks(docs_dir: Path) -> list[Chunk]:
    """One chunk per '## ' section, which keeps each answer tied to a citable heading."""
    chunks: list[Chunk] = []
    for path in sorted(docs_dir.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        title_match = re.search(r"^# (.+)$", text, re.M)
        title = title_match.group(1).strip() if title_match else path.stem
        for block in re.split(r"^## ", text, flags=re.M)[1:]:
            heading, _, body = block.partition("\n")
            body = " ".join(body.split())
            if body:
                chunks.append(Chunk(len(chunks), path.name, title, heading.strip(), body))
    if not chunks:
        raise ValueError(f"No '## ' sections found in {docs_dir}/*.md")
    return chunks


class BM25Index:
    def __init__(self, chunks: list[Chunk], extra_stopwords: frozenset[str] = frozenset(),
                 synonyms: dict[str, str] | None = None, k1=1.5, b=0.75):
        self.chunks, self.k1, self.b, self.extra = chunks, k1, b, extra_stopwords
        self.synonyms = synonyms or {}
        # The section heading is counted twice: headings are the best summary of what a chunk is about.
        self.docs = [Counter(tokenize(f"{c.section} {c.section} {c.text}", extra_stopwords)) for c in chunks]
        self.lengths = [sum(d.values()) for d in self.docs]
        self.avg_len = sum(self.lengths) / len(self.lengths)
        df = Counter(term for d in self.docs for term in d)
        n = len(chunks)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}
        self.unknown_idf = math.log(1 + (n + 0.5) / 0.5)  # a word the docs never use is maximally specific

    def query_terms(self, query: str) -> list[str]:
        return list(dict.fromkeys(tokenize(expand_synonyms(query, self.synonyms), self.extra)))

    def search(self, query: str, k: int = 3) -> list[tuple[Chunk, float, float]]:
        """Return (chunk, bm25 score, coverage).

        Coverage is the idf-weighted share of query terms found in the chunk, so missing a
        specific word ("student" discount) counts far more than missing a generic one.
        """
        terms = self.query_terms(query)
        if not terms:
            return []
        weights = {t: self.idf.get(t, self.unknown_idf) for t in terms}
        total = sum(weights.values())
        results = []
        for chunk, doc, length in zip(self.chunks, self.docs, self.lengths):
            score, matched = 0.0, 0.0
            for t in terms:
                tf = doc.get(t, 0)
                if tf:
                    matched += weights[t]
                    score += self.idf[t] * tf * (self.k1 + 1) / (tf + self.k1 * (1 - self.b + self.b * length / self.avg_len))
            if score > 0:
                results.append((chunk, score, matched / total))
        results.sort(key=lambda r: r[1], reverse=True)
        return results[:k]
