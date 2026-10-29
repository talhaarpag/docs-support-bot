"""Turn retrieved chunks into a cited answer, or an honest "not in the docs"."""

import json
import logging
import re
import urllib.request
from dataclasses import dataclass, field

from .retrieval import BM25Index, Chunk, tokenize

log = logging.getLogger("bot")

NOT_FOUND = "I couldn't find that in the documentation. Please contact support@nimbusdrive.example and a person will help."


@dataclass
class Answer:
    text: str
    found: bool
    sources: list[dict] = field(default_factory=list)
    backend: str = "extractive"


def _sources(chunks: list[Chunk]) -> list[dict]:
    return [
        {"n": i, "file": c.file, "title": c.title, "section": c.section, "snippet": c.text[:220]}
        for i, c in enumerate(chunks, 1)
    ]


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


class Bot:
    def __init__(self, index: BM25Index, min_score: float = 2.0, min_coverage: float = 0.5, llm: "OpenAICompatibleLLM | None" = None):
        self.index, self.min_score, self.min_coverage, self.llm = index, min_score, min_coverage, llm

    def retrieve(self, question: str) -> list[Chunk]:
        hits = self.index.search(question, k=3)
        if not hits:
            return []
        top_chunk, top_score, top_coverage = hits[0]
        # Refusal gate, applied before any LLM sees the question: weak matches are not answered.
        if top_score < self.min_score or top_coverage < self.min_coverage:
            log.info("refused %r (score=%.2f coverage=%.2f)", question, top_score, top_coverage)
            return []
        return [c for c, score, _ in hits if score >= 0.5 * top_score]

    def ask(self, question: str) -> Answer:
        chunks = self.retrieve(question)
        if not chunks:
            return Answer(NOT_FOUND, found=False, backend=self.llm.name if self.llm else "extractive")
        if self.llm:
            try:
                return self.llm.answer(question, chunks)
            except Exception as exc:  # provider down / quota / timeout: degrade instead of failing
                log.warning("LLM failed, falling back to extractive: %s", exc)
        return self.extractive(question, chunks)

    def extractive(self, question: str, chunks: list[Chunk], max_sentences: int = 3) -> Answer:
        """Pick the sentences that best match the question, quoted verbatim with citations."""
        terms = set(self.index.query_terms(question))
        scored = []
        for rank, chunk in enumerate(chunks):
            for pos, sentence in enumerate(_sentences(chunk.text)):
                words = set(tokenize(sentence, self.index.extra))
                score = sum(self.index.idf.get(t, 0) for t in terms & words)
                if score > 0:
                    # Small boost for the best chunk so its sentences win ties.
                    scored.append((score * (1.2 if rank == 0 else 1.0), rank, pos, sentence))

        if not scored:  # the section matched on its heading only: quote its opening instead
            scored = [(1.0, 0, pos, s) for pos, s in enumerate(_sentences(chunks[0].text)[:2])]
        top = max(s[0] for s in scored)
        # Other sections only contribute strong matches, so "Pro plan cost" doesn't pull in a Trash sentence.
        keep = [s for s in scored if s[1] == 0 or s[0] >= 0.8 * top]
        best = sorted(keep, key=lambda s: s[0], reverse=True)[:max_sentences]
        best.sort(key=lambda s: (s[1], s[2]))  # restore reading order
        used = sorted({rank for _, rank, _, _ in best})
        renumber = {rank: i for i, rank in enumerate(used, 1)}
        text = " ".join(f"{sentence} [{renumber[rank]}]" for _, rank, _, sentence in best)
        return Answer(text, found=True, sources=_sources([chunks[r] for r in used]))


SYSTEM_PROMPT = """You are a customer support assistant. Answer ONLY using the numbered documentation excerpts.
Cite every claim with its excerpt number in square brackets, like [1].
If the excerpts do not contain the answer, reply exactly: NOT_FOUND
Be concise: at most 4 sentences. Do not mention the excerpts themselves."""


class OpenAICompatibleLLM:
    """Any /v1/chat/completions endpoint: Ollama (local, free), OpenAI, Groq, OpenRouter, etc."""

    def __init__(self, base_url: str, model: str, api_key: str | None = None, timeout: float = 60):
        self.base_url, self.model, self.api_key, self.timeout = base_url.rstrip("/"), model, api_key, timeout
        self.name = f"llm:{model}"

    def answer(self, question: str, chunks: list[Chunk]) -> Answer:
        context = "\n\n".join(f"[{i}] {c.label}\n{c.text}" for i, c in enumerate(chunks, 1))
        body = {
            "model": self.model,
            "temperature": 0,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Documentation:\n{context}\n\nQuestion: {question}"},
            ],
        }
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        req = urllib.request.Request(f"{self.base_url}/chat/completions", json.dumps(body).encode(), headers)
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            text = json.loads(resp.read())["choices"][0]["message"]["content"].strip()

        if "NOT_FOUND" in text:
            return Answer(NOT_FOUND, found=False, backend=self.name)
        cited = sorted({int(n) for n in re.findall(r"\[(\d+)\]", text) if 1 <= int(n) <= len(chunks)})
        return Answer(text, found=True, sources=_sources(chunks) if not cited else
                      [s for s in _sources(chunks) if s["n"] in cited], backend=self.name)
