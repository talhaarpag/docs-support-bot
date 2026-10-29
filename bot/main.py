import json
import logging
import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from .answer import Bot, OpenAICompatibleLLM
from .retrieval import BM25Index, load_chunks

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("bot")

ROOT = Path(__file__).resolve().parent.parent


def build_bot() -> Bot:
    """Configured from env vars, so the same code runs keyless or with any LLM provider."""
    docs = Path(os.getenv("DOCS_DIR", ROOT / "docs"))
    product_words = frozenset(os.getenv("PRODUCT_STOPWORDS", "nimbus,drive").lower().split(","))
    synonyms_file = docs / "synonyms.json"
    synonyms = json.loads(synonyms_file.read_text(encoding="utf-8")) if synonyms_file.exists() else {}
    index = BM25Index(load_chunks(docs), extra_stopwords=product_words, synonyms=synonyms)

    llm = None
    if os.getenv("BOT_BACKEND", "extractive") == "llm":
        llm = OpenAICompatibleLLM(
            base_url=os.getenv("LLM_BASE_URL", "http://localhost:11434/v1"),  # Ollama's default
            model=os.getenv("LLM_MODEL", "llama3.2:3b"),
            api_key=os.getenv("LLM_API_KEY") or None,
        )
    log.info("loaded %d chunks from %s; backend=%s", len(index.chunks), docs, llm.name if llm else "extractive")
    return Bot(index, llm=llm)


class Question(BaseModel):
    question: str = Field(min_length=2, max_length=500)


def create_app(bot: Bot | None = None) -> FastAPI:
    bot = bot or build_bot()
    app = FastAPI(title="Docs Support Bot", description="Answers questions from your documentation, with citations.")

    @app.get("/", include_in_schema=False)
    def ui():
        return FileResponse(ROOT / "static" / "index.html")

    @app.get("/health")
    def health():
        return {"status": "ok", "chunks": len(bot.index.chunks), "backend": bot.llm.name if bot.llm else "extractive"}

    @app.post("/api/ask")
    def ask(q: Question):
        answer = bot.ask(q.question.strip())
        log.info("Q=%r found=%s sources=%s", q.question, answer.found, [s["file"] for s in answer.sources])
        return {"answer": answer.text, "found": answer.found, "sources": answer.sources, "backend": answer.backend}

    return app
