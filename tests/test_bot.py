import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest
from fastapi.testclient import TestClient

from bot.answer import NOT_FOUND, Bot, OpenAICompatibleLLM
from bot.main import build_bot, create_app
from bot.retrieval import stem, tokenize
from evals.run_eval import evaluate


@pytest.fixture(scope="module")
def bot():
    return build_bot()


@pytest.fixture(scope="module")
def client(bot):
    return TestClient(create_app(bot))


@pytest.mark.parametrize("words", [("share", "shared", "sharing"), ("sync", "syncing"), ("cancel", "cancellation", "cancelled"), ("encrypted", "encryption")])
def test_stemmer_groups_word_forms(words):
    assert len({stem(w) for w in words}) == 1


def test_product_name_is_ignored_in_queries():
    assert tokenize("nimbus drive refund", frozenset({"nimbus", "drive"})) == ["refund"]


def test_eval_quality_does_not_regress(bot):
    scores = evaluate(bot)
    assert scores["retrieval@3"] >= 0.95
    assert scores["answer"] >= 0.9
    assert scores["refusal"] == 1.0  # never answer out-of-scope questions


def test_answer_cites_its_source(client):
    data = client.post("/api/ask", json={"question": "How much does the Pro plan cost?"}).json()
    assert data["found"] and "$9" in data["answer"] and "[1]" in data["answer"]
    assert data["sources"][0]["file"] == "pricing-and-plans.md"


def test_out_of_scope_question_is_refused(client):
    data = client.post("/api/ask", json={"question": "What is the capital of France?"}).json()
    assert data == {"answer": NOT_FOUND, "found": False, "sources": [], "backend": "extractive"}


def test_question_validation(client):
    assert client.post("/api/ask", json={"question": ""}).status_code == 422
    assert client.post("/api/ask", json={"question": "x" * 501}).status_code == 422


def test_ui_and_health(client):
    assert "Nimbus Drive Help" in client.get("/").text
    assert client.get("/health").json()["status"] == "ok"


class FakeLLM(BaseHTTPRequestHandler):
    """Stands in for any OpenAI-compatible server (Ollama, OpenAI, Groq...)."""

    reply = "The Pro plan costs $9 per month [1]."
    last_request: dict = {}

    def do_POST(self):
        FakeLLM.last_request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        body = json.dumps({"choices": [{"message": {"content": FakeLLM.reply}}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


@pytest.fixture
def fake_llm_url():
    server = HTTPServer(("127.0.0.1", 0), FakeLLM)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_port}/v1"
    server.shutdown()


def test_llm_backend_gets_numbered_context_and_answer_keeps_citations(bot, fake_llm_url):
    llm_bot = Bot(bot.index, llm=OpenAICompatibleLLM(fake_llm_url, "test-model"))
    answer = llm_bot.ask("How much does the Pro plan cost?")
    assert answer.found and answer.backend == "llm:test-model"
    assert [s["n"] for s in answer.sources] == [1]
    prompt = FakeLLM.last_request["messages"][1]["content"]
    assert "[1] Pricing and Plans › Available plans" in prompt


def test_llm_not_found_is_respected(bot, fake_llm_url, monkeypatch):
    monkeypatch.setattr(FakeLLM, "reply", "NOT_FOUND")
    answer = Bot(bot.index, llm=OpenAICompatibleLLM(fake_llm_url, "m")).ask("How much does the Pro plan cost?")
    assert not answer.found and answer.text == NOT_FOUND


def test_out_of_scope_question_never_reaches_the_llm(bot, fake_llm_url):
    FakeLLM.last_request = {}
    answer = Bot(bot.index, llm=OpenAICompatibleLLM(fake_llm_url, "m")).ask("How do I bake sourdough bread?")
    assert not answer.found and FakeLLM.last_request == {}


def test_llm_outage_falls_back_to_extractive(bot):
    dead = OpenAICompatibleLLM("http://127.0.0.1:9/v1", "m", timeout=1)
    answer = Bot(bot.index, llm=dead).ask("How much does the Pro plan cost?")
    assert answer.found and answer.backend == "extractive" and "$9" in answer.text
