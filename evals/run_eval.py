"""Measure the bot on a fixed question set.

    python -m evals.run_eval            # uses the backend configured by env vars
    python -m evals.run_eval --verbose  # print every question

Metrics:
  retrieval@1  the right doc section is the top source
  retrieval@3  the right section is among the sources
  answer       the answer contains the expected fact
  refusal      out-of-scope questions get "not found" instead of a made-up answer
"""

import argparse
import json
import sys
from pathlib import Path

from bot.answer import Bot
from bot.main import build_bot

QUESTIONS = Path(__file__).with_name("questions.jsonl")


def evaluate(bot: Bot, verbose: bool = False) -> dict[str, float]:
    cases = [json.loads(line) for line in QUESTIONS.read_text(encoding="utf-8").splitlines() if line.strip()]
    answerable = [c for c in cases if not c.get("unanswerable")]
    unanswerable = [c for c in cases if c.get("unanswerable")]
    hits1 = hits3 = correct = refused = 0

    for case in answerable:
        chunks = bot.retrieve(case["q"])
        ids = [(c.file, c.section) for c in chunks]
        target = (case["file"], case["section"])
        h1, h3 = ids[:1] == [target], target in ids
        answer = bot.ask(case["q"])
        ok = answer.found and case["expect"].lower() in answer.text.lower()
        hits1, hits3, correct = hits1 + h1, hits3 + h3, correct + ok
        if verbose or not ok:
            print(f"{'PASS' if ok else 'FAIL'}  {case['q']}\n      -> {answer.text[:160]}")

    for case in unanswerable:
        answer = bot.ask(case["q"])
        refused += not answer.found
        if verbose or answer.found:
            print(f"{'PASS' if not answer.found else 'FAIL'}  {case['q']} (should refuse)\n      -> {answer.text[:160]}")

    n = len(answerable)
    return {
        "retrieval@1": hits1 / n,
        "retrieval@3": hits3 / n,
        "answer": correct / n,
        "refusal": refused / len(unanswerable),
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()
    scores = evaluate(build_bot(), args.verbose)
    print("\n" + "  ".join(f"{k}: {v:.0%}" for k, v in scores.items()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
