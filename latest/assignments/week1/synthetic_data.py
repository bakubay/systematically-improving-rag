"""
Week 1: Synthetic Data Generator (Learning Tool)

This module turns documents into evaluation questions.

Why this exists:
- Chapter 1 teaches that you can measure before you have users.
- Synthetic queries are a fast way to build a baseline.

This file is intentionally small and readable.
"""

import json
from dataclasses import dataclass
from typing import Literal

try:
    from openai import AsyncOpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


QuestionType = Literal["factual", "inferential", "comparative"]


@dataclass(frozen=True)
class SyntheticQuestion:
    question: str
    qtype: QuestionType


async def generate_synthetic_questions(
    *,
    document: str,
    n: int = 5,
    model: str = "gpt-5.2",
) -> list[SyntheticQuestion]:
    """Generate diverse evaluation questions for a single document.

    Notes for learners:
    - "Factual" questions test retrieval of explicit facts.
    - "Inferential" questions test whether the passage supports a conclusion.
    - "Comparative" questions test whether retrieval pulls the right contrast.
    """
    if not OPENAI_AVAILABLE:
        # Fallback: simple heuristic questions (not great, but runnable).
        short = document.strip().splitlines()[0][:80]
        return [
            SyntheticQuestion(question=f"What is this passage about: {short}?", qtype="factual"),
            SyntheticQuestion(question="What is the main idea of the passage?", qtype="inferential"),
            SyntheticQuestion(question="How is this topic different from a related topic?", qtype="comparative"),
        ][:n]

    client = AsyncOpenAI()

    prompt = f"""
You are generating evaluation questions for a retrieval system.

Write exactly {n} questions that can be answered using ONLY the text below.
Make them diverse across types:
- factual
- inferential
- comparative

Return JSON as:
{{"questions": [{{"question": "...", "qtype": "factual|inferential|comparative"}}]}}

Text:
{document}
""".strip()

    resp = await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
    )

    data = json.loads(resp.choices[0].message.content)
    out: list[SyntheticQuestion] = []
    for item in data.get("questions", []):
        q = str(item.get("question", "")).strip()
        t = str(item.get("qtype", "factual")).strip()
        if not q:
            continue
        if t not in ("factual", "inferential", "comparative"):
            t = "factual"
        out.append(SyntheticQuestion(question=q, qtype=t))  # type: ignore[arg-type]
    return out


if __name__ == "__main__":
    # Minimal smoke test (no OpenAI key required).
    import asyncio

    sample_doc = "Python is a programming language. It is used for data science and web development."
    questions = asyncio.run(generate_synthetic_questions(document=sample_doc, n=3))
    for q in questions:
        print(f"- ({q.qtype}) {q.question}")

