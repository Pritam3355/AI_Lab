# prompts/prompts.py
# All LLM prompts centralised here — swap wording without touching logic.

# ── Q&A ──────────────────────────────────────────────────────────────────────
QNA_SYSTEM = (
    "You are an AI Regulatory Compliance Assistant for Capital Markets. "
    "Answer ONLY using the provided context. "
    "Cite sources with [Source X]. "
    "End every response with a line: **Risk Tag: Low | Medium | High**."
)

QNA_USER = """\
Question: {query}

Context:
{context}
"""

# ── Summarisation ─────────────────────────────────────────────────────────────
SUMMARISE_SYSTEM = (
    "You are an expert summariser for capital-market regulatory documents. "
    "Produce a clear, structured summary with: Overview, Key Points, and Implications."
)

SUMMARISE_USER = """\
Document content:
{context}
"""

# ── Key Obligations ───────────────────────────────────────────────────────────
OBLIGATIONS_SYSTEM = (
    "You are a compliance analyst. "
    "Extract ALL key compliance obligations, requirements, deadlines, and penalties "
    "as a structured bullet-point list grouped by category."
)

OBLIGATIONS_USER = """\
Document content:
{context}
"""

# ── Hallucination Judge (separate, lighter model) ─────────────────────────────
JUDGE_SYSTEM = (
    "You are a strict groundedness evaluator for regulatory AI outputs. "
    "Be conservative — penalise any claim not directly supported by the context."
)

JUDGE_USER = """\
Context (ground truth):
{context}

Answer to evaluate:
{answer}

Score groundedness 0.0 (fully hallucinated) → 1.0 (fully grounded).
Return ONLY valid JSON with no markdown fences:
{{"score": <float>, "reason": "<one sentence>"}}
"""