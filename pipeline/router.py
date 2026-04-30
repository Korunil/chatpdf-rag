from llm.generator import safe_generate
from pipeline.retrieval import compute_retrieval_confidence

def classify_intent(query):

    prompt = f"""
Classify query into ONE label:

summary
extract
qa

No explanation.
No punctuation.

Return ONLY one word.

Query:
{query}
"""

    result = safe_generate(prompt).lower()

    if "summary" in result:
        return "summary"
    if "extract" in result:
        return "extract"
    return "qa"


def fast_intent(query):
    q = query.lower()

    if any(p in q for p in ["what", "define", "explain", "how", "where", "when", "which"]):
        return "qa"

    if any(p in q for p in ["list", "extract"]):
        return "extract"
        
    if any(p in q for p in ["summary", "summarize", "brief", "synopsis", "outline"]):
        return "summary"

    return None


def classify_intent_safe(query):
    fast = fast_intent(query)
    if fast:
        return fast
    return classify_intent(query)


def rewrite_query(query, memory):
    history = memory.format()

    if not history:
        return query # first query

    prompt = f"""
Rewrite the user query into standalone question.

Chat History:
{history}

Query:
{query}

Rewritten:
"""

    rewritten = safe_generate(prompt)

    return rewritten if rewritten else query


def decide_source(query, retriever, reranker):

    q = query.lower()
    if any(x in q for x in ["latest", "news", "current", "today", "breaking", "update"]):
        return "internet", 1.0
    
    docs = retriever.retrieve(query, k=5)
    confidence = compute_retrieval_confidence(query, docs, reranker)

    if confidence < 0.3:
        return "internet", confidence

    if confidence > 0.7:
        return "pdf", confidence
    
    return "hybrid", confidence