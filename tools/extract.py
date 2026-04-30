from config import K, TOP_K, GEN_CONFIGS, FETCH_K

def extract_all(retriever, query, rerank_docs, safe_generate):

    docs = retriever.retrieve(query, k=2*K, fetch_k = FETCH_K)
    
    docs = [d for d in docs if d and hasattr(d, "page_content")]

    top_docs = rerank_docs(query, docs, top_k=3*TOP_K)
    
    if not top_docs:
        return "❌ No relevant content found for extraction.", []
    
    text = "\n".join([
        d.page_content for d in top_docs
        if d and hasattr(d, "page_content")
        ])

    prompt = f"""
<s>[INST]
You extract structured information.

Task:
Extract ALL relevant items for the query.

Rules:
- Return ONLY bullet points
- Be concise
- Do not add explanations
- If nothing found, return "No relevant items"

Query:
{query}

<<<TEXT>>>
{text}
<<<END_TEXT>>>
[/INST]
"""

    return safe_generate(prompt, **GEN_CONFIGS["summarization"]), top_docs