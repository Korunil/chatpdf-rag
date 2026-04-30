import os
from config import BATCH_SIZE, K, TOP_K, FETCH_K, MAX_CONTEXT_CHARS
from tools.helper import build_context

def compute_retrieval_confidence(query, docs, reranker):

    docs = [d for d in docs if d and hasattr(d, "page_content")]

    if not docs:
        return 0.0

    pairs = [(query, d.page_content) for d in docs]
    try:
        scores = reranker.predict(pairs, batch_size=BATCH_SIZE)
    except:
        return 0.2

    top3 = sorted(scores, reverse=True)[:3]
    confidence = float(0.7 * max(scores) + 0.3 * (sum(top3) / 3))
    
    return confidence


def rag_answer(query, retriever, memory, rewrite_query, rerank_docs):

    rewritten = rewrite_query(query, memory)

    docs = retriever.retrieve(rewritten, k=K, fetch_k = FETCH_K)

    if not docs:
        return None, []

    clean_docs=[]
    
    for d in docs: 
        if (
            not d or 
            not hasattr(d, "page_content") or
            not isinstance(d.page_content, str) or
            not d.page_content.strip()
        ):
            continue
        clean_docs.append(d)

    if not clean_docs:
        return None, []
    
    top_docs = rerank_docs(rewritten, clean_docs, top_k=TOP_K)
    
    top_docs = sorted(
            top_docs, 
            key=lambda d: getattr(d.metadata, "score", 0) if hasattr(d, "metadata") else 0, 
            reverse=True
        )
    
    top_docs = top_docs[:3]    

    context, citations = build_context(mode="pdf", docs=top_docs, web_text=None)
    
    context = context[:MAX_CONTEXT_CHARS]
    
    return context, top_docs, citations