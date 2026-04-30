from sentence_transformers import CrossEncoder
from config import CROSS_ENCODER, TOP_K, BATCH_SIZE, DEVICE_RERANKER, CACHE_DIR, PROD_RERANKER
from cache_store import _RERANKER_CACHE

def get_reranker(name=PROD_RERANKER):
    global _RERANKER_CACHE

    if _RERANKER_CACHE is not None:
        return _RERANKER_CACHE

    _RERANKER_CACHE = CrossEncoder(
        name,
        device=DEVICE_RERANKER,
        cache_folder=CACHE_DIR
    )

    return _RERANKER_CACHE

def rerank_docs(query, docs, top_k=2*TOP_K):

    docs = [d for d in docs if d and hasattr(d, "page_content")]
    if not docs:
        return []

    reranker = get_reranker()
    
    pairs = [(query, d.page_content) for d in docs]
	
    try:
        scores = reranker.predict(pairs, batch_size=BATCH_SIZE, convert_to_numpy=True)
    except Exception:
        return docs[:top_k]

    scored = list(zip(docs, scores))
    scored.sort(key=lambda x: x[1], reverse=True)

    return [d for d, _ in scored[:top_k] if d]