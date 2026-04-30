from langchain_community.vectorstores import FAISS
from config import K, FETCH_K

class VectorRetriever:
    def __init__(self, documents, embeddings):
        self.vectorstore = FAISS.from_documents(documents, embeddings)

    def retrieve(self, query, k=K, fetch_k=FETCH_K, use_mmr=True):
        if use_mmr:
            return self.vectorstore.max_marginal_relevance_search(
                query, k=k, fetch_k=fetch_k
            )
        return self.vectorstore.similarity_search(query, k=k)