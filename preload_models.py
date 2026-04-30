from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer, CrossEncoder

from config import CACHE_DIR, MODEL_MAP, EMBEDDING_MAP, PROD_LLM, PROD_EMBED, PROD_RERANKER
from llm.llm import load_llm
from rag.embedding import get_embeddings
from rag.reranker import get_reranker

def preload_production_models():
    print("Preloading production LLM...")
    load_llm(PROD_LLM, MODEL_MAP, quantization=True)

    print("Preloading embeddings...")
    get_embeddings(EMBEDDING_MAP, PROD_EMBED)

    print("Preloading reranker...")
    get_reranker()

    print("All production models ready.")