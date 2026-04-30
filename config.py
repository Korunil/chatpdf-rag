#CACHE FOLDER
CACHE_DIR = "./model_cache"

#LLM TOKEN GENERATION LIMIT
MAX_NEW_TOKENS = 400

MODEL_MAP = {
    "mistral": "mistralai/Mistral-7B-Instruct-v0.2",
    "mistral2": "mistralai/Mistral-7B-Instruct-v0.3",
    "llama": "meta-llama/Meta-Llama-3-8B-Instruct"
}

#EMBEDDING
EMBEDDING_MAP = {
    "fast": "sentence-transformers/all-MiniLM-L6-v2",
    "balanced": "BAAI/bge-base-en-v1.5",
    "better": "BAAI/bge-large-en-v1.5",
    "strong": "intfloat/e5-large-v2",
}
DEVICE_EMBEDDINGS = "cpu"
NORMALIZE_EMBEDDINS = True

#RERANKER
CROSS_ENCODER = "BAAI/bge-reranker-base"
CROSS_ENCODER_LIGHT = "cross-encoder/ms-marco-MiniLM-L-6-v2"
DEVICE_RERANKER = "cpu"

#INGESTION
CHUNK_SIZE=1100
CHUNK_OVERLAP=150

#SUMMARY
SUMMARY_CHUNKS=50
PAGE_CONTENT=500

#ORCHESTRATOR
K=6
FETCH_K=15
TOP_K=6
BATCH_SIZE=32
MAX_CONTEXT_CHARS = 4000

#FILES
MAX_SIZE_MB=10
MAX_FILES=5
TIME_OUT=600

#PRESETS PER TASK
GEN_CONFIGS = {
    "extraction": {
        "temperature": 0.0,
        "top_p": 1.0,
        "do_sample": False,
        "max_new_tokens": 512,
    },
    "summarization": {
        "temperature": 0.3,
        "top_p": 0.9,
        "do_sample": True,
        "max_new_tokens": 512,
    },
    "qa": {
        "temperature": 0.2,
        "top_p": 0.9,
        "do_sample": True,
        "max_new_tokens": 512,
    },
    "fusion": {
        "temperature": 0.2,
        "top_p": 0.9,
        "do_sample": True,
        "max_new_tokens": 512,
    },
    "evaluation": {
        "temperature": 0.0,
        "top_p": 1.0,
        "do_sample": False,
        "max_new_tokens": 256,
    }
}

#DEFAULT
PROD_LLM = "mistral"
PROD_EMBED = "balanced"
PROD_RERANKER = CROSS_ENCODER