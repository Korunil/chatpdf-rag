from langchain_huggingface import HuggingFaceEmbeddings
from config import BATCH_SIZE, DEVICE_EMBEDDINGS, NORMALIZE_EMBEDDINS, CACHE_DIR
from cache_store import _EMBED_CACHE

def get_embeddings(embedding_map, embed_choice):
    if embed_choice in _EMBED_CACHE:
        return _EMBED_CACHE[embed_choice]
    
    embedding_name = embedding_map.get(embed_choice, embedding_map["balanced"])

    embeddings = HuggingFaceEmbeddings(
        model_name=embedding_name,
        cache_folder=CACHE_DIR,
        model_kwargs={"device": DEVICE_EMBEDDINGS},
        encode_kwargs={"batch_size": BATCH_SIZE, 
                        "normalize_embeddings" : NORMALIZE_EMBEDDINS
                        },
        )
     
    _EMBED_CACHE[embed_choice] = embeddings
    return embeddings