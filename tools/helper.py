import re
import os
from config import CHUNK_SIZE

def truncate_sentences(text, max_chars=300):
    sentences = re.split(r'(?<=[.!?]) +', text)
    
    result = ""
    for s in sentences:
        if len(result) + len(s) > max_chars:
            break
        result += s + " "
    
    return result.strip()
    
    
def serialize_docs(docs, max_chars=550):
    
    if not isinstance(docs, list):
        docs = [docs]
    
    serialized = []

    for d in docs[:3]:  # limit here itself
        if not d or not hasattr(d, "page_content"):
            continue

        serialized.append(
            {
            "source": os.path.basename(d.metadata.get("source", "Unknown")),
            "page": d.metadata.get("page") or "?",
            "text": d.page_content[:max_chars],
            }
        )

    return serialized

def build_context(mode, docs=None, web_text=None):
    
    # PDF mode → citations
    if mode == "pdf":
        citations = []
        context_blocks = []

        for i, d in enumerate(docs, start=1):
            source = d.metadata.get("source", "Unknown")
            page = d.metadata.get("page", "?")
            text = d.page_content.strip()

            citations.append({
                "id": i,
                "source": source,
                "page": page,
                "text": text[:CHUNK_SIZE]
            })

            context_blocks.append(f"[{i}] Page {page} | {text}")

        return "\n\n".join(context_blocks), citations

    # Internet mode → plain text
    if mode == "internet":
        return web_text, []

    # Hybrid mode → merge both
    if mode == "hybrid":
        pdf_context, citations = build_context("pdf", docs=docs)
        combined = pdf_context + "\n\n" + web_text
        return combined, citations