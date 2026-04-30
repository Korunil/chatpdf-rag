from config import GEN_CONFIGS
from llm.generator import safe_generate

from tools.summary import generate_summary
from tools.extract import extract_all
from tools.web_search import internet_answer

from pipeline.evaluator import run_evaluation
from pipeline.retrieval import rag_answer
from prompt.prompts import build_rag_prompt, fusion_prompt
from pipeline.router import classify_intent_safe, rewrite_query, decide_source

def process_query(query, 
                  retriever, 
                  memory, 
                  mode, 
                  rerank_docs, 
                  safe_generate,
                  chunks,
                  reranker
                  ):
    
    evaluation = {}
    
    if mode == "auto":
        route, confidence = decide_source(query, retriever, reranker)
    else:
        route = mode
        confidence = confidence = {
            "pdf": 0.9,
            "internet": 0.8,
            "hybrid": 0.85
        }.get(mode, 1.0)
    
    if route == "internet":
        answer = internet_answer(query, safe_generate)
        return answer, [], {
            "context": "",
            "route": route,
            "confidence": confidence,
            "source": "internet",
            "citations": []
        }

    intent = classify_intent_safe(query)

    if intent == "summary":
        answer, docs = generate_summary(chunks, safe_generate)
        return answer, docs, {
            "context": "",
            "route": route,
            "confidence": confidence,
            "source": "pdf",
            "citations": []
        }

    if intent == "extract":
        answer, docs = extract_all(retriever, query, rerank_docs, safe_generate)
        return answer, docs, {
            "context": "",
            "route": route,
            "confidence": confidence,
            "source": "pdf",
            "citations": []
        }

    context, docs, citations = rag_answer(query, retriever, memory, rewrite_query, rerank_docs)

    if not context:
        answer = internet_answer(query, safe_generate)
        return answer, [], {
            "context": "",
            "route": route,
            "confidence": confidence,
            "source": "internet",
            "citations": []
        }

    prompt = build_rag_prompt(query, context)
    answer = safe_generate(prompt, **GEN_CONFIGS["qa"])

    if route == "hybrid":
        web_info = internet_answer(query, safe_generate)
        hybrid_prompt = fusion_prompt(answer, web_info)
        hybrid_answer = safe_generate(hybrid_prompt, **GEN_CONFIGS["fusion"])
        evaluation = run_evaluation(query, hybrid_answer, context, web=web_info)

        return hybrid_answer, docs, {
            "pdf": answer,
            "web": web_info,
            "route": route,
            "confidence": confidence,
            "evaluation": evaluation
        }
    if confidence < 0.5:
        evaluation = run_evaluation(query, answer, context, web=None)

    return answer, docs, {
    "context": context,
    "route": route,
    "confidence": confidence,
    "source": "pdf",
    "evaluation": evaluation,
    "citations": citations
    }