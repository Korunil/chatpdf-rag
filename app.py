import chainlit as cl
from chainlit.input_widget import Select, RadioGroup, Checkbox
import torch, gc, time, asyncio

from config import *
from memory.memory import ChatMemory
from rag.ingestion import load_pdf
from rag.retriever import VectorRetriever
from rag.reranker import get_reranker, rerank_docs
from rag.embedding import get_embeddings

from llm.generator import safe_generate
from llm.llm import load_llm
from pipeline.retrieval import rag_answer
from pipeline.orchestrator import process_query
from tools.helper import truncate_sentences, serialize_docs

import threading
from preload_models import preload_production_models

threading.Thread(target=preload_production_models).start()

class ProgressUI:
    def __init__(self):
        self.msg = None

    async def start(self):
        self.msg = cl.Message(
            content="🟡 Initializing...",
        )
        await self.msg.send()

    async def update(self, step: str):
        if self.msg:
            self.msg.content = step
            await self.msg.update()

@cl.action_callback("mode_pdf")
async def set_pdf(action: cl.Action):
    cl.user_session.set("source_mode", "pdf")
    await cl.Message("📄 ✅ **PDF Mode Activated** — I’ll only use your documents.").send()

@cl.action_callback("mode_internet")
async def set_internet(action: cl.Action):
    cl.user_session.set("source_mode", "internet")
    await cl.Message("🌐 **Internet mode enabled** — I’ll only use internet.").send()

@cl.action_callback("mode_auto")
async def set_auto(action: cl.Action):
    cl.user_session.set("source_mode", "auto")
    await cl.Message("🤖 **Auto mode enabled** — I’ll use both your documents and internet.").send()

@cl.on_chat_start
async def start():
    
    await cl.Message(content="# 📄 ChatPDF").send()    
    
    settings = await cl.ChatSettings(
        [
            Select(
                id="Model",
                label="AI Models",
                values=["mistral", "mistral2", "llama"],
                initial_index=0,
            ),
            Checkbox(
                id="Quantization",
                label="Enable Quantization",
                initial=True,
            ),
            RadioGroup(
                id="Embedding",
                label="Embedding of PDF",
                values=["Fast", "Balanced", "Better", "Strong"],
                initial_index=1,
            ),
        ]
    ).send()
    
    model_choice = settings["Model"]
    quantization = settings["Quantization"]
    embed_choice = settings["Embedding"]

    cl.user_session.set("quantization_mode", quantization)
    cl.user_session.set("model_choice", model_choice)
    cl.user_session.set("embed_choice", embed_choice)

    pipe = load_llm(model_choice, MODEL_MAP, quantization)
    
    cl.user_session.set("pipe", pipe)

    files = await cl.AskFileMessage(
        content=f"📄 Please upload a pdf file to begin! [max {MAX_FILES} files]",
        accept=["application/pdf"],
        max_size_mb = MAX_SIZE_MB,
        max_files = MAX_FILES,
        timeout = TIME_OUT,
    ).send()
    
    if not files:
        await cl.Message("❌ No files uploaded. Please restart.").send()
        return

    all_chunks = []

    async with cl.Step(name="PDF Processing") as step:

        step.output = "📄 Loading files..."
    
        start_time = time.time()

        for file in files:
            step.output = f"📄 Parsing {file.name}"
            chunks = await asyncio.to_thread(load_pdf, file.path, file.name)
            
            all_chunks.extend(chunks)

        step.output = "🔎 Building index..."
        
        embeddings = get_embeddings(EMBEDDING_MAP, embed_choice)

        retriever = VectorRetriever(all_chunks, embeddings)

        memory = ChatMemory()

        cl.user_session.set("retriever", retriever)
        cl.user_session.set("memory", memory)
        cl.user_session.set("chunks", all_chunks)

        end_time = time.time()
        
        elapsed = end_time - start_time

        step.output = f"✅ PDF {file.name} Ready in {elapsed:.2f}s — start chatting!"
        await step.update()

    async with cl.Step(
        name="⚡ How should I answer?") as step:
        step.output = """
Select a mode:

- 📄 **PDF Mode** → answers only from uploaded documents  
- 🌐 **Internet Mode** → answers from web search  
- 🤖 **Auto Mode** → best of both (recommended)
"""

    await cl.Message(
        content="👇 Select a mode:",
        actions=[
            cl.Action(name="mode_pdf", payload={"mode": "pdf"}, label="📄 PDF"),
            cl.Action(name="mode_internet", payload={"mode": "internet"}, label="🌐 Internet"),
            cl.Action(name="mode_auto", payload={"mode": "auto"}, label="🤖 Auto"),
        ]
    ).send()


@cl.on_message
async def main(message: cl.Message):

    query = message.content
    
    docs = []
    context = ""
    answer = ""
    
    retriever = cl.user_session.get("retriever")
    chunks = cl.user_session.get("chunks")
    memory = cl.user_session.get("memory")
    source_mode = cl.user_session.get("source_mode")

    if not source_mode:
        await cl.Message(
            content="⚠️ Please select a mode above before proceeding..."
        ).send()
        return

    if query.lower() in ["pdf", "internet", "auto"]:
        cl.user_session.set("source_mode", query.lower())
        await cl.Message(f"✅ Switched to {query.upper()} mode").send()
        return
    
    progress=ProgressUI()
    await progress.start()
    
    await progress.update("🔍 Understanding query...")

    await progress.update("📡 Retrieving & reasoning...")
    
    reranker = get_reranker()
    
    answer, docs, meta = await asyncio.to_thread(
        process_query,
        query,
        retriever,
        memory,
        source_mode,
        rerank_docs,
        safe_generate,
        chunks,
        reranker
    )
    if "I don't know" in answer:
        answer = "I don't know"
    
    await progress.update("🧠 Finalizing answer...")
    
    route = meta.get("route")
    confidence = meta.get("confidence")
    evaluation = meta.get("evaluation", {})
    context = meta.get("context", "")
    
    if source_mode == "auto":
        if route == "pdf":
            msg = "📄 Answering from PDF."
        elif route == "internet":
            if confidence == 1.0:
                msg = "🌐 Using Internet (fresh information requested)."
            else:
                msg = "⚠️ Low PDF confidence → switching to Internet"
        elif route == "hybrid":
            msg = "🔀 Mixed mode → using both PDF and Internet."
    elif source_mode == "pdf":
        msg = "📄 Answering from PDF."
    else:
        msg = "🌐 Using Internet"
    
    formatted_sources = "\n".join(
        f"- {i}. {d['source']} • Page {d['page']}"
        for i, d in enumerate(serialize_docs(docs[:2]), 1)
    )
    
    content = f"{msg}\n\n{answer}\n\n{formatted_sources}"

    await progress.update(content)

    memory.add(query, answer)

    gc.collect()
    #torch.cuda.empty_cache()

@cl.action_callback("toggle_details")
async def show_details(action: cl.Action):
    payload = action.payload or {}

    docs = payload.get("docs", [])
    evaluation = payload.get("evaluation", {})

    if not docs and not evaluation:
        await cl.Message("No details available.").send()
        return

    content = "\n\n---\n\n"

    # 📚 Sources (clean, structured)
    if docs:
        content += "### 📚 Sources\n\n"

        for i, d in enumerate(docs, 1):
            content += (
                f"**{i}. {d['source']} • (p.{d['page']})**\n"
                f"{d['text']}...\n\n"
            )

    # 🧠 Evaluation
    if evaluation:
        content += "### 🧠 Evaluation\n\n"
        content += (
            f"- Confidence: {evaluation.get('confidence','N/A')}\n"
            f"- Groundedness: {evaluation.get('groundedness','N/A')}\n"
            f"- Hallucination Risk: {evaluation.get('hallucination_risk','N/A')}\n"
        )

    await cl.Message(content=content).send()