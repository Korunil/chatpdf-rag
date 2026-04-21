import time
import asyncio
import chainlit as cl
import torch
import gc
import os
import re

from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from sentence_transformers import CrossEncoder

from langchain_classic.memory import ConversationBufferWindowMemory

from transformers import BitsAndBytesConfig

from langchain_community.tools import DuckDuckGoSearchRun

search_tool = DuckDuckGoSearchRun()

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True
)

# =========================
# 🔧 MODELS
# =========================

embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-base-en-v1.5",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"batch_size": 32}
)

model_name = "mistralai/Mistral-7B-Instruct-v0.2"

tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype=torch.float16,
    device_map="auto",
    quantization_config=bnb_config,
    low_cpu_mem_usage=True
)

# =========================
# PIPELINE
# =========================
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=250,
    do_sample=False,
    return_full_text=False,
    pad_token_id=tokenizer.eos_token_id
)

# =========================
# 🔥 RERANKER MODEL
# =========================
reranker = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2",
    device="cpu"
    )

# =========================
# 📄 PDF
# =========================

def load_pdf(file_path):
    loader = PDFPlumberLoader(file_path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=200
    )

    return splitter.split_documents(docs)

# =========================
# 🧠 VECTOR STORE
# =========================

def build_vectorstore(chunks, file_name):
    index_path = f"faiss_index_{file_name}"
    
    if os.path.exists(index_path):
        return FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)

    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(index_path)

    return vectorstore

# =========================
# UTIL
# =========================

def safe_generate(prompt):
    try:
        result = pipe(prompt)

        if not result or "generated_text" not in result[0]:
            return "❌ Model returned empty response."

        text = result[0]["generated_text"].strip()

        return text if text else "❌ Empty response."

    except Exception as e:
        return f"❌ Error: {str(e)}"

def rerank_docs(query, docs, top_k=5):

    # ✅ remove None docs early
    docs = [d for d in docs if d and hasattr(d, "page_content")]

    if not docs:
        return []

    pairs = [(query, d.page_content) for d in docs]

    try:
        scores = reranker.predict(pairs, batch_size=32, convert_to_numpy=True)
    except Exception:
        return docs[:top_k]  # fallback

    scored_docs = list(zip(docs, scores))

    scored_docs.sort(key=lambda x: x[1], reverse=True)

    return [d for d, _ in scored_docs[:top_k] if d]

# =========================
# 🧠 ROUTERS
# =========================

def classify_intent(query):

    prompt = f"""
Classify the query strictly:

- summary → asking overall document meaning
- extract → asking for ALL items / list
- qa → specific question about content

Examples:
"What is this document about?" → summary
"List all risks" → extract
"What is revenue?" → qa

Query:
{query}

Answer ONLY one word: summary / extract / qa
"""

    result = safe_generate(prompt).lower()

    if "summary" in result:
        return "summary"
    elif "extract" in result:
        return "extract"
    else:
        return "qa"

def decide_source(query, source_mode, vectorstore):

    if source_mode == "pdf":
        return "pdf"

    if source_mode == "internet":
        return "internet"

    # AUTO mode
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    docs = retriever.invoke(query)

    if not docs:
        return "internet"

    # simple heuristic: weak matches → internet
    # if len(docs) < 2:
    #    return "internet"

    return "pdf"

# =========================
# 🧠 QA WITH RERANKING
# =========================

def rag_answer(vectorstore, query, memory):

    # 🔥 rewrite query
    rewritten_query = rewrite_query(query, memory)

    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 6, "fetch_k": 12}  # 🔥 higher recall
    )

    docs = retriever.invoke(rewritten_query)

    if not docs:
        return None, []

    # 🔥 RERANK HERE
    top_docs = rerank_docs(rewritten_query, docs, top_k=4)

    context = "\n\n".join([
        f"(Source: {d.metadata.get('source')} | Page {d.metadata.get('page')}) {d.page_content}"
        for d in top_docs
    ])

    prompt = f"""
Answer ONLY using the context.

Context:
{context}

Question:
{query}

Cite sources like (Page X).
"""

    return prompt, top_docs

# =========================
# 🧠 SUMMARY
# =========================

def generate_summary(chunks):

    selected_chunks = chunks[:20]

    text = "\n".join([c.page_content[:500] for c in selected_chunks])

    prompt = f"""
Summarize the document:

{text}

Provide key points.
"""

    answer = safe_generate(prompt)

    return answer, selected_chunks

# =========================
# 🧠 EXTRACTION (IMPROVED)
# =========================

def extract_all(vectorstore, query):

    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 15}
    )

    docs = retriever.invoke(query)
    
    docs = [d for d in docs if d and hasattr(d, "page_content")]

    # 🔥 RERANK FOR EXTRACTION
    top_docs = rerank_docs(query, docs, top_k=8)
    
    if not top_docs:
        return "❌ No relevant content found for extraction.", []
    
    text = "\n".join([
        d.page_content for d in top_docs
        if d and hasattr(d, "page_content")
        ])

    prompt = f"""
Extract ALL relevant items.

Task: {query}

Text:
{text}

Return as bullet points.
"""

    answer = safe_generate(prompt)

    return answer, top_docs

# =========================
# 🧠 INTERNET
# =========================

def internet_answer(query):

    results = search_tool.run(query)

    if not results or len(results.strip()) < 50:
        return "❌ Could not retrieve reliable internet results."

    prompt = f"""
You are answering using web search results.

Use ONLY the information below.
If insufficient, say "Not enough information."

Web Results:
{results}

Question:
{query}

Answer concisely:
"""

    return safe_generate(prompt)

# =========================
# REWRITE QUERY
# =========================
def rewrite_query(query, memory):

    history = memory.load_memory_variables({}).get("chat_history", "")

    if not history:
        return query  # first question

    prompt = f"""
Rewrite the user query into a standalone question.

Chat History:
{history}

Query:
{query}

Rewritten:
"""

    rewritten = safe_generate(prompt)

    return rewritten if rewritten else query

# =========================
# RELEVANT SNIPPET
# =========================
def extract_relevant_snippet(query, doc):

    sentences = re.split(r'(?<=[.!?]) +', doc.page_content)

    scored = [(s, reranker.predict([(query, s)])[0]) for s in sentences]

    best = sorted(scored, key=lambda x: x[1], reverse=True)[0][0]

    return best

# =========================
# 🧠 STREAM GENERATION
# =========================

async def stream_answer(prompt, msg):

    loop = asyncio.get_event_loop()

    result = await loop.run_in_executor(
        None,
        lambda: pipe(prompt)
    )

    if not result or "generated_text" not in result[0]:
        await msg.stream_token("❌ Error generating response.")
        return ""

    text = result[0]["generated_text"]

    for token in text.split():
        await msg.stream_token(token + " ")

    return text

# =========================
# ACTION CALLBACKS
# =========================
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

# =========================
# 🚀 START
# =========================

@cl.on_chat_start
async def start():
    
    await cl.Message(
        content="""
# 📄 ChatPDF

Upload your PDF and ask questions.

- Supports technical documents  
- Context-aware answers  
- Source references included  
"""
    ).send()

    start_time = time.time()
    
    files = await cl.AskFileMessage(
        content="📄 Upload PDF(s) [Max 5 files]",
        accept=["application/pdf"],
        max_size_mb=20,
        timeout=600,
        max_files=5   # ✅ allow multiple
    ).send()

    all_chunks = []

    for file in files:
        await cl.Message(content=f"📄 Selected: {file.name}").send()
        await cl.Message(content=f"⏳ Processing PDF...").send()
        chunks = await asyncio.to_thread(load_pdf, file.path)

        # tag source
        for c in chunks:
            c.metadata["source"] = file.name

        all_chunks.extend(chunks)

    await cl.Message("🔎 Building Index").send()
    vectorstore = await asyncio.to_thread(lambda: FAISS.from_documents(all_chunks, embeddings))
    
    # 🔥 LangChain memory
    memory = ConversationBufferWindowMemory(
        k=3,
        return_messages=False  # IMPORTANT (we want plain text)
    )

    cl.user_session.set("vectorstore", vectorstore)
    cl.user_session.set("memory", memory)
    cl.user_session.set("chunks", all_chunks)

    elapsed = time.time() - start_time

    await cl.Message(
        content=f"✅ Ready in {elapsed:.2f}s — start chatting!"
    ).send()

    # SHOW BUTTONS ONCE
    await cl.Message(
        content="""
## ⚡ How should I answer?

Please select a mode before asking your question:

- 📄 **PDF Mode** → answers only from uploaded documents  
- 🌐 **Internet Mode** → answers from web search  
- 🤖 **Auto Mode** → best of both (recommended)

👇 Select one below:
""",
        actions=[
            cl.Action(name="mode_pdf", payload={"mode": "pdf"}, label="📄 PDF"),
            cl.Action(name="mode_internet", payload={"mode": "internet"}, label="🌐 Internet"),
            cl.Action(name="mode_auto", payload={"mode": "auto"}, label="🤖 Auto"),
        ]
    ).send()

# =========================
# 🚀 MESSAGE HANDLER
# =========================

@cl.on_message
async def main(message: cl.Message):

    docs = []
    query = message.content

    vectorstore = cl.user_session.get("vectorstore")
    chunks = cl.user_session.get("chunks")
    memory = cl.user_session.get("memory")
    source_mode = cl.user_session.get("source_mode")

    msg = cl.Message(content="🔍 Processing...")
    await msg.send()

    if not source_mode:
        await cl.Message(
            content="⚠️ Please select a mode above before asking a question."
        ).send()
        return

    if query.lower() in ["pdf", "internet", "auto"]:
        cl.user_session.set("source_mode", query.lower())
        await cl.Message(f"✅ Switched to {query.upper()} mode").send()
        return
    
    intent = classify_intent(query)
    source = decide_source(query, source_mode, vectorstore)
    
    #AUTO
    if source_mode == "auto":

        prompt, docs = rag_answer(vectorstore, query, memory)

        # Case 1: Good PDF context
        if prompt and docs:

            pdf_answer = safe_generate(prompt)

            web_info = search_tool.run(query)

            if web_info and len(web_info.strip()) > 50:

                final_prompt = f"""
You are refining an answer using two sources:

1. PDF Answer (more reliable)
2. Web Info (for enrichment)

Guidelines:
- Prefer PDF information when conflicts arise
- Use web info only to enhance completeness
- Do NOT contradict PDF unless clearly outdated
- Keep answer concise

PDF Answer:
{pdf_answer}

Web Info:
{web_info}

Final Answer:
"""
                answer = safe_generate(final_prompt)
                msg.content = f"🤖 Auto Mode (PDF + Web)\n\n{answer}"

            else:
                # fallback → only PDF
                answer = pdf_answer
                msg.content = f"📄 PDF Only\n\n{answer}"

        # Case 2: Weak / no PDF
        else:
            answer = internet_answer(query)

        msg.content = answer
        await msg.update()

        memory.save_context({"input": query}, {"output": answer})
        return
    
    # INTERNET
    if source == "internet":
        answer = internet_answer(query)
        msg.content = answer
        await msg.update()

        memory.save_context({"input": query}, {"output": answer})
        return

    # SUMMARY
    if intent == "summary":
        answer, docs = generate_summary(chunks)
        msg.content = answer
        await msg.update()

        memory.save_context({"input": query}, {"output": answer})
        return

    # EXTRACTION
    if intent == "extract":
        answer, docs = extract_all(vectorstore, query)
        msg.content = answer
        await msg.update()

        memory.save_context({"input": query}, {"output": answer})
        return

    # QA
    prompt, docs = rag_answer(vectorstore, query, memory)
    
    if source_mode == "pdf" and not prompt:
        msg.content = "❌ No relevant content found in PDFs."
        await msg.update()
        return
    
    elif not prompt:
        msg.content = "❌ No relevant content found in PDFs. Checking Internet..."
        answer = internet_answer(query)
        msg.content = answer
        await msg.update()
    else:
        msg.content = ""
        await msg.update()
        answer = await stream_answer(prompt, msg)
        msg.content = answer

    # await cl.Message(content=f"🧠 Intent: {intent} | Source: {source}"    ).send()
    
    if intent == "summary":
        snippet_fn = lambda q, d: d.page_content[:50]

    elif intent == "extract":
        snippet_fn = lambda q, d: d.page_content[:50]

    else:  # QA
        snippet_fn = extract_relevant_snippet
    
    msg.elements = [
        cl.Text(
            content="\n\n".join([
                f"(Source: {doc.metadata.get('source')} | Page {doc.metadata.get('page','?')}) "
                f"{snippet_fn(query, doc)}"
                for doc in docs[:3]
            ]),
            name="📚 Sources",
            display="side"
        )
    ]
    
    await msg.update()

    # 🔥 save to memory (FIXED)
    memory.save_context(
        {"input": query},
        {"output": answer}
    )

    gc.collect()
    torch.cuda.empty_cache()