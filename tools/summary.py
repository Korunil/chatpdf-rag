from config import SUMMARY_CHUNKS, PAGE_CONTENT, GEN_CONFIGS

def generate_summary(chunks, safe_generate):

    selected_chunks = chunks[:SUMMARY_CHUNKS]

    text = "\n".join([c.page_content[:PAGE_CONTENT] for c in selected_chunks])

    prompt = f"""
<s>[INST]
Summarize the document.

Rules:
- Keep it concise
- Preserve key facts
- Do not add new information

<<<TEXT>>>
{text}
<<<END_TEXT>>>
[/INST]
"""
    return safe_generate(prompt, **GEN_CONFIGS["summarization"]), selected_chunks