
def build_rag_prompt(query, context):
    return f"""
<s>[INST]
You are a question-answering system.

Rules:
- Answer ONLY from the context but do not say "from context"
- If answer is not in the context then do NOT include any citation, say: "I don't know"
- Cite sources ONLY [number], e.g. [1].
- Answer only in natural language
- If multiple pages contain the answer, cite the most relevant page only
- Prefer the MOST relevant and concise information (usually from [1])
- For "what is" questions, return ONLY the definition (1–2 sentences)
- Do NOT include unrelated technical details
- Be concise and factual

<<<CONTEXT>>>
{context}
<<<END_CONTEXT>>>

Question:
{query}
[/INST]
"""

def fusion_prompt(answer, web_info):
    return f"""
<s>[INST]
Combine the answers.

Rules:
- Prefer PDF information
- Use web only to supplement missing details
- Do not contradict PDF
- Do NOT include unrelated technical details
- Do NOT invent new questions and answers
- Keep final answer concise

<<<PDF_ANSWER>>>
{answer}
<<<END_PDF_ANSWER>>>

<<<WEB_ANSWER>>>
{web_info}
<<<END_WEB_ANSWER>>>

Return:
Final answer only.
[/INST]
"""