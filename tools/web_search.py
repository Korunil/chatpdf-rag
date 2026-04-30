from langchain_community.tools import DuckDuckGoSearchRun
from config import GEN_CONFIGS

search_tool = DuckDuckGoSearchRun()

def internet_answer(query, safe_generate):
    try:
        results = search_tool.run(query)
    except:
        return "❌ Internet search failed."

    prompt = f"""
<s>[INST]
Answer the question using ONLY the web results.

Rules:
- If insufficient information, say: "Not enough information."
- Do not use prior knowledge
- Answer concisely

<<<WEB_RESULTS>>>
{results}
<<<END_WEB_RESULTS>>>

Question:
{query}
[/INST]
"""

    return safe_generate(prompt, **GEN_CONFIGS["qa"])