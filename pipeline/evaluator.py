from llm.generator import safe_generate
from config import GEN_CONFIGS

def evaluate_answer(query, answer, context, safe_generate):
    prompt = f"""
<s>[INST]
Evaluate the answer.

Rules:
- Return ONLY valid JSON
- No extra text
- Confidence must be an integer (0-100)

Question: {query}

<<<ANSWER>>>
{answer}
<<<END_ANSWER>>>

<<<CONTEXT>>>
{context}
<<<END_CONTEXT>>>

Output format:
{{
  "confidence": int,
  "groundedness": "high" | "medium" | "low",
  "hallucination_risk": "low" | "medium" | "high"
}}
[/INST]
"""

    return safe_generate(prompt, **GEN_CONFIGS["evaluation"])


def evaluate_hybrid(query, pdf, web, final, safe_generate):
    prompt = f"""
<s>[INST]
Evaluate the answer using BOTH sources.

Rules:
- Return ONLY valid JSON
- No explanations

Question: {query}

<<<PDF>>>
{pdf}
<<<END_PDF>>>

<<<WEB>>>
{web}
<<<END_WEB>>>

<<<FINAL_ANSWER>>>
{final}
<<<END_FINAL_ANSWER>>>

Output format:
{{
  "confidence": int,
  "consistency": "high" | "medium" | "low",
  "hallucination_risk": "low" | "medium" | "high"
}}
[/INST]
"""

    return safe_generate(prompt, **GEN_CONFIGS["evaluation"])    


def parse_eval(text):
    import json, re
    try:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        return json.loads(match.group(0))
    except:
        return {}


def run_evaluation(query, answer, context, web=None, safe_generate=None):

    if web is None:
        raw = evaluate_answer(query, answer, context, safe_generate)
    else:
        raw = evaluate_hybrid(query, context, web, answer, safe_generate)

    return parse_eval(raw)