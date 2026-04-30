import chainlit as cl

def safe_generate(prompt, **gen_kwargs):
    try:
        pipe = cl.user_session.get("pipe")
        result = pipe(prompt, **gen_kwargs)

        if not result or "generated_text" not in result[0]:
            return "❌ Model returned empty response."

        text = result[0]["generated_text"].strip()

        return text

    except Exception as e:
        return f"❌ Error: {str(e)}"