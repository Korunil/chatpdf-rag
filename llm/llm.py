import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from transformers import BitsAndBytesConfig
from config import MAX_NEW_TOKENS, CACHE_DIR
from cache_store import _MODEL_CACHE

def load_llm(model_choice, model_map, quantization):
    key = f"{model_choice}_{quantization}"

    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]
    
    model_name = model_map.get(model_choice, model_map["mistral"])

    if quantization:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True
        )

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            quantization_config=bnb_config,
            low_cpu_mem_usage=True,
            cache_dir=CACHE_DIR
        )

    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            dtype=torch.float16,
            cache_dir=CACHE_DIR
        )

    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=CACHE_DIR)

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=MAX_NEW_TOKENS, 
        return_full_text=False
    )

    _MODEL_CACHE[key] = pipe
    return pipe