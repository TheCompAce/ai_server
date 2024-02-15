import torch
from transformers import pipeline

def ask_tinyllama(system, user):
    text_pipe = pipeline("text-generation", model="TinyLlama/TinyLlama-1.1B-Chat-v1.0", torch_dtype=torch.bfloat16, device_map="auto")
    
    messages = [
        {
            "role": "system",
            "content": system,
        },
        {"role": "user", "content": user},
    ]

    prompt = text_pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    outputs = text_pipe(prompt, max_new_tokens=1024, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)
    response = outputs[0]["generated_text"]
    # response = your_llm_function(question)
    return response