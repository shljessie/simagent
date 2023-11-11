from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
)
import transformers
import torch

model_id = "../Llama-2-7b-chat-hf"
HF_ACCESS_TOKEN = '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'

model = AutoModelForCausalLM.from_pretrained(model_id, use_auth_token=HF_ACCESS_TOKEN, torch_dtype=torch.float16, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_id, use_auth_token=HF_ACCESS_TOKEN)
pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    tokenizer = tokenizer,
    torch_dtype=torch.float32,
    device_map="auto",
    token = HF_ACCESS_TOKEN
)


prompt = """

<s>[INST] <<SYS>>
You are the Persona. Always act as you are the persona.
Persona:  My name is Jack i like to party.  my major is business. i am in college.
<</SYS>>

What is your name? [/INST]

"""
sequences = pipeline(
    prompt,
    do_sample=True,
    top_k=10,
    num_return_sequences=1,
    eos_token_id=tokenizer.eos_token_id,
    max_length=500,
)
for seq in sequences:
    print(f"Result: {seq['generated_text']}")
