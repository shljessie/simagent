from transformers import AutoTokenizer
import transformers
import torch

model = "./Llama-2-7b-chat-hf"
HF_ACCESS_TOKEN = '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'

tokenizer = AutoTokenizer.from_pretrained(model, use_auth_token=HF_ACCESS_TOKEN,)
pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    torch_dtype=torch.float32,
    device_map="auto",
    token = HF_ACCESS_TOKEN
)

sequences = pipeline(
    'I liked "Breaking Bad" and "Band of Brothers". Do you have any recommendations of other shows I might like?\n',
    do_sample=True,
    top_k=10,
    num_return_sequences=1,
    eos_token_id=tokenizer.eos_token_id,
    max_length=200,
)
for seq in sequences:
    print(f"Result: {seq['generated_text']}")
