from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch

# Initialize model and tokenizer
model_id = "../Llama-2-7b-chat-hf"
HF_ACCESS_TOKEN = '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'

model = AutoModelForCausalLM.from_pretrained(model_id, use_auth_token=HF_ACCESS_TOKEN, torch_dtype=torch.float16, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_id, use_auth_token=HF_ACCESS_TOKEN)
chat_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.float32,
    device_map="auto",
    token=HF_ACCESS_TOKEN
)

# Function to generate a response
def generate_response(prompt):
    sequences = chat_pipeline(prompt, do_sample=True, top_k=10, num_return_sequences=1, eos_token_id=tokenizer.eos_token_id, max_length=250)
    response = sequences[0]['generated_text']
    marker = "[/INST]\n\n"
    index = response.find(marker)
    if index != -1:
        return response[index + len(marker):].strip()
    return response.strip()

# Initial prompt
initial_prompt = "What is your name?"

# Simulate conversation
conversation_history = [initial_prompt]
for _ in range(5):  # Number of exchanges
    bot1_prompt = conversation_history[-1]
    bot1_response = generate_response(bot1_prompt)
    conversation_history.append(f"Bot1: {bot1_response}")

    bot2_prompt = bot1_response
    bot2_response = generate_response(bot2_prompt)
    conversation_history.append(f"Bot2: {bot2_response}")

# Print conversation
for line in conversation_history:
    print(line)
