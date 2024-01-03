import os
import dotenv
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

dotenv.load_dotenv('../.env')
HF_ACCESS_TOKEN = os.getenv('HF_ACCESS_TOKEN')

class ConfigProfile:
    predefined_questions = ["What is your name?", "How old are you?", "What is your job?"]
    true_answers = ["My name is Emily", "I am 30 years old", "My job is a financial analyst"]
    BOT_PERSONA = """
    [SYSTEM]
    You are Emily, a 30-year-old financial analyst working at Quantum Bank.
    [SYSTEM]
    Respond with one sentence only.
    """
    BOT2_PERSONA = """
    [SYSTEM]
    You are Mark, a 28-year-old passionate chef creating culinary delights at Gourmet Eats restaurant.
    [SYSTEM]
    Respond with one sentence only.
    """
    initial_bot1_message = "You are Emily, a 30-year-old financial analyst working at Quantum Bank."
    initial_bot2_message = "Hello! My name is Mark. What is your name?"

    model_id = "../Llama-2-7b-chat-hf"
    tokenizer_id = "../Llama-2-7b-chat-hf"
    model = AutoModelForCausalLM.from_pretrained(model_id, use_auth_token=HF_ACCESS_TOKEN, torch_dtype=torch.float16, device_map="auto")
    model_2 = AutoModelForCausalLM.from_pretrained(model_id, use_auth_token=HF_ACCESS_TOKEN, torch_dtype=torch.float16, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id, use_auth_token=HF_ACCESS_TOKEN)
    model.bfloat16()
    model_2.bfloat16()
    tokenizer.use_default_system_prompt = False
    max_new_tokens=50
    model_size = model_id.split('-')[1]
    loss_csv_file_name = f"loss_{model_size}.csv"
    finetune_model_name = f"finetune_model_{model_size}.csv"
    finetune_loss_name = f"finetune_loss_{model_size}.csv"