import os
import torch
import math
import dotenv
import torch
from transformers import (
    AutoConfig, 
    AutoModelForCausalLM, 
    AutoTokenizer, 
    BitsAndBytesConfig, 
    pipeline
)
import csv
import subprocess
from diagnostic import calculate_loss  #import the calc loss function
import json
from base import generate

# Model Configurations
dotenv.load_dotenv('../.env')
HF_ACCESS_TOKEN = os.getenv('HF_ACCESS_TOKEN')
model_id = '../Llama-2-7b-chat-hf'

# Configuration settings
bnb_config = BitsAndBytesConfig(
    bnb_4bit_compute_dtype='float16',
    bnb_4bit_quant_type='nf4',
    load_in_4bit=True,
)

# Load model and tokenizer
def initialize_model(model_id=None, HF_ACCESS_TOKEN=None):
  model_config = AutoConfig.from_pretrained(model_id, use_auth_token=HF_ACCESS_TOKEN)
  model = AutoModelForCausalLM.from_pretrained(model_id, config=model_config, quantization_config=bnb_config, use_auth_token=HF_ACCESS_TOKEN)
  tokenizer = AutoTokenizer.from_pretrained(model_id, use_auth_token=HF_ACCESS_TOKEN)
  model.eval()

  return model, tokenizer

# intitialize two models
model1, tokenizer1 = initialize_model(model_id, HF_ACCESS_TOKEN)
model2, tokenizer2 = initialize_model(model_id, HF_ACCESS_TOKEN)

def multi_agent_chat(turns):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    history1 = torch.tensor([], dtype=torch.long)
    history2 = torch.tensor([], dtype=torch.long)

    for _ in range(turns):
        # Bot 1 generates a response
        encoded_prompt1 = tokenizer1.encode("", device=device) # Start with an empty prompt or an introduction
        response1 = generate(model1, encoded_prompt1, model1.max_seq_length, history1)
        history1 = torch.cat([history1, response1])

        # Feed Bot 1's response to Bot 2 as a prompt
        history2 = torch.cat([history2, response1])
        response2 = generate(model2, response1, model2.max_seq_length, history2)
        history2 = torch.cat([history2, response2])

        # Now feed Bot 2's response back to Bot 1
        history1 = torch.cat([history1, response2])





multi_agent_chat(5)
