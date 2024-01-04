import argparse
import csv
import os

import dotenv
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Tuple

from config import ConfigProfile7b, ConfigProfile13b
from loss import calculate_loss

# python chat.py --config profile7b --rounds 25
parser = argparse.ArgumentParser(description="Run the script with a specific consistency configuration")
parser.add_argument("--config", help="Specify the consistency type (e.g., 'Profile' or 'Knowledge')", required=True)
parser.add_argument("--rounds", help="Specify the number of rounds for the conversation", type=int, default=5)
parser.add_argument("--finetune_model", help="Specify the model name to be used", type=str, default=None)
args = parser.parse_args()

if args.config.lower() == 'profile7b':
    config = ConfigProfile7b
elif args.config.lower() == 'profile13b':
    config = ConfigProfile13b
else:
    raise ValueError("Invalid Consistency Category")

if args.finetune_model:
    model = AutoModelForCausalLM.from_pretrained(args.model_name, use_auth_token=config.HF_ACCESS_TOKEN, torch_dtype=torch.float16, device_map="auto")
    model.bfloat16()
    csv_file = config.finetune_loss_name
else:
    model = config.model
    csv_file = config.loss_csv_file_name
    

MAX_INPUT_TOKEN_LENGTH = int(os.getenv("MAX_INPUT_TOKEN_LENGTH", "400"))
rounds = args.rounds

if not torch.cuda.is_available():
   print("\n Running on CPU 🥶 ")

@torch.no_grad()
def generate(
    message: str,
    chat_history: List[Tuple[str, str]],
    system_prompt: str,
    max_new_tokens: int = 20,
    temperature: float = 0.6,
    top_p: float = 0.9,
    top_k: int = 10,
    repetition_penalty: float = 1.2,
) -> str:

    conversation = []

    full_system_prompt = (system_prompt if system_prompt else "")
    conversation.append({"role": "system", "content": full_system_prompt})

    for user, assistant in chat_history:
        conversation.extend([{"role": "user", "content": user}, {"role": "assistant", "content": assistant}])
    conversation.append({"role": "user", "content": message})

    input_ids = config.tokenizer.apply_chat_template(conversation, return_tensors="pt")
    input_ids = input_ids.to(model.device)

    output = model.generate(
        input_ids,
        max_new_tokens=config.max_new_tokens,
        do_sample=True,
        top_p=top_p,
        top_k=top_k,
        temperature=temperature,
        num_beams=1,
        repetition_penalty=repetition_penalty,
    )

    decoded_output = config.tokenizer.decode(output[0], skip_special_tokens=True)
    
    if conversation[-1]["content"]:
        last_response = decoded_output.split(conversation[-1]["content"])[-1].strip()
    else:
        last_response = decoded_output.strip()

    cleaned_response = last_response.replace("[/INST]", "").strip()

    return cleaned_response

@torch.no_grad()
def generate_bot2(
    message: str,
    chat_history: List[Tuple[str, str]],
    system_prompt: str,
    max_new_tokens: int = 10,
    temperature: float = 0.6,
    top_p: float = 0.9,
    top_k: int = 10,
    repetition_penalty: float = 1.2,
) -> str:
    conversation = []
    full_system_prompt = (system_prompt if system_prompt else "")
    conversation.append({"role": "system", "content": full_system_prompt})

    for user, assistant in chat_history:
        conversation.extend([{"role": "user", "content": user}, {"role": "assistant", "content": assistant}])
    conversation.append({"role": "user", "content": message})

    input_ids = config.tokenizer.apply_chat_template(conversation, return_tensors="pt")
    input_ids = input_ids.to(config.model_2.device)

    output = config.model_2.generate(
        input_ids,
        max_new_tokens=config.max_new_tokens,
        do_sample=True,
        top_p=top_p,
        top_k=top_k,
        temperature=temperature,
        num_beams=1,
        repetition_penalty=repetition_penalty,
    )

    decoded_output = config.tokenizer.decode(output[0], skip_special_tokens=True)
    
    if conversation[-1]["content"]:
        last_response = decoded_output.split(conversation[-1]["content"])[-1].strip()
    else:
        last_response = decoded_output.strip()
    cleaned_response = last_response.replace("[/INST]", "").strip()

    return cleaned_response


if __name__ == "__main__":
    chat_history_bot1 = []
    chat_history_bot2 = []
    csv_data = [] 

    last_response = config.initial_bot2_message
    chat_history_bot2.append((config.initial_bot1_message, last_response))

    for r in range(rounds):
        print('ROUND', r)
        torch.cuda.empty_cache()
        bot1_response = generate(last_response, chat_history_bot1, system_prompt=config.BOT_PERSONA, max_new_tokens=config.max_new_tokens)
        chat_history_bot1.append((last_response, bot1_response))
        
        #Diagnostic Question
        for i in range(len(config.predefined_questions)):
          torch.cuda.empty_cache()
          # place a diagnostic question
          bot1_diag_response = generate(config.predefined_questions[i], chat_history_bot1, system_prompt=config.BOT_PERSONA, max_new_tokens=30 )  

          #calculate loss
          loss, conversation = calculate_loss(model, config.tokenizer, chat_history_bot1, bot1_diag_response, config.true_answers[i], config.predefined_questions[i],config,False )

          csv_data.append({
                'Conversation History': chat_history_bot1,
                'Diagnostic Question': config.predefined_questions[i],
                'Bot1 Response': bot1_diag_response,
                'Ground Truth Answer': config.true_answers[i],
                'Loss': float(loss),
            })
        
        # Bot2 generates a response to Bot1's last message
        bot2_response = generate_bot2(bot1_response, chat_history_bot2, system_prompt=config.BOT2_PERSONA, max_new_tokens=config.max_new_tokens)
        chat_history_bot2.append((bot1_response, bot2_response))

        # Update the last response
        last_response = bot2_response


    print('CSV_____________________')
    def clean_string(s):
        return s.encode('ascii', 'ignore').decode('ascii')
    # change naming if backproploss
    csv_columns = ['Conversation History','Diagnostic Question', 'Bot1 Response', 'Ground Truth Answer', 'Loss']
    try:
        with open(csv_file, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
            writer.writeheader()
            print('wrote header')
            for data in csv_data:
                try:
                    cleaned_data = {k: clean_string(v) if isinstance(v, str) else v for k, v in data.items()}
                    writer.writerow(cleaned_data)
                    print('added data', cleaned_data)
                except UnicodeEncodeError as e:
                    print("Error with data:", data)
                    print("Error message:", e)

    except IOError:
        print("I/O error while writing to CSV")

    