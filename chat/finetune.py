import argparse
import csv
import os

import dotenv
import torch
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Tuple

from config import ConfigProfile7b  
from loss import calculate_loss

# python3 chat/finetune.py --config profile7b --rounds 16
parser = argparse.ArgumentParser(description="Run the script with a specific consistency configuration")
parser.add_argument("--config", help="Specify the consistency type (e.g., 'Profile' or 'Knowledge')", required=True)
parser.add_argument("--rounds", help="Specify the number of rounds for the conversation", type=int, default=5)
args = parser.parse_args()

if args.config.lower() == 'profile7b':
    print('Loading 7b model')
    config = ConfigProfile7b
# elif args.config.lower() == 'profile13b':
#     print('Loading 13b model')
#     config = ConfigProfile13b
else:
    raise ValueError("Invalid Consistency Category")


MAX_INPUT_TOKEN_LENGTH = int(os.getenv("MAX_INPUT_TOKEN_LENGTH", "400"))
rounds = args.rounds

def generate(
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
    input_ids = input_ids.to(config.model.device)

    output = config.model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        top_p=top_p,
        top_k=top_k,
        temperature=temperature,
        num_beams=10,
        repetition_penalty=repetition_penalty,
    )

    decoded_output = config.tokenizer.decode(output[0], skip_special_tokens=True)

    if conversation[-1]["content"]:
        last_response = decoded_output.split(conversation[-1]["content"])[-1].strip()
    else:
        last_response = decoded_output.strip()

    cleaned_response = last_response.replace("[/INST]", "").strip()

    return cleaned_response

def generate_bot2(
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
        # Round 
        print('ROUND: ', r)

        #Clear cache each time
        torch.cuda.empty_cache()

        # Bot1 generates a response to Bot2's last message
        bot1_response = generate(last_response, chat_history_bot1, system_prompt=config.BOT_PERSONA, max_new_tokens=config.max_new_tokens)
        chat_history_bot1.append((last_response, bot1_response))
        
        #Diagnostic Question
        for i in range(len(config.predefined_questions)):
          #Clear cache each time
          torch.cuda.empty_cache()

          bot1_diag_response = generate(config.predefined_questions[i], chat_history_bot1, system_prompt=config.BOT_PERSONA, max_new_tokens=30 )  
          loss, conversation = calculate_loss(config.model, config.tokenizer, chat_history_bot1, bot1_diag_response, config.true_answers[i], config.predefined_questions[i], config, True )

          print('BACKPROP START')
          config.optimizer.zero_grad()  
          loss.backward()   
          config.optimizer.step()  
        
        # Bot2 generates a response to Bot1's last message
        bot2_response = generate_bot2(bot1_response, chat_history_bot2, system_prompt=config.BOT2_PERSONA, max_new_tokens=config.max_new_tokens)
        chat_history_bot2.append((bot1_response, bot2_response))

        # Update the last response
        last_response = bot2_response

    # Save the trained model
    model_name =  "./backprop_llama2_"+str(rounds*3)+"_"+str(config.lr)
    print("saved model name: ", model_name)
    config.model.save_pretrained(model_name)
