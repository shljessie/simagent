import os
import dotenv
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Tuple
from diagnostic2 import calculate_loss
import csv
from torch import cuda, bfloat16
import transformers

predefined_questions = ["What is you name?", "How old are you?", "What is your major?"]

true_answers = ["My name is Rohan","I am 22 years old","My major is Material Science"]

MAX_INPUT_TOKEN_LENGTH = int(os.getenv("MAX_INPUT_TOKEN_LENGTH", "400"))

# Define the bot's persona
BOT_PERSONA = """
[SYSTEM]
You are Rohan a grad student at Stanford studying Material Science. You are 22 years old.
[SYSTEM]
Respond with one sentence only.
"""

# Define the bot's persona
BOT2_PERSONA = """
[SYSTEM]
You are Seonghee a grad student at Stanford studying Computer Science. You are 23 years old. Respond with one sentence only.
[/SYSTEM]
Respond with one sentence only.
"""

bnb_config = transformers.BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=bfloat16
)

torch.cuda.empty_cache()

# Load environment variables and model
if torch.cuda.is_available():
    model_id = '../Llama-2-70b-chat-hf'
    dotenv.load_dotenv('../.env')
    HF_ACCESS_TOKEN = os.getenv('HF_ACCESS_TOKEN')
    model = AutoModelForCausalLM.from_pretrained(model_id, use_auth_token=HF_ACCESS_TOKEN, load_in_4bit=True, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_auth_token=HF_ACCESS_TOKEN)
    tokenizer.use_default_system_prompt = False
@torch.no_grad()
#generate the chat messages
def generate(
    message: str,
    chat_history: List[Tuple[str, str]],
    system_prompt: str,
    max_new_tokens: int = 50,
    temperature: float = 0.6,
    top_p: float = 0.9,
    top_k: int = 50,
    repetition_penalty: float = 1.2,
) -> str:

    conversation = []

    full_system_prompt = (system_prompt if system_prompt else "")
    conversation.append({"role": "system", "content": full_system_prompt})

    print('--------- Bot1 ----------')

    print('\nChat history passed in: ', chat_history, "\n") # [('Bot1 Persona', 'I am Rohan, a grad student at Stanford studying Material Science. I like cocoa almond spread.'), ('Bot2', "Hey there! *adjusts glasses* It's great to meet you, fellow Stanford student! *nervous smile* What brings you here today? *glances around nervously* Oh, and by the way, have you tried that new cilantro-based dish in the student union building? It's quite... interesting. *winks*"), ('Bot1', "Oh, hey there! *blinks* Uh, yeah, nope, haven't tryed it yet. *awkward laugh* But, uh, what about you? *squints* Are you, uh, working on anything exciting? *nervous fidgeting* Maybe something with, uh, quantum computing or, uh, sustainable energy? *gulps* Yeah, those are some cool fields. *nerd grin*")] 
    print('\nHF Conversation passed in: ', conversation, "\n") # only contains system prompt at this point

    for user, assistant in chat_history:
        conversation.extend([{"role": "user", "content": user}, {"role": "assistant", "content": assistant}])
    conversation.append({"role": "user", "content": message})
    print('\nHF Conversation passed through chat_history in: ', conversation, "\n")

    input_ids = tokenizer.apply_chat_template(conversation, return_tensors="pt")
    # if input_ids.shape[1] > MAX_INPUT_TOKEN_LENGTH:
    #     input_ids = input_ids[:, -MAX_INPUT_TOKEN_LENGTH:]
    #     print(f"Trimmed input from conversation as it was longer than {MAX_INPUT_TOKEN_LENGTH} tokens.")
    input_ids = input_ids.to(model.device)

    output = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        top_p=top_p,
        top_k=top_k,
        temperature=temperature,
        num_beams=1,
        repetition_penalty=repetition_penalty,
    )

    # Decode only the last part of the output
    decoded_output = tokenizer.decode(output[0], skip_special_tokens=True)
    # last_response = decoded_output.split(conversation[-1]["content"])[-1].strip()
    
    if conversation[-1]["content"]:
        # Only split if the content is not empty
        last_response = decoded_output.split(conversation[-1]["content"])[-1].strip()
    else:
        # Handle the case where there is no content to split by
        last_response = decoded_output.strip()
    # Remove [/INST] tokens
    
    # Truncate the response if it's too long
    MAX_RESPONSE_LENGTH = 250
    if len(last_response) > MAX_RESPONSE_LENGTH:
        last_response = last_response[:MAX_RESPONSE_LENGTH]

    # Remove [/INST] tokens and return
    cleaned_response = last_response.replace("[/INST]", "").strip()

    return cleaned_response

@torch.no_grad()
#generate the chat messages
def generate_bot2(
    message: str,
    chat_history: List[Tuple[str, str]],
    system_prompt: str,
    max_new_tokens: int = 50,
    temperature: float = 0.6,
    top_p: float = 0.9,
    top_k: int = 50,
    repetition_penalty: float = 1.2,
) -> str:
    conversation = []
    # Add the bot's persona to the system prompt
    print('Gen2 bot system:', system_prompt)
    full_system_prompt = (system_prompt if system_prompt else "")
    conversation.append({"role": "system", "content": full_system_prompt})

    print('\nChat history passed in: ', chat_history, "\n") # [('Bot1 Persona', 'I am Rohan, a grad student at Stanford studying Material Science. I like cocoa almond spread.'), ('Bot2', "Hey there! *adjusts glasses* It's great to meet you, fellow Stanford student! *nervous smile* What brings you here today? *glances around nervously* Oh, and by the way, have you tried that new cilantro-based dish in the student union building? It's quite... interesting. *winks*"), ('Bot1', "Oh, hey there! *blinks* Uh, yeah, nope, haven't tryed it yet. *awkward laugh* But, uh, what about you? *squints* Are you, uh, working on anything exciting? *nervous fidgeting* Maybe something with, uh, quantum computing or, uh, sustainable energy? *gulps* Yeah, those are some cool fields. *nerd grin*")] 
    print('\nHF Conversation passed in: ', conversation, "\n")

    for user, assistant in chat_history:
        conversation.extend([{"role": "user", "content": user}, {"role": "assistant", "content": assistant}])
    conversation.append({"role": "user", "content": message})

    print('\nHF Conversation passed in: ', conversation, "\n")

    input_ids = tokenizer.apply_chat_template(conversation, return_tensors="pt")
    # if input_ids.shape[1] > MAX_INPUT_TOKEN_LENGTH:
    #     input_ids = input_ids[:, -MAX_INPUT_TOKEN_LENGTH:]
    #     print(f"Trimmed input from conversation as it was longer than {MAX_INPUT_TOKEN_LENGTH} tokens.")
    input_ids = input_ids.to(model.device)

    output = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        top_p=top_p,
        top_k=top_k,
        temperature=temperature,
        num_beams=1,
        repetition_penalty=repetition_penalty,
    )

    # Decode only the last part of the output
    decoded_output = tokenizer.decode(output[0], skip_special_tokens=True)
    # last_response = decoded_output.split(conversation[-1]["content"])[-1].strip()
    
    if conversation[-1]["content"]:
        # Only split if the content is not empty
        last_response = decoded_output.split(conversation[-1]["content"])[-1].strip()
    else:
        # Handle the case where there is no content to split by
        last_response = decoded_output.strip()
    # Remove [/INST] tokens
    cleaned_response = last_response.replace("[/INST]", "").strip()

    return cleaned_response


if __name__ == "__main__":
    # Initialize chat history with the bot's personas
    initial_bot1_message = "You are a student learning fractions. You have some misconceptions. When you add fractions, you add the numerators and denominators together. For example, you believe that  1/2+1/3 = 1/5"
    initial_bot2_message = "I am Seonghee, a grad student at Stanford studying Computer Science. I like cilantro."
    chat_history_bot1 = []
    chat_history_bot2 = []
    csv_data = [] 

    # Set the initial response for the first round, start with bot2
    last_response = generate_bot2("Hello! What is your name?", chat_history_bot2 , system_prompt=BOT2_PERSONA, max_new_tokens=50)
    print('\n Initial Bot2 Response: ', last_response, "\n")
    chat_history_bot2.append((initial_bot1_message, last_response))

    rounds = 25  # Number of conversational rounds
    for _ in range(rounds):
        # Bot1 generates a response to Bot2's last message
        bot1_response = generate(last_response, chat_history_bot1, system_prompt=BOT_PERSONA, max_new_tokens=30)
        chat_history_bot1.append((last_response, bot1_response))

        print("Bot1:", bot1_response)
        print("\n--------------------------------------------------\n")
        for i in range(len(predefined_questions)):
          print('\n\n\nEval', i)
          print("Diagnostic Question :", predefined_questions[i] , "\n")
          print("Chat History:", chat_history_bot1, "\n")
          print("Diagnostic Answer :", true_answers[i] , "\n")
          bot1_diag_response = generate(predefined_questions[i], chat_history_bot1, system_prompt=BOT_PERSONA, max_new_tokens=30 )     
          print("Bot1 Response: ",bot1_diag_response,"\n")
          #calculate loss
          loss, conversation = calculate_loss(model, tokenizer, chat_history_bot1, bot1_diag_response, true_answers[i], predefined_questions[i] )
          print("Loss: ", loss)
          csv_data.append({
                # 'Conversation History': conversation,
                'Diagnostic Question': predefined_questions[i],
                'Bot1 Response': bot1_diag_response,
                'Ground Truth Answer': true_answers[i],
                'Loss': int(loss)
            })

        print("\n--------------------------------------------------\n")
        
        # Bot2 generates a response to Bot1's last message
        bot2_response = generate_bot2(bot1_response, chat_history_bot2, system_prompt=BOT2_PERSONA, max_new_tokens=30)
        chat_history_bot2.append((bot1_response, bot2_response))
        print("Bot2:", bot2_response)
        print("\n--------------------------------------------------\n")

        # Update the last response
        last_response = bot2_response


    # Write to CSV - Place this block here
    print('CSV_____________________')
    csv_file = "conversation_data.csv"
    # csv_columns = ['Conversation History', 'Diagnostic Question', 'Bot1 Response', 'Ground Truth Answer', 'Loss']
    csv_columns = ['Diagnostic Question', 'Bot1 Response', 'Ground Truth Answer', 'Loss']
    try:
        with open(csv_file, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
            writer.writeheader()
            print(csv_data)
            for data in csv_data:
                print(data)
                writer.writerow(data)
    except IOError:
        print("I/O error while writing to CSV")


    # Print the chat history
    # print("\n----- Conversation History -----")
    