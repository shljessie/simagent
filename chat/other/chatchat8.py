import os
import dotenv
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Tuple
from diagnostic3 import calculate_loss
import csv

predefined_questions = ["Hello! What is your name?", "What do you like?", "What is your major?"]

true_answers = ["Hey there! My name is Rohan","I like coco almond spread","I'm a grad student at Stanford studying Material Science."]

MAX_INPUT_TOKEN_LENGTH = int(os.getenv("MAX_INPUT_TOKEN_LENGTH", "4096"))
# Define the bot's persona
BOT_PERSONA = """
[SYSTEM]
I am Rohan a grad student at Stanford studying Material Science. I like cocoalmond spread.
[/SYSTEM]
"""

# Define the bot's persona
BOT2_PERSONA = """
[SYSTEM]
I am Seonghee a grad student at Stanford studying Computer Science. I like cilantro.
[/SYSTEM]
"""

# Load environment variables and model
if torch.cuda.is_available():
    model_id = "../Llama-2-7b-chat-hf"
    dotenv.load_dotenv('../.env')
    HF_ACCESS_TOKEN = os.getenv('HF_ACCESS_TOKEN')
    model = AutoModelForCausalLM.from_pretrained(model_id, use_auth_token=HF_ACCESS_TOKEN, torch_dtype=torch.float16, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_auth_token=HF_ACCESS_TOKEN)
    tokenizer.use_default_system_prompt = False

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
    # Add the bot's persona to the system prompt
    full_system_prompt = BOT_PERSONA + (system_prompt if system_prompt else "")
    conversation.append({"role": "system", "content": full_system_prompt})

    for user, assistant in chat_history:
        conversation.extend([{"role": "user", "content": user}, {"role": "assistant", "content": assistant}])
    conversation.append({"role": "user", "content": message})


    input_ids = tokenizer.apply_chat_template(conversation, return_tensors="pt")
    if input_ids.shape[1] > MAX_INPUT_TOKEN_LENGTH:
        input_ids = input_ids[:, -MAX_INPUT_TOKEN_LENGTH:]
        print(f"Trimmed input from conversation as it was longer than {MAX_INPUT_TOKEN_LENGTH} tokens.")
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

    # testfix

    if conversation[-1]["content"]:
        # Only split if the content is not empty
        last_response = decoded_output.split(conversation[-1]["content"])[-1].strip()
    else:
        # Handle the case where there is no content to split by
        last_response = decoded_output.strip()

    # Remove [/INST] tokens
    cleaned_response = last_response.replace("[/INST]", "").strip()

    return cleaned_response


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
    full_system_prompt = BOT2_PERSONA + (system_prompt if system_prompt else "")
    conversation.append({"role": "system", "content": full_system_prompt})

    for user, assistant in chat_history:
        conversation.extend([{"role": "user", "content": user}, {"role": "assistant", "content": assistant}])
    conversation.append({"role": "user", "content": message})


    input_ids = tokenizer.apply_chat_template(conversation, return_tensors="pt")
    if input_ids.shape[1] > MAX_INPUT_TOKEN_LENGTH:
        input_ids = input_ids[:, -MAX_INPUT_TOKEN_LENGTH:]
        print(f"Trimmed input from conversation as it was longer than {MAX_INPUT_TOKEN_LENGTH} tokens.")
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
    initial_bot1_message = "I am Rohan, a grad student at Stanford studying Material Science. I like cocoa almond spread."
    initial_bot2_message = "I am Seonghee, a grad student at Stanford studying Computer Science. I like cilantro."
    chat_history = [("Bot1 Persona", initial_bot1_message)]
    csv_data = [] 

    # Set the initial response for the first round
    last_response = "Hello!"  # Starting with Bot2's persona message

    rounds = 50  # Number of conversational rounds
    for _ in range(rounds):
        # Bot1 generates a response to Bot2's last message
        # print('last_response: ', last_response)
        # print('chat_history: ,', chat_history)

        bot1_response = generate(last_response, chat_history, system_prompt="", max_new_tokens=50)
        chat_history.append(("Bot1", bot1_response))

        print("Bot1:", bot1_response)
        print("\n--------------------------------------------------\n")
        for i in range(len(predefined_questions)):
          print('\n\n\nEval', i)
          print("Diagnostic Question :", predefined_questions[i] , "\n")
          print("Chat History:", chat_history, "\n")
          print("Diagnostic Answer :", true_answers[i] , "\n")
          
          bot1_diag_response = generate(predefined_questions[i], chat_history, system_prompt="", max_new_tokens=50 )     
          print("Bot1 Response: ",bot1_diag_response,"\n")
          #calculate loss
          flattened_history = ' '.join([f"{speaker}: {text}" for speaker, text in chat_history])
          loss = calculate_loss(model, tokenizer, flattened_history, bot1_diag_response, true_answers[i] )
          print("Loss: ", loss)
          csv_data.append({
                'Conversation History': chat_history,
                'Diagnostic Question': predefined_questions[i],
                'Bot1 Response': bot1_diag_response,
                'Ground Truth Answer': true_answers[i],
                'Loss': loss
            })

        print("\n--------------------------------------------------\n")
        
        # Bot2 generates a response to Bot1's last message
        bot2_response = generate_bot2(bot1_response, chat_history, system_prompt="", max_new_tokens=50)
        chat_history.append(("Bot2", bot2_response))
        print("Bot2:", bot2_response)
        print("\n--------------------------------------------------\n")

        # Update the last response
        last_response = bot2_response


    # Write to CSV - Place this block here
    csv_file = "conversation_data_take2.csv"
    csv_columns = ['Conversation History', 'Diagnostic Question', 'Bot1 Response', 'Ground Truth Answer', 'Loss']
    try:
        with open(csv_file, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
            writer.writeheader()
            for data in csv_data:
                writer.writerow(data)
    except IOError:
        print("I/O error while writing to CSV")


    # Print the chat history
    print("\n----- Conversation History -----")
    for sender, msg in chat_history:
        print(f"{sender}: {msg}\n")
        print("--------------------------------------------------")
