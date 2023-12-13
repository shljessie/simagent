import os
import dotenv
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Tuple
from diagnostic_backprop import calculate_loss
from torch.optim import Adam
import csv
from torch.utils.data import DataLoader, TensorDataset

predefined_questions = ["What is your name?", "How old are you?", "What is your major?"]
true_answers = ["My name is Rohan", "I am 22 years old", "My major is Material Science"]

BOT_PERSONA = """
[SYSTEM]
You are Rohan a grad student at Stanford studying Material Science. You are 22 years old.
[SYSTEM]
Respond with one sentence only.
"""

BOT2_PERSONA = """
[SYSTEM]
You are Seonghee a grad student at Stanford studying Computer Science. You are 23 years old. Respond with one sentence only.
[/SYSTEM]
Respond with one sentence only.
"""

if torch.cuda.is_available():
    model_id = "../Llama-2-7b-chat-hf"
    dotenv.load_dotenv('../.env')
    HF_ACCESS_TOKEN = os.getenv('HF_ACCESS_TOKEN')
    model = AutoModelForCausalLM.from_pretrained(model_id, use_auth_token=HF_ACCESS_TOKEN, torch_dtype=torch.float16, device_map="auto")
    model.bfloat16()
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_auth_token=HF_ACCESS_TOKEN)
    tokenizer.use_default_system_prompt = False
    optimizer = Adam(model.parameters(), lr=0.001)
    model.train()

MAX_INPUT_TOKEN_LENGTH = int(os.getenv("MAX_INPUT_TOKEN_LENGTH", "400"))

def generate_batched(
    messages: List[str],
    chat_history: List[Tuple[str, str]],
    system_prompt: str,
    max_new_tokens: int = 10,
    temperature: float = 0.8,
    top_p: float = 0.9,
    top_k: int = 50,
    repetition_penalty: float = 1.2,
) -> List[str]:
    conversation = []

    full_system_prompt = (system_prompt if system_prompt else "")
    conversation.append({"role": "system", "content": full_system_prompt})

    for user, assistant in chat_history:
        conversation.extend([{"role": "user", "content": user}, {"role": "assistant", "content": assistant}])

    for message in messages:
        conversation.append({"role": "user", "content": message})

    input_ids = tokenizer.apply_chat_template(conversation, return_tensors="pt")
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

    decoded_outputs = [tokenizer.decode(output[i], skip_special_tokens=True) for i in range(len(messages))]

    cleaned_responses = [output.split("[/INST]")[0].strip() for output in decoded_outputs]

    return cleaned_responses

if __name__ == "__main__":
    initial_bot1_message = "You are Rohan a grad student at Stanford studying Material Science. You are 22 years old."
    initial_bot2_message = "I am Seonghee, a grad student at Stanford studying Computer Science. I like cilantro."
    chat_history_bot1 = []
    chat_history_bot2 = []
    csv_data = []

    last_response = generate_batched(["Hello! What is your name?"], chat_history_bot2 , system_prompt=BOT2_PERSONA, max_new_tokens=30)
    chat_history_bot2.append((initial_bot1_message, last_response[0]))

    rounds = 30
    for _ in range(rounds):
        bot1_responses = generate_batched([last_response[0]], chat_history_bot1, system_prompt=BOT_PERSONA, max_new_tokens=30)
        chat_history_bot1.append((last_response[0], bot1_responses[0]))

        print("Bot1:", bot1_responses[0])
        print("\n--------------------------------------------------\n")

        # Diagnostic Question
        for i in range(len(predefined_questions)):
            bot1_diag_responses = generate_batched([predefined_questions[i]], chat_history_bot1, system_prompt=BOT_PERSONA, max_new_tokens=30)

            # Assuming calculate_loss returns a list of losses
            losses = calculate_loss(model, tokenizer, chat_history_bot1, bot1_diag_responses[0], true_answers[i], predefined_questions[i])

            csv_data.append({
                'Diagnostic Question': predefined_questions[i],
                'Bot1 Response': bot1_diag_responses[0],
                'Ground Truth Answer': true_answers[i],
                'Loss': float(losses[0].item()),  # You may need to adjust this part based on your loss calculation
            })

            optimizer.zero_grad()
            losses[0].backward()
            optimizer.step()
            print('Backward Step Complete')

        print("\n--------------------------------------------------\n")

        bot2_responses = generate_batched([bot1_responses[0]], chat_history_bot2, system_prompt=BOT2_PERSONA, max_new_tokens=30)
        chat_history_bot2.append((bot1_responses[0], bot2_responses[0]))

        print("Bot2:", bot2_responses[0])
        print("\n--------------------------------------------------\n")

        last_response = bot2_responses[0]

    print('CSV_____________________')

    def clean_string(s):
        return s.encode('ascii', 'ignore').decode('ascii')

    csv_file = f"loss_7b_backprop.csv"
    csv_columns = ['Diagnostic Question', 'Bot1 Response', 'Ground Truth Answer', 'Loss']

    try:
        with open(csv_file, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
            writer.writeheader()
            for data in csv_data:
                try:
                    cleaned_data = {k: clean_string(v) if isinstance(v, str) else v for k, v in data.items()}
                    writer.writerow(cleaned_data)
                except UnicodeEncodeError as e:
                    print("Error with data:", data)
                    print("Error message:", e)

    except IOError:
        print("I/O error while writing to CSV")

    # Save the trained model (optional)
    model.save_pretrained("./backprop_llama2.py")
