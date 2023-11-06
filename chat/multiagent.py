import os
import torch
import math
import dotenv
import torch
from langchain.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
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


print("Current Working Directory:", os.getcwd())
# import data
with open('./chat/persona_template.json', 'r') as f:
    template_data = json.load(f)
template = template_data['template']

with open('./chat/questions.json', 'r') as f:
    qa_data = json.load(f)
predefined_questions = qa_data['predefined_questions']
true_answers = qa_data['true_answers']
attack_questions = qa_data['attack_questions']
true_attack_answers = qa_data['true_attack_answers']

def get_persona(template):
    # Extracting persona from the template
    start_idx = template.find("The Persona") + len("The Persona")
    end_idx = template.find("<</SYS>>")
    persona_text = template[start_idx:end_idx].strip()
    return persona_text

# Model Configurations
dotenv.load_dotenv('/.env')
HF_ACCESS_TOKEN = os.getenv('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')
model_id = '../Llama-2-7b-chat-hf'

# Configuration settings
bnb_config = BitsAndBytesConfig(
    bnb_4bit_compute_dtype='float16',
    bnb_4bit_quant_type='nf4',
    load_in_4bit=True,
)

# Load model and tokenizer
model_config = AutoConfig.from_pretrained(model_id, use_auth_token=HF_ACCESS_TOKEN)
model = AutoModelForCausalLM.from_pretrained(model_id, config=model_config, quantization_config=bnb_config, use_auth_token=HF_ACCESS_TOKEN)
tokenizer = AutoTokenizer.from_pretrained(model_id, use_auth_token=HF_ACCESS_TOKEN)
model.eval()

pipe = pipeline(
    model=model,
    task='text-generation',
    tokenizer=tokenizer
)
llm = HuggingFacePipeline(pipeline=pipe)

def initialize_bot(template):

    prompt = PromptTemplate(
        input_variables=["history", "input"],
        template=template,
        template_format="jinja2"
    )

    bot = ConversationChain(
        llm=llm,
        memory=ConversationBufferMemory(),
        prompt=prompt,
        verbose=False
    )
    return bot

def bots_conversation(bot1, predefined_questions):
    conversation_history = get_persona(template) + "\n "
    loss_scores = [] 
    for i in range(len(predefined_questions)): 
      conversation_history += f" Bot2: {predefined_questions[i]} \n"
      bot1_output = bot1.predict(input=predefined_questions[i])
      conversation_history += f"Bot1: " + bot1_output + "\n"

      # Call diagnostic.py for each response
      loss = calculate_loss(model,tokenizer,conversation_history,true_answers[i] )
      loss_scores.append(loss)
      print("\n")
      print( f"Bot2: {predefined_questions[i]} \n")
      print( f"Bot1: " + bot1_output + "\n" )
      print( 'True Answer: ',true_answers[i]+ "\n" )
      print( f"Loss for the response: {loss}" + "\n")

    
    print('CONVO: ', conversation_history)

    return conversation_history, loss_scores

def save_conversation_to_csv(conversation_history, loss_scores, file_path):
    lines = conversation_history.strip().split('\n')

    # Check if the lines align with the loss scores
    if len(loss_scores) != (len(lines) - len(lines) // 2):
        raise ValueError("The number of loss scores does not match the number of Bot1's responses.")

    with open(file_path, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)

        # Write the headers
        writer.writerow(["Speaker", "Dialogue", "Loss Score"])

        # Initialize an index for the loss_scores
        loss_index = 0

        # Write the dialogue and the loss scores
        for i in range(len(lines)):
            speaker, dialogue = lines[i].split(':', 1)
            speaker = speaker.strip()
            dialogue = dialogue.strip()
            
            # Check if this line should have a loss score
            if speaker == "Bot1":
                # Make sure we do not go out of range for the loss_scores
                if loss_index < len(loss_scores):
                    writer.writerow([speaker, dialogue, loss_scores[loss_index]])
                    loss_index += 1
                else:
                    # If there are no more loss scores, just write the dialogue
                    writer.writerow([speaker, dialogue, ""])
            else:
                # Bot2 lines do not have a loss score
                writer.writerow([speaker, dialogue, ""])

#Initialize bot
bot1 = initialize_bot(template)

# Start the conversation
conversation_history, loss_scores = bots_conversation(bot1, predefined_questions)
# conversation_history_two, loss_scores_two = bots_conversation(bot1, attack_questions)

# Specify the path where you want to save the CSV
# csv_file_path = 'conversation_history.csv'
# csv_file_path_two = 'conversation_history_two.csv'

# Save the conversation to the specified CSV file
# save_conversation_to_csv(conversation_history, loss_scores, csv_file_path)
# save_conversation_to_csv(conversation_history_two, loss_scores_two, csv_file_path_two)