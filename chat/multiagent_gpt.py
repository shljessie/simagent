import os
import torch
import math
import dotenv
import torch
import csv
import subprocess
from diagnostic import calculate_loss  #import the calc loss function
import json
from transformers import (
    AutoConfig, 
    AutoModelForCausalLM, 
    AutoTokenizer, 
    BitsAndBytesConfig, 
    pipeline
)

print("Current Working Directory:", os.getcwd())
# import data
with open('./persona_template.json', 'r') as f:
    template_data = json.load(f)
template = template_data['template']
template_two = template_data['template_two']

with open('./questions.json', 'r') as f:
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

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained('gpt2')
tokenizer = AutoTokenizer.from_pretrained('gpt2')

generator = pipeline(
    model=model,
    task='text-generation',
    tokenizer=tokenizer
)

def initialize_bot(template):

    generated = generator(template, max_length=50, num_return_sequences=1)
    return generated

def diagnostic_q(bot1, predefined_questions):
    conversation_history = get_persona(template) + "\n "
    loss_scores = [] 
    for i in range(len(predefined_questions)): 
      conversation_history += f" Bot2: {predefined_questions[i]} \n"
      bot1_output = generator(conversation_history, max_length=50, num_return_sequences=1)
      conversation_history += f"Bot1: " + bot1_output + "\n"

      # Call diagnostic.py for each response
      loss = calculate_loss(model,tokenizer,conversation_history,true_answers[i] )
      loss_scores.append(loss)
      print("\n")
      print( f"Bot2: {predefined_questions[i]} \n")
      print( f"Bot1: " + bot1_output + "\n" )
      print( 'True Answer: ',true_answers[i]+ "\n" )
      print( f"Loss for the response: {loss}" + "\n")

    return loss_scores

def bot_convo(bot1, bot2,round):
  #default starting convo
  bot1_output = bot1.predict(input=predefined_questions[i])
  bot_convo =  f"Bot1: " + bot1_output + "\n"
  for i in range(round):
    bot2_output = bot2.predict(input=bot1_output)
    bot1_output = bot1.predict(input=bot2_output)
    bot_convo +=  f"Bot1: " + bot1_output + "\n" + f" Bot2: {predefined_questions[i]} \n"

    print( f"Bot1: " + bot1_output + "\n" )
    print( f"Bot2: " + bot2_output + "\n" )

    # ask the diagnostic questions 
    loss_scores = diagnostic_q(bot1, predefined_questions)
    bot_convo +=  f" Loss Score: {loss_scores} \n"

  return bot_convo

#Initialize bot
bot1 = initialize_bot(template)
bot2 = initialize_bot(template_two)

# Start the conversation
bot_conversation =  bot_convo(bot1, bot2, 5)

# Specify the path where you want to save the CSV
csv_file_path = 'conversation_history.csv'