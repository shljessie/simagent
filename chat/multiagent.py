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
template_two = template_data['template_two']

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

def diagnostic_q(bot1, predefined_questions, conversational_history):
    diagnostic_history ="" 
    loss_scores = [] 
    perplexities = []
    for i in range(len(predefined_questions)): 
      diagnostic_history += f" Bot2: {predefined_questions[i]} \n" # diagnostic question 
      bot1_output = bot1.predict(input=predefined_questions[i])
      diagnostic_history += f"Bot1: " + bot1_output + "\n" # diagnostic question answer
       # conversational history = bot chat -1 
      loss , perplexity = calculate_loss(model, tokenizer, conversational_history, true_answers[i])
      loss_scores.append(loss)
      perplexities.append(perplexity)

      print("\n")
      print( f"Bot2: {predefined_questions[i]} \n")
      print( f"Bot1: " + bot1_output + "\n" )
      print( 'True Answer: ',true_answers[i]+ "\n" )
      print( f"Loss: {loss}" + "\n")
      print( f"Perplexity: {perplexity}" + "\n")

    return loss_scores, perplexities




def bot_convo(bot1, bot2,round):

  bot_convo =""
  #default starting convo
  bot1_output = bot1.predict(input=predefined_questions[0])
  bot_convo =  f"Bot1: " + bot1_output + "\n"
  for i in range(round):
    bot2_output = bot2.predict(input=bot1_output)
    bot1_output = bot1.predict(input=bot2_output)
    bot_convo +=  f" Bot2: {predefined_questions[i]} \n" + f"Bot1: " + bot1_output

    print( f"Bot1: " + bot1_output + "\n" )
    print( f"Bot2: " + bot2_output + "\n" )

    # ask the diagnostic questions 
    loss_scores = diagnostic_q(bot1, predefined_questions, bot_convo)
    bot_convo +=  f" Loss Score: {loss_scores} \n"

  return bot_convo

#Initialize bot
bot1 = initialize_bot(template)
bot2 = initialize_bot(template_two)

# Start the conversation
bot_conversation =  bot_convo(bot1, bot2, 5)

# Specify the path where you want to save the CSV
csv_file_path = 'conversation_history.csv'