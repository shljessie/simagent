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

# Configurations
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

template = """

Do not write any emojis.
The Persona Description: 
<s>[INST] <<SYS>>
My name is Jack
i like to party. my major is business. i am in college. i love the beach. i work part time at a pizza restaurant.
i am a business major but have a part time job
i am trying to get my ba in finance
no still in school work at pizza hut part time
i really hope they have a frat party again soon
i used to party a lot
it is fun i cant get enough
i am an undergrad in college
i love going to the beach
<</SYS>>


Do not write any emojis. Only respond with spoken text. Do not include terms like *smiling* *nods* *excitedly*

Current conversation:
{{ history }}

{% if history %}
    <s>[INST] Human: {{ input }} [/INST] AI: </s>
{% else %}
    Human: {{ input }} [/INST] AI: </s>
{% endif %} 
"""

# Function to initialize and return the LLM chain with a specified template
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


template_two = """
You are a bot that asks questions
"""

# Predefined questions for bot2
predefined_questions = [
    "What is your name?",
    "What do you like to do?",
    "What is your major?",
    "Do you like to hang out with friends?",
    "How old are you?"
]


true_answers= [
    "Jack",
    "I like to party. I like to surf",
    "Business",
    "Yes",
    "I am 22 years old"
]



# Initializing two bots with different personalities
bot1 = initialize_bot(template)
# bot2 = initialize_bot(template_two)


def bots_conversation(bot1, predefined_questions):
    conversation_history =""
    loss_scores = [] 
    for i in range(len(predefined_questions)): 
      conversation_history += f" Bot2: {predefined_questions[i]} \n"
      bot1_output = bot1.predict(input=predefined_questions[i])
      conversation_history += f"Bot1: " + bot1_output + "\n"

      # Call diagnostic.py for each response
      loss = calculate_loss(model,tokenizer,conversation_history,true_answers[i] )
      loss_scores.append(loss)

      print( f" Bot2: {predefined_questions[i]} \n")
      print( f"Bot1: " + bot1_output + "\n" )
      print( 'True Answer: ',true_answers[i]+ "\n" )
      print( f"Loss for the response '{bot1_output}': {loss}" + "\n")

    return conversation_history, loss_scores

def save_conversation_to_csv(conversation_history, loss_scores, file_path):
    lines = conversation_history.strip().split('\n')

    with open(file_path, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        
        # Write the headers
        writer.writerow(["Speaker", "Dialogue", "Loss"])
        
        # Assume that for every two lines, the first is Bot2 and the second is Bot1
        for i in range(0, len(lines), 2):
            writer.writerow([lines[i].split(':', 1)[0].strip(), lines[i].split(':', 1)[1].strip(), ""])  # Bot2's line has no loss
            writer.writerow([lines[i+1].split(':', 1)[0].strip(), lines[i+1].split(':', 1)[1].strip(), loss_scores[i//2]])  # Bot1's line has loss


# Start the conversation
conversation_history, loss_scores = bots_conversation(bot1, predefined_questions)

# Specify the path where you want to save the CSV
csv_file_path = 'conversation_history.csv'

# Save the conversation to the specified CSV file
save_conversation_to_csv(conversation_history, loss_scores, csv_file_path)
