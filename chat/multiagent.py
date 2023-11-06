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


# Function to initialize and return the LLM chain with a specified template
def initialize_bot(persona_template):

    prompt = PromptTemplate(
        input_variables=["history", "input"],
        template=persona_template,
        template_format="jinja2"
    )

    bot = ConversationChain(
        llm=llm,
        memory=ConversationBufferMemory(),
        prompt=prompt,
        verbose=False
    )
    return bot

template = """

Do not write any emojis.
The AI's Persona Description: 
<s>[INST] <<SYS>>
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

neg_template = """
Do not write any emojis.
The AI's Persona Description: 
<s>[INST] <<SYS>>
i dislike social gatherings and parties. my major is computer science. i am a graduate. i prefer staying indoors and find solace in solitude. i work full time at a tech company.
i am a computer science major and have a full time job
i have already earned my master's in computer science
i work full time and no longer involved in part-time jobs
i rarely attend social events or parties anymore
i value peace and quiet, and prefer spending time alone or in small gatherings
i have graduated and am currently working in the tech industry
i am not very fond of outdoor activities like going to the beach
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

# Initializing two bots with different personalities
bot1 = initialize_bot(template)
bot2 = initialize_bot(neg_template)

# Function to make the bots talk to each other
def bots_conversation(bot1, bot2, rounds=5):
    conversation_history = ""
    for _ in range(rounds):
        # Bot 1 sends a message to Bot 2
        output_bot1 = bot1.predict(input=conversation_history)
        conversation_history += f"Bot 1: {output_bot1}\n"
        
        # Bot 2 responds
        output_bot2 = bot2.predict(input=conversation_history)
        conversation_history += f"Bot 2: {output_bot2}\n"

        print("Bot 1:", output_bot1)
        print("Bot 2:", output_bot2)
        print()

    return conversation_history

def save_conversation_to_csv(conversation_history, file_path):
    # Split the conversation into lines
    lines = conversation_history.strip().split('\n')
    
    # Open the file in write mode
    with open(file_path, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        
        # Write the headers
        writer.writerow(["Speaker", "Dialogue"])
        
        # Write each line of dialogue to the CSV
        for line in lines:
            # Assuming each line starts with "Bot X:" where X is the bot number
            speaker, dialogue = line.split(':', 1)
            writer.writerow([speaker.strip(), dialogue.strip()])


# ... rest of your script ...

# Start the conversation
conversation_history = bots_conversation(bot1, bot2)

# Specify the path where you want to save the CSV
csv_file_path = 'conversation_history.csv'

# Save the conversation to the specified CSV file
save_conversation_to_csv(conversation_history, csv_file_path)
