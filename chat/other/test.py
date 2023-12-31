from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
)
import os
import transformers
import torch

model_id = "../Llama-2-7b-chat-hf"
dotenv.load_dotenv('../.env')
HF_ACCESS_TOKEN = os.getenv('HF_ACCESS_TOKEN')

model = AutoModelForCausalLM.from_pretrained(model_id, use_auth_token=HF_ACCESS_TOKEN, torch_dtype=torch.float16, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_id, use_auth_token=HF_ACCESS_TOKEN)
pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    tokenizer = tokenizer,
    torch_dtype=torch.float32,
    device_map="auto",
    token = HF_ACCESS_TOKEN
)


initial_prompt = """

<s>[INST] <<SYS>>
You are the Persona and Bot1. Always act as you are the persona. Respond to Bot2.
Persona:  My name is Jack i like to party.  my major is business. i am in college.
<</SYS>>

[/INST]

"""


bot2_initial_prompt = """

<s>[INST] <<SYS>>
You are the Persona and Bot2. Always act as you are the persona. Respond to Bot1.
Persona:  My name is Susan i like to knit.  my major is art. i am a professor.
<</SYS>>

[/INST]

"""

def generate(prompt ,bot):
  """Take in convo history and generate response to last utterance

  conversational_history -- list

  """
  if bot=='bot2':
      print('here')
      sequences_2 = pipeline(
        prompt,
        do_sample=True,
        top_k=10,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
        max_length=1024,
        )
      # parsing part of bot response, removing the prompt returned
      for seq in sequences_2:
        print( 'checking resposne bot2, ', seq['generated_text'], '\n')
        response = seq['generated_text']
        result = response[1 + len(prompt):1 + len(prompt)+100]

  else:
    #get model response/bot response
    sequences = pipeline(
      prompt,
      do_sample=True,
      top_k=10,
      num_return_sequences=1,
      eos_token_id=tokenizer.eos_token_id,
      max_length=1024,
      )

    print('Model Output: ',sequences, "\n")

    # parsing part of bot response, removing the prompt returned
    for seq in sequences:
      response = seq['generated_text']
      result = response[1+ len(prompt): 1 + len(prompt)+100]

    print('Conversation Output: ',result, "\n")

  #return the removed prompt, true bot response
  return result


# 
conversation_history=[initial_prompt]
for i in range(5):
  print('ROUND ', i, "\n")
  if i ==0:
    bot1_prompt = initial_prompt
    bot2_prompt = bot2_initial_prompt
  else:
    bot1_prompt = initial_prompt + conversation_history[-1]
    bot2_prompt = bot2_initial_prompt + conversation_history[-1]
  # initial rpompt 
  print('BOT 1 PROMPT: ', bot1_prompt ,"\n")
  bot1_response = generate(bot1_prompt,'bot1')
  #answer to what is your name
  print('BOT 1 RESPONSE: ', bot1_response ,"\n")
  # adding that to convo history
  conversation_history.append(f"Bot1: {bot1_response}")
  print('BOT 1 CONVO HISTORY: ', conversation_history,"\n")

  # bot1 answer to what is your name
  print('BOT 2 PROMPT: ', bot2_prompt ,"\n")
  bot2_response = generate(bot2_prompt, 'bot2')
  print('BOT 2 RESPONSE: ', bot2_response ,"\n")
  conversation_history.append(f"Bot2: {bot2_response}")
  print('BOT 2 CONVO HISTORY: ', conversation_history ,"\n")

# for seq in sequences:
#     conversation_history=[initial_prompt]
#     conversation_history.append('Bot1: ',seq['generated_text'])

#     print(f"Result: {seq['generated_text']}")
