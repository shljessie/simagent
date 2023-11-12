from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
)
import transformers
import torch

model_id = "../Llama-2-7b-chat-hf"
HF_ACCESS_TOKEN = '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'

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
You are the Persona. Always act as you are the persona.
Persona:  My name is Jack i like to party.  my major is business. i am in college.
<</SYS>>

[/INST]

"""


bot2_initial_prompt = """

<s>[INST] <<SYS>>
You are the Persona. Always act as you are the persona.
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
      sequences = pipeline(
        prompt,
        do_sample=True,
        top_k=10,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
        max_length=250,
        )
      # parsing part of bot response, removing the prompt returned
      for seq in sequences:
        response = seq['generated_text']
        result = response[1 + len(bot2_prompt):]

  else:
    #get model response/bot response
    sequences = pipeline(
      prompt,
      do_sample=True,
      top_k=10,
      num_return_sequences=1,
      eos_token_id=tokenizer.eos_token_id,
      max_length=500,
      )

    print('Model Output: ',sequences, "\n")

    # parsing part of bot response, removing the prompt returned
    for seq in sequences:
      response = seq['generated_text']
      marker = "[/INST]\n\n"
      index = response.find(marker)
      if index != -1:
        result = response[index + len(marker):]

    print('Conversation Output: ',result, "\n")

  #return the removed prompt, true bot response
  return result


# 
conversation_history=[initial_prompt]

bot1_prompt = initial_prompt
# initial rpompt 
print('BOT 1 PROMPT: ', bot1_prompt ,"\n")
bot1_response = generate(bot1_prompt,'bot1')
#answer to what is your name
print('BOT 1 RESPONSE: ', bot1_response ,"\n")
# adding that to convo history
conversation_history.append(f"Bot1: {bot1_response}")
print('BOT 1 CONVO HISTORY: ', conversation_history,"\n")

# bot1 answer to what is your name
bot2_prompt = bot2_initial_prompt + conversation_history[-1]
print('BOT 2 PROMPT: ', bot2_prompt ,"\n")
bot2_response = generate(bot2_prompt, 'bot2')
print('BOT 2 RESPONSE: ', bot2_response ,"\n")
conversation_history.append(f"Bot2: {bot2_response}")
print('BOT 2 CONVO HISTORY: ', bot2_response ,"\n")

# for seq in sequences:
#     conversation_history=[initial_prompt]
#     conversation_history.append('Bot1: ',seq['generated_text'])

#     print(f"Result: {seq['generated_text']}")
