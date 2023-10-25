import torch
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline
import gradio as gr
import dotenv
import os
import pandas as pd
import numpy as np

from langchain.llms import HuggingFacePipeline
from langchain.prompts.prompt import PromptTemplate
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory

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

def get_next_token_predictions(text, model, tokenizer):
    tokens = tokenizer.encode(text, return_tensors="pt")
    tokens = torch.cat((torch.tensor([tokenizer.eos_token_id]), tokens[0])).reshape(1,-1)

    print('tokens',tokens)

    for i in np.arange(0,len(tokens[0])-1):
        outputs = model.generate(tokens[0][:i+1].reshape(1,-1), max_new_tokens=1, output_scores=True, return_dict_in_generate=True, pad_token_id=tokenizer.eos_token_id)
        scores = F.softmax(outputs.scores[0], dim=-1)
        top_10 = torch.topk(scores, 10)
        df = pd.DataFrame()
        a = scores[0][tokens[0][i+1]]
        b = top_10.values
        df["probs"] = list(np.concatenate([a.reshape(-1,1).numpy()[0], b[0].numpy()]))
        diff = 100*(df["probs"].iloc[0]-df["probs"].iloc[1])
        if np.abs(diff)<1:
          color = "mystronggreen"
        elif np.abs(diff)<10:
          color = "mygreen"
        elif np.abs(diff)<20:
          color = "myorange"
        elif np.abs(diff)<30:
          color = "myyellow"
        else:
          color = "myred"
        df["probs"] = [f"{value:.2%}" for value in df["probs"].values]
        aux = [tokenizer.decode(tokens[0][i+1])] + [tokenizer.decode(top_10.indices[0][i]) for i in range(10)]
        df["predicted next token"] = aux
    
    print('probs:', df['probs'])
    print('next token:', df['predicted next token'])
    return df["probs"], df["predicted next token"]



template = """
<s>[INST] <<SYS>>
The following is a friendly conversation between a human and an AI. 
The AI  i like to party. my major is business. i am in college. i love the beach. i work part time at a pizza restaurant.
<</SYS>>

Current conversation:
{{ history }}

{% if history %}
    <s>[INST] Human: {{ input }} [/INST] AI: </s>
{% else %}
    Human: {{ input }} [/INST] AI: </s>
{% endif %} 
"""

prompt = PromptTemplate(
    input_variables=["history", "input"],
    template=template,
    template_format="jinja2"
)

# Initialize the conversation chain
conversation = ConversationChain(
    llm=llm,
    memory=ConversationBufferMemory(),
    prompt=prompt,
    verbose=False
)


def predict(message: str, history: str=""):

    print('input message', message)
    response = conversation.predict(input=message)

    next_token_predictions = get_next_token_predictions(message, model, tokenizer)
    full_response = f"{response}<br><br>Next token predictions: {next_token_predictions}"

    return full_response


interface = gr.Interface(
    fn=predict,
    inputs=["text", "text"],
    outputs=gr.outputs.HTML(),
)

interface.launch(
    height=600,
    inline=True,
    share=True,
    width=800
)

# # Set up the user interface
# interface = gr.ChatInterface(
#     clear_btn=None,
#     fn=predict,
#     retry_btn=None,
#     undo_btn=None,
# )

# # Launch the user interface
# interface.launch(
#     height=600,
#     inline=True,
#     share=True,
#     width=800
# )