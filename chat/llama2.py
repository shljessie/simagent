import torch
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline
import gradio as gr
import dotenv
import os

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
    outputs = model(tokens, output_scores=True)
    scores = F.softmax(outputs.scores[0], dim=-1)[-1]
    top_tokens = torch.topk(scores, 5)

    formatted_predictions = []
    for value, index in zip(top_tokens.values, top_tokens.indices):
        token = tokenizer.decode(index)
        probability = value.item()
        if probability > 0.9:
            color = "green"
        elif probability > 0.5:
            color = "yellow"
        else:
            color = "red"
        formatted_predictions.append(f"<span style='color:{color}'>{token} ({probability:.2f})</span>")

    return ' '.join(formatted_predictions)


# Template using jinja2 syntax
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