from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline
import dotenv
import os
from langchain.llms import HuggingFacePipeline
from langchain.prompts.prompt import PromptTemplate

from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
import gradio as gr

dotenv.load_dotenv('/.env')
HF_ACCESS_TOKEN = os.getenv('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')

model_id = '../Llama-2-7b-chat-hf'

bnb_config = BitsAndBytesConfig(
    bnb_4bit_compute_dtype = 'float16',
    bnb_4bit_quant_type='nf4',
    load_in_4bit=True,
)

# Load model configuration
model_config = AutoConfig.from_pretrained(
    model_id,
    use_auth_token=HF_ACCESS_TOKEN
)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    config=model_config,
    device_map='auto',
    quantization_config=bnb_config,
    use_auth_token=HF_ACCESS_TOKEN
)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    model_id,
    use_auth_token=HF_ACCESS_TOKEN
)

# Set model into evaluation mode (optimizes inference)
model.eval()

pipe = pipeline(
    model=model,
    task='text-generation',
    tokenizer=tokenizer
)

llm = HuggingFacePipeline(pipeline=pipe)

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

# Start the conversation
def predict(message: str, history: str):
    response = conversation.predict(input=message)

    return response

# Set up the user interface
interface = gr.ChatInterface(
    clear_btn=None,
    fn=predict,
    retry_btn=None,
    undo_btn=None,
)

# Launch the user interface
interface.launch(
    height=600,
    inline=True,
    share=True,
    width=800
)