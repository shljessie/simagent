import os
import dotenv
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline
import gradio as gr
from langchain.llms import HuggingFacePipeline
from langchain.prompts.prompt import PromptTemplate
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory

# Load environment variables
dotenv.load_dotenv('/.env')
HF_ACCESS_TOKEN = os.getenv('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')

model_id = 'meta-llama/Llama-2-7b-chat-hf'

# Configure for 4-bit quantization (optimizes model deployment)
bnb_config = BitsAndBytesConfig(
    bnb_4bit_compute_dtype = 'float16',
    bnb_4bit_quant_type='nf4',
    load_in_4bit=True,
)

# Load model configuration
model_config = AutoConfig.from_pretrained(
    model_id,
    use_auth_token='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'
)

# Load model
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    config=model_config,
    device_map='auto',
    quantization_config=bnb_config,
    use_auth_token='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'
)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    model_id,
    use_auth_token='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'
)

# Set model into evaluation mode (optimizes inference)
model.eval()

pipe = pipeline(
    model=model,
    task='text-generation',
    tokenizer=tokenizer
)

llm = HuggingFacePipeline(pipeline=pipe)

template = """
<s>[INST] <<SYS>>
The following is a friendly conversation between a human and an AI. 
The AI is talkative and provides lots of specific details from its context. 
If the AI does not know the answer to a question, it truthfully says it does not know.
Please be concise.
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

def predict(message: str, history: str):
    response = conversation.predict(input=message)
    return response

# Create and launch the Gradio interface
interface = gr.ChatInterface(
    clear_btn=None,
    fn=predict,
    retry_btn=None,
    undo_btn=None,
)

interface.launch(
    height=600,
    inline=True,
    share=True,
    width=800
)