from transformers import AutoTokenizer, pipeline, logging
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
import argparse
from langchain import LLMChain
from langchain.prompts.prompt import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.llms import HuggingFacePipeline
import transformers
from torch import cuda, bfloat16

model_name_or_path = "../Llama-2-7b-chat-hf"
model_basename = "Llama-2-7b-chat-hf"
use_triton = False

device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'


tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)

quant_config = transformers.BitsAndBytesConfig(

    load_in_4bit=True,

    bnb_4bit_quant_type='nf4',

    bnb_4bit_use_double_quant=True,

    bnb_4bit_compute_dtype=bfloat16

)

model_config = transformers.AutoConfig.from_pretrained(

    model_basename,

    use_auth_token='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'

)

model = transformers.AutoModelForCausalLM.from_pretrained(

    model_basename,

    trust_remote_code=True,

    config=model_config,

    quantization_config=quant_config,

    use_auth_token='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'

)

model.eval()

print(f"Model loaded on {device}")

tokenizer = transformers.AutoTokenizer.from_pretrained(

    model_basename,

    use_auth_token='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'

)

model = transformers.AutoModelForCausalLM.from_pretrained(

    model_basename,

    trust_remote_code=True,

    config=model_config,

    quantization_config=quant_config,

    use_auth_token='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'

)


# model = AutoGPTQForCausalLM.from_quantized(model_name_or_path,
#         model_basename=model_basename,
#         use_safetensors=True,
#         trust_remote_code=True,
#         use_triton=use_triton,
#         quantize_config=None)


logging.set_verbosity(logging.CRITICAL)

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=512,
    temperature=0.1,
    top_p=0.95,
    repetition_penalty=1.15
)

template = """
[INST] <<SYS>>
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.
<</SYS>>
here is the chat history
{chat_history}

{prompt} [/INST]
"""

prompt = PromptTemplate(
    input_variables=["chat_history", "prompt"],
    template=template
)

llm=HuggingFacePipeline(pipeline=pipe)
memory = ConversationBufferMemory(memory_key="chat_history",    k=3,
    return_messages=True)


llm_chain = LLMChain(llm=llm, prompt=prompt, verbose=False,  memory=memory)

while 1:
  text=input("You: ")
  if text=='end':
    break
  output=llm_chain.predict(prompt=text)
  print("Chatbot: ",output)