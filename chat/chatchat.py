import os
from threading import Thread
from typing import Iterator
import dotenv
import gradio as gr
import spaces
import torch
from transformers import (
    AutoConfig, 
    AutoModelForCausalLM, 
    AutoTokenizer, 
    BitsAndBytesConfig, 
    TextIteratorStreamer
)

#for default
DESCRIPTION = """\
# Llama-2 7B Chat
This Space demonstrates model [Llama-2-7b-chat](https://huggingface.co/meta-llama/Llama-2-7b-chat) by Meta, a Llama 2 model with 7B parameters fine-tuned for chat instructions. Feel free to play with it, or duplicate to run generations without a queue! If you want to run your own service, you can also [deploy the model on Inference Endpoints](https://huggingface.co/inference-endpoints).
🔎 For more details about the Llama 2 family of models and how to use them with `transformers`, take a look [at our blog post](https://huggingface.co/blog/llama2).
🔨 Looking for an even more powerful model? Check out the [13B version](https://huggingface.co/spaces/huggingface-projects/llama-2-13b-chat) or the large [70B model demo](https://huggingface.co/spaces/ysharma/Explore_llamav2_with_TGI).
"""

LICENSE = """
<p/>
---
As a derivate work of [Llama-2-7b-chat](https://huggingface.co/meta-llama/Llama-2-7b-chat) by Meta,
this demo is governed by the original [license](https://huggingface.co/spaces/huggingface-projects/llama-2-7b-chat/blob/main/LICENSE.txt) and [acceptable use policy](https://huggingface.co/spaces/huggingface-projects/llama-2-7b-chat/blob/main/USE_POLICY.md).
"""

MAX_MAX_NEW_TOKENS = 2048
DEFAULT_MAX_NEW_TOKENS = 1024
MAX_INPUT_TOKEN_LENGTH = int(os.getenv("MAX_INPUT_TOKEN_LENGTH", "4096"))

# Model Configurations
dotenv.load_dotenv('../.env')
HF_ACCESS_TOKEN = os.getenv('HF_ACCESS_TOKEN')
model_id = '../Llama-2-7b-chat-hf'

# Configuration settings
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype='float16',
)

#load model
model_config = AutoConfig.from_pretrained(model_id, use_auth_token=HF_ACCESS_TOKEN)
model = AutoModelForCausalLM.from_pretrained(model_id, config=model_config, use_auth_token=HF_ACCESS_TOKEN) # remove quantization
tokenizer = AutoTokenizer.from_pretrained(model_id, use_auth_token=HF_ACCESS_TOKEN)


print(model)
print(tokenizer)

@spaces.GPU
def generate(
    message: str,
    chat_history, 
    system_prompt: str,
    max_new_tokens: int = 1024,
    temperature: float = 0.6,
    top_p: float = 0.9,
    top_k: int = 50,
    repetition_penalty: float = 1.2,
) -> Iterator[str]:
    conversation = []
    if system_prompt:
        conversation.append({"role": "system", "content": system_prompt})
    for user, assistant in chat_history:
        conversation.extend([{"role": "user", "content": user}, {"role": "assistant", "content": assistant}])
    conversation.append({"role": "user", "content": message})

    print('CONVP',conversation)

    input_ids = tokenizer.apply_chat_template(conversation, return_tensors="pt")
    if input_ids.shape[1] > MAX_INPUT_TOKEN_LENGTH:
        input_ids = input_ids[:, -MAX_INPUT_TOKEN_LENGTH:]
        gr.Warning(f"Trimmed input from conversation as it was longer than {MAX_INPUT_TOKEN_LENGTH} tokens.")
    input_ids = input_ids.to(model.device)

    streamer = TextIteratorStreamer(tokenizer, timeout=10.0, skip_prompt=True, skip_special_tokens=True)
    generate_kwargs = dict(
        {"input_ids": input_ids},
        streamer=streamer,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        top_p=top_p,
        top_k=top_k,
        temperature=temperature,
        num_beams=1,
        repetition_penalty=repetition_penalty,
    )
    t = Thread(target=model.generate, kwargs=generate_kwargs)
    t.start()

    outputs = []
    for text in streamer:
        outputs.append(text)
        yield "".join(outputs)


chat_interface = gr.ChatInterface(
    fn=generate,
    additional_inputs=[
        gr.Textbox(label="System prompt", lines=6),
        gr.Slider(
            label="Max new tokens",
            minimum=1,
            maximum=MAX_MAX_NEW_TOKENS,
            step=1,
            value=DEFAULT_MAX_NEW_TOKENS,
        ),
        gr.Slider(
            label="Temperature",
            minimum=0.1,
            maximum=4.0,
            step=0.1,
            value=0.6,
        ),
        gr.Slider(
            label="Top-p (nucleus sampling)",
            minimum=0.05,
            maximum=1.0,
            step=0.05,
            value=0.9,
        ),
        gr.Slider(
            label="Top-k",
            minimum=1,
            maximum=1000,
            step=1,
            value=50,
        ),
        gr.Slider(
            label="Repetition penalty",
            minimum=1.0,
            maximum=2.0,
            step=0.05,
            value=1.2,
        ),
    ],
    stop_btn=None,
    examples=[
        ["Hello there! How are you doing?"],
        ["Can you explain briefly to me what is the Python programming language?"],
        ["Explain the plot of Cinderella in a sentence."],
        ["How many hours does it take a man to eat a Helicopter?"],
        ["Write a 100-word article on 'Benefits of Open-Source in AI research'"],
    ],
)

with gr.Blocks(css="style.css") as demo:
    gr.Markdown(DESCRIPTION)
    gr.DuplicateButton(value="Duplicate Space for private use", elem_id="duplicate-button")
    chat_interface.render()
    gr.Markdown(LICENSE)

if __name__ == "__main__":
    demo.queue(max_size=20).launch()
