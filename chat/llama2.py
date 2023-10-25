import os
import math
import dotenv
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import gradio as gr
from transformers import (
    AutoConfig, 
    AutoModelForCausalLM, 
    AutoTokenizer, 
    BitsAndBytesConfig, 
    pipeline
)
from langchain.llms import HuggingFacePipeline
from langchain.prompts.prompt import PromptTemplate
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory


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

#load a hf text generation pipeline with the llama2 model
pipe = pipeline(
    model=model,
    task='text-generation',
    tokenizer=tokenizer
)
llm = HuggingFacePipeline(pipeline=pipe)


# model persona  template
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


# Initialize the conversation chain -langchain
conversation = ConversationChain(
    llm=llm,
    memory=ConversationBufferMemory(),
    prompt=prompt,
    verbose=False
)


# Function to tokenize the persona
def tokenize_persona(template, tokenizer):
    # Extracting persona from the template
    start_idx = template.find("The AI") + len("The AI")
    end_idx = template.find("<</SYS>>")
    persona_text = template[start_idx:end_idx].strip()
    return tokenizer.encode(persona_text, return_tensors="pt")


# tokenize user input
def tokenize_user_input(text,tokenizer):
    return tokenizer.encode(text, return_tensors="pt")


# generate output of model
def predict(message: str, history: str=""):
    return conversation.predict(input=message)


# tokenize output of model
def tokenize_output(output,tokenizer):
    return tokenizer.encode(output, return_tensors="pt")


#combine the persona and input tokens
def combine_inputs(persona_tokens,input_tokens ):
    return torch.cat((persona_tokens, input_tokens), dim=-1)


#calculate log likelihood
def calculate_log_likelihood(input_tokens, output_tokens, model):

    # Print original shapes for debugging
    print(f"Original input tokens shape: {input_tokens.shape}")
    print(f"Original output tokens shape: {output_tokens.shape}")

    # Determine the maximum sequence length between input and output tokens
    max_length = max(input_tokens.size(1), output_tokens.size(1))
    
    # Padding input_tokens
    if input_tokens.size(1) < max_length:
        padding_size = max_length - input_tokens.size(1)
        input_tokens = F.pad(input_tokens, pad=(0, padding_size), value=tokenizer.pad_token_id)
    
    # Padding output_tokens
    if output_tokens.size(1) < max_length:
        padding_size = max_length - output_tokens.size(1)
        output_tokens = F.pad(output_tokens, pad=(0, padding_size), value=tokenizer.pad_token_id)

    # Print padded shapes for debugging
    print(f"Padded input tokens shape: {input_tokens.shape}")
    print(f"Padded output tokens shape: {output_tokens.shape}")

    # Passing tokens through the model
    with torch.no_grad():  # Disable gradient calculations
        outputs = model(input_ids=input_tokens, labels=output_tokens)
    
    logits = outputs.logits  # Get the prediction logits
    log_probs = F.log_softmax(logits, dim=-1)  # Calculate log probabilities
    
    # Get log probabilities of actual tokens
    actual_log_probs = log_probs.gather(-1, output_tokens.unsqueeze(-1)).squeeze(-1)
    
    # Ignore padding tokens (if any)
    mask = (output_tokens != tokenizer.pad_token_id)
    actual_log_probs = actual_log_probs * mask
    
    # Sum the log probabilities
    log_likelihood = actual_log_probs.sum().item()
    
    return log_likelihood


def calculate_perplexity(input_tensor, output_tensor, model):
    with torch.no_grad():
        loss = model(input_ids=input_tensor, labels=output_tensor).loss
    
    return math.exp(loss)



def predict(message: str):
    # Get model prediction
    output = conversation.predict(input=message)
    
    # Tokenizing persona, user input, and model output
    persona_tokens = tokenize_persona(template, tokenizer)
    input_tokens = tokenize_user_input(message, tokenizer)
    output_tokens = tokenize_output(output, tokenizer)
    
    # Combining tokens
    input_combined = combine_inputs(persona_tokens, input_tokens)
    output_combined = combine_inputs(persona_tokens, output_tokens)
    
    # Calculating log likelihood
    log_likelihood = calculate_log_likelihood(input_combined, output_combined, model)
    
    # Calculating perplexity
    perplexity = calculate_perplexity(input_combined, output_combined, model)
    
    # Printing log likelihood and perplexity
    print(f"Persona Alignment Log Likelihood: {log_likelihood}")
    print(f"Persona Alignment Perplexity: {perplexity}")
    
    return output, log_likelihood, perplexity




# Chat Interface
interface = gr.Interface(
    fn=predict,
    inputs=["text"],
    outputs=[
        gr.outputs.Textbox(label="Output"),
        gr.outputs.Textbox(label="Persona Alignment Log Likelihood"),
        gr.outputs.Textbox(label="Persona Alignment Perplexity")
    ],
)

interface.launch(
    height=600,
    inline=True,
    share=True,
    width=800
)










# # get model output token predictions
# def get_next_token_predictions(text, model, tokenizer):
#     # tokenizing the user input of the model
#     tokens = tokenizer.encode(text, return_tensors="pt")
#     tokens = torch.cat((torch.tensor([tokenizer.eos_token_id]), tokens[0])).reshape(1,-1)

#     print('tokens',tokens)

#     for i in np.arange(0,len(tokens[0])-1):
#         outputs = model.generate(tokens[0][:i+1].reshape(1,-1), max_new_tokens=1, output_scores=True, return_dict_in_generate=True, pad_token_id=tokenizer.eos_token_id)
#         scores = F.softmax(outputs.scores[0], dim=-1)
#         top_10 = torch.topk(scores, 10)
#         df = pd.DataFrame()
#         a = scores[0][tokens[0][i+1]]
#         b = top_10.values
#         df["probs"] = list(np.concatenate([a.reshape(-1,1).numpy()[0], b[0].numpy()]))
#         diff = 100*(df["probs"].iloc[0]-df["probs"].iloc[1])
#         if np.abs(diff)<1:
#           color = "mystronggreen"
#         elif np.abs(diff)<10:
#           color = "mygreen"
#         elif np.abs(diff)<20:
#           color = "myorange"
#         elif np.abs(diff)<30:
#           color = "myyellow"
#         else:
#           color = "myred"
#         df["probs"] = [f"{value:.2%}" for value in df["probs"].values]
#         aux = [tokenizer.decode(tokens[0][i+1])] + [tokenizer.decode(top_10.indices[0][i]) for i in range(10)]
#         df["predicted next token"] = aux
    
#     print('probs:', df['probs'])
#     print('next token:', df['predicted next token'])
#     return df["probs"], df["predicted next token"]

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


# f1 score calculation when I have the true_texts
# def calculate_f1_score(true_texts, generated_texts, tokenizer):
#     true_tokens = tokenizer.batch_encode_plus(true_texts, add_special_tokens=False)['input_ids']
#     gen_tokens = tokenizer.batch_encode_plus(generated_texts, add_special_tokens=False)['input_ids']
    
#     # Flattening the lists of tokens
#     true_tokens_flat = [tok for sublist in true_tokens for tok in sublist]
#     gen_tokens_flat = [tok for sublist in gen_tokens for tok in sublist]
    
#     # Calculating F1 Score
#     return f1_score(true_tokens_flat, gen_tokens_flat, average='weighted')


# interface = gr.Interface(
#     fn=predict,
#     inputs=["text", "text"],
#     outputs=gr.outputs.HTML(),
# )

# interface.launch(
#     height=600,
#     inline=True,
#     share=True,
#     width=800
# )