import os
import math
import dotenv
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import gradio as gr
from sklearn.metrics.pairwise import cosine_similarity
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

# tokenize output of model
def tokenize_output(output,tokenizer):
    return tokenizer.encode(output, return_tensors="pt")


#combine the persona and input tokens
def combine_inputs(persona_tokens,input_tokens ):
    return torch.cat((persona_tokens, input_tokens), dim=-1)



def calculate_log_likelihood(input_tokens, output_tokens, model, tokenizer, debug=False):
    if debug:
        print(f"Original input tokens shape: {input_tokens.shape}")
        print(f"Original output tokens shape: {output_tokens.shape}")

    max_length = max(input_tokens.size(1), output_tokens.size(1))

    input_tokens = F.pad(input_tokens, pad=(0, max_length - input_tokens.size(1)), value=tokenizer.pad_token_id)
    output_tokens = F.pad(output_tokens, pad=(0, max_length - output_tokens.size(1)), value=tokenizer.pad_token_id)

    if debug:
        print(f"Padded input tokens shape: {input_tokens.shape}")
        print(f"Padded output tokens shape: {output_tokens.shape}")

    with torch.no_grad():
        outputs = model(input_ids=input_tokens, labels=output_tokens)

    logits = outputs.logits
    log_probs = F.log_softmax(logits, dim=-1)

    actual_log_probs = log_probs.gather(-1, output_tokens.unsqueeze(-1)).squeeze(-1)
    mask = (output_tokens != tokenizer.pad_token_id)
    actual_log_probs = actual_log_probs * mask

    log_likelihood = actual_log_probs.sum().item()
    
    return log_likelihood

def calculate_similarity_score(persona_tokens, output_tokens, model, tokenizer, debug=False):
    if debug:
        print(f"Original persona tokens shape: {persona_tokens.shape}")
        print(f"Original output tokens shape: {output_tokens.shape}")
    
    # Getting embeddings from the model
    with torch.no_grad():
        persona_embeddings = model(input_ids=persona_tokens).last_hidden_state.mean(dim=1)
        output_embeddings = model(input_ids=output_tokens).last_hidden_state.mean(dim=1)
    
    if debug:
        print(f"Persona embeddings shape: {persona_embeddings.shape}")
        print(f"Output embeddings shape: {output_embeddings.shape}")
    
    # Calculating cosine similarity
    persona_embeddings = persona_embeddings.cpu().numpy()
    output_embeddings = output_embeddings.cpu().numpy()
    similarity_score = cosine_similarity(persona_embeddings, output_embeddings)
    
    if debug:
        print(f"Similarity score: {similarity_score}")
    
    return similarity_score



def calculate_perplexity(input_tensor, output_tensor, model):
    max_length = max(input_tensor.size(1), output_tensor.size(1))
    
    if input_tensor.size(1) < max_length:
        padding_size = max_length - input_tensor.size(1)
        input_tensor = F.pad(input_tensor, pad=(0, padding_size), value=tokenizer.pad_token_id)
    
    if output_tensor.size(1) < max_length:
        padding_size = max_length - output_tensor.size(1)
        output_tensor = F.pad(output_tensor, pad=(0, padding_size), value=tokenizer.pad_token_id)

    with torch.no_grad():
        loss = model(input_ids=input_tensor, labels=output_tensor).loss
    
    return math.exp(loss)



def predict(message: str):
    # Get model prediction
    output = conversation.predict(input=message)
    
    # Tokenizing persona, user input, and model output
    print('persona', template)
    print('message', message)
    persona_tokens = tokenize_persona(template, tokenizer)
    # input_tokens = tokenize_user_input(message, tokenizer)
    output_tokens = tokenize_output(output, tokenizer)
    
    # Combining tokens
    # input_combined = combine_inputs(persona_tokens, input_tokens)

    print(f'input Persona Tokens', {persona_tokens.shape})
    print(f'output Tokens', {output_tokens.shape})
    
    # Calculating log likelihood
    log_likelihood = calculate_log_likelihood(persona_tokens, output_tokens, model, tokenizer)

    #Calculate similarity
    similarity = calculate_similarity_score(persona_tokens, output_tokens, model, tokenizer)
    
    # Calculating perplexity
    perplexity = calculate_perplexity(persona_tokens, output_tokens, model)
    
    # Printing log likelihood and perplexity
    print(f"Persona Alignment Log Likelihood: {log_likelihood}")
    print(f"Persona Alignment Similarity: {similarity}")
    
    return output, log_likelihood, similarity


# Chat Interface
interface = gr.Interface(
    fn=predict,
    inputs=["text"],
    outputs=[
        gr.outputs.HTML(label="Output"),
        gr.outputs.Textbox(label="Persona Alignment Log Likelihood"),
        gr.outputs.Textbox(label="Persona Alignment Similarity")
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