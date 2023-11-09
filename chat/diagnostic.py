import os
import dotenv
import torch
from torch.nn import CrossEntropyLoss
from transformers import (
    AutoConfig, 
    AutoModelForCausalLM, 
    AutoTokenizer, 
    BitsAndBytesConfig, 
    pipeline
)
import math
#load model
# model = AutoModelForCausalLM.from_pretrained('gpt2')
# tokenizer = AutoTokenizer.from_pretrained('gpt2')

# Model Configurations
dotenv.load_dotenv('/.env')
HF_ACCESS_TOKEN = os.getenv('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')
model_id = '../Llama-2-7b-chat-hf'
# model_id ='gpt2'

# Configuration settings
# bnb_config = BitsAndBytesConfig(
#     bnb_4bit_compute_dtype='float16',
#     bnb_4bit_quant_type='nf4',
#     load_in_4bit=True,
# )

# Load model and tokenizer
model_config = AutoConfig.from_pretrained(model_id, use_auth_token=HF_ACCESS_TOKEN)
model = AutoModelForCausalLM.from_pretrained(model_id, config=model_config, quantization_config=bnb_config, use_auth_token=HF_ACCESS_TOKEN)
tokenizer = AutoTokenizer.from_pretrained(model_id, use_auth_token=HF_ACCESS_TOKEN)

# tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-1b")
# model = AutoModelForCausalLM.from_pretrained("EleutherAI/pythia-1b")
model.eval()

# Function to calculate loss
@torch.no_grad()
def calculate_loss(model: model, tokenizer:tokenizer, convo_history, bot1_diag_response, ground_truth_answers):
    """
    model: model to use. identical to the chat model used.
    tokenizer: identical model to the chat model
    conv_history: conversation history of bot1 and bot2 
    bot1_diag_response: bot1 response to diagnostic question 
    ground_truth_answers: ground truth answers to diagnostic question
    
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inputs = tokenizer(convo_history, return_tensors='pt')["input_ids"].to(device)
    bot1_diag_response = tokenizer(bot1_diag_response, return_tensors='pt')["input_ids"].to(device)
    ground_truth_answers = tokenizer(ground_truth_answers, return_tensors='pt')["input_ids"].to(device)
    diag_question_response = torch.concat([inputs, bot1_diag_response], dim=-1).to(device) # add tensors together,

    check = tokenizer.decode(diag_question_response[0])
    print( "\n" + ' CHECKING ',check + "\n" )

    outputs = model(diag_question_response, output_hidden_states=True) # pass to model , get hiddenstate
    hiddens_diag_response = outputs.hidden_states[-1][:, -1+(-1*bot1_diag_response.shape[-1]):-1] # get hidden state of response to diagnostic output

    logits  = model.lm_head(hiddens_diag_response)
    loss_fct = CrossEntropyLoss(reduction="mean")
    loss = loss_fct(logits.squeeze(), ground_truth_answers.squeeze()) # get the logits probabilities

    return loss.item()

# def calculate_loss(model, tokenizer, text, answers ,):
#     # text : convo history + last bot answer
#     # answers : ground truth answers
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     inputs = tokenizer(text, return_tensors='pt')["input_ids"].to(device)
#     answers = tokenizer(answers, return_tensors='pt')["input_ids"].to(device)
#     inputs_and_answers = torch.concat([inputs, answers], dim=-1).to(device) # add tensors together,

#     outputs = model(inputs_and_answers, output_hidden_states=True) # pass to model , get hiddenstate
#     hiddens_answer = outputs.hidden_states[-1][:, -1+(-1*answers.shape[-1]):-1] # get hidden state of last answer
#     logits  = model.lm_head(hiddens_answer)
#     loss_fct = CrossEntropyLoss(reduction="mean")
#     loss = loss_fct(logits.squeeze(), answers.squeeze()) # get the logits probabilities

#     return loss.item()


#parameters : just for testing !
history = "Prompt: Your name is Jack and you are from California. You are an introvert that likes to meditate. "
questions = "What is your name?"
answers = "Jack"
bot1_output = "Jack"


if __name__ == "__main__":
  calculate_loss(model, tokenizer, history+questions, answers, bot1_output )







# bert_tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
# bert_model = AutoModelForCausalLM.from_pretrained('bert-base-uncased')





    # logits = bert_model(inputs_and_answers).logits


    # Get the model's output (last hidden states)
    # print(logits)

    # probs = torch.softmax(logits[:, :, [tokenizer("part", return_tensors='pt')["input_ids"][0], tokenizer("sleep", return_tensors='pt')["input_ids"][0]]], dim=-1)


    # top_predictions = logits.topk(5, dim=-1)  # Get the top 5 predictions
    # # Flatten the list of top prediction indices
    # top_pred_indices = top_predictions.indices.squeeze().tolist()

    # # Now, decode each token ID into text
    # for idx in top_pred_indices:
    #     token = tokenizer.decode([idx])  # decode expects a list of integers
    #     print(f"Token ID {idx} is token '{token}'")

    # print("loss: ",loss)
    # print("check : ",tokenizer.convert_ids_to_tokens(198))
    # print(logits.argmax(-1).flatten())
    # print('model prediction:',tokenizer.decode(logits.argmax(-1).flatten()))


'''
Diagnostic questions that evaluate model consistency at each model response turn
'''
# Diagnostic function
# def diagnostic(history, model, questions, answers):
#    #get response from questions 

#    # nll : 

#     losses = []
#     for question, answer in zip(questions, answers):
#         # Combine history and question for context
#         input_text = history + " " + question
#         # Calculate loss
#         loss = calculate_loss(model, gpt2_tokenizer, input_text, answer)
#         losses.append(loss)
    
#     return losses


# result = diagnostic(history, model, questions, answers)
# print(result)


  


