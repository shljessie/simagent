import os
import dotenv
import torch
from torch.nn import CrossEntropyLoss
from transformers import (
    AutoConfig, 
    AutoModelForCausalLM, 
    AutoTokenizer, 
    BitsAndBytesConfig, 
)

# Model Configurations
dotenv.load_dotenv('../.env')
HF_ACCESS_TOKEN = os.getenv('HF_ACCESS_TOKEN')
model_id = '../Llama-2-7b-chat-hf'

# Configuration settings
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype='float16',
)

# Load model and tokenizer
model_config = AutoConfig.from_pretrained(model_id, use_auth_token=HF_ACCESS_TOKEN)
model = AutoModelForCausalLM.from_pretrained(model_id, config=model_config, use_auth_token=HF_ACCESS_TOKEN) # remove quantization
tokenizer = AutoTokenizer.from_pretrained(model_id, use_auth_token=HF_ACCESS_TOKEN)
model.eval()

# Function to calculate loss
@torch.no_grad()
def calculate_loss(model: model, tokenizer:tokenizer, convo_history, bot1_diag_response, ground_truth_answers):
    """Calculate the cross entropy loss of the diagnostic responses and ground_truth answers.
    This loss is calculated for each diagnostic question.

    Parameters: 
    model -- model to use. This should be identical to the model used in the chat.
    tokenizer -- This should be identical to the model used in the chat.
    conv_history -- The conversation history of bot1 and bot2. The last response is the last utterance of bot1
    bot1_diag_response -- Bot1 response to diagnostic question 
    ground_truth_answers -- Ground truth answers to diagnostic question
    
    """

    # check model inputs
    print("Calculate Loss \n")
    print("Conversation History: \n", convo_history )
    print("Bot1 Diagnostic Response: \n", bot1_diag_response)
    print("Ground Truth Answers: \n", ground_truth_answers, "\n")

    # tokenize inputs
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inputs = tokenizer(convo_history, return_tensors='pt')["input_ids"].to(device)
    bot1_diag_response = tokenizer(bot1_diag_response, return_tensors='pt')["input_ids"].to(device)
    ground_truth_answers = tokenizer(ground_truth_answers, return_tensors='pt')["input_ids"].to(device)
    diag_question_response = torch.concat([inputs, bot1_diag_response], dim=-1).to(device) # add tensors together

    # check tokenized inputs
    check = tokenizer.decode(diag_question_response[0])
    print("Check decoded tokenizer: \n", check, "\n")

    # pass through model, get hidden state
    outputs = model(diag_question_response, output_hidden_states=True) 
    # get hidden state of response to diagnostic output
    hiddens_diag_response = outputs.hidden_states[-1][:, -1+(-1*bot1_diag_response.shape[-1]):-1]

    # calculate loss
    logits  = model.lm_head(hiddens_diag_response) #compare model output against actual tokens
    loss_fct = CrossEntropyLoss(reduction="mean")
    loss = loss_fct(logits.squeeze(), ground_truth_answers.squeeze()) # get the logits probabilities bot1_diag_response and ground_truth answers

    # Q: should we append the ground truth answers too?
    return loss.item()

# for testing
history = "Prompt: Your name is Jack and you are from California. You are an introvert that likes to meditate. "
questions = "What is your name?"
answers = "Jack"
bot1_output = "Jack"


if __name__ == "__main__":
    history = "Prompt: Your name is Jack and you are from California. You are an introvert that likes to meditate. "
    questions = "What is your name?"
    answers = "Jack"
    bot1_output = "Jack"
    
    calculate_loss(model, tokenizer, history+questions, answers, bot1_output )