# diagnostic.py
import torch
from torch.nn import CrossEntropyLoss

@torch.no_grad()
def calculate_loss(model, tokenizer, convo_history, bot1_diag_response, ground_truth_answers):
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
    print("------------------------ Calculating Loss ----------------------")
    print("Bot1 Diagnostic Response: \n", bot1_diag_response)
    print("Ground Truth Answers: \n", ground_truth_answers, "\n")

    # tokenize inputs
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bot1_diag_response = tokenizer(bot1_diag_response, return_tensors='pt')["input_ids"].to(device)
    ground_truth_answers = tokenizer(ground_truth_answers, return_tensors='pt')["input_ids"].to(device)

    # Check token lengths
    print("Bot1 response tokens length:", bot1_diag_response.size(), "\n")
    print("Ground truth tokens length:", ground_truth_answers.size(), "\n")
    
    # check tokenized inputs
    check = tokenizer.decode(bot1_diag_response[0])
    print("Check decoded tokenizer: \n", check, "\n")

    # pass through model, get hidden state
    outputs = model(bot1_diag_response, output_hidden_states=True) 
    # get hidden state of response to diagnostic output
    hiddens_diag_response = outputs.hidden_states[-1][:, -1+(-1*bot1_diag_response.shape[-1]):-1]
    print("hiddens_diag_response zie: ", hiddens_diag_response.size(), "\n")

    # calculate loss
    logits  = model.lm_head(hiddens_diag_response) #compare model output against actual tokens
    logits = logits[:, -ground_truth_answers.size(1):, :].contiguous()
    loss_fct = CrossEntropyLoss(reduction="mean")
    loss = loss_fct(logits.view(-1, logits.size(-1)), ground_truth_answers.view(-1)) # get the logits probabilities bot1_diag_response and ground_truth answers

    # Q: should we append the ground truth answers too?
    return loss.item()
