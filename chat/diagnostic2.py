# diagnostic.py
import torch
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F

BOT_PERSONA = """
[SYSTEM]
You are Rohan a grad student at Stanford studying Material Science. You are 22 years old.
[SYSTEM]
Respond with one sentence only.
"""
@torch.no_grad()
def calculate_loss(model, tokenizer, convo_history, bot1_diag_response, ground_truth_answers, diagnostic_question):
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
    print("Conversation History: \n", convo_history, "\n" )
    print("Bot1 Diagnostic Response: \n", bot1_diag_response)
    print("Ground Truth Answers: \n", ground_truth_answers, "\n")

    conversation = []
    full_system_prompt = BOT_PERSONA
    conversation.append({"role": "system", "content": full_system_prompt})

    for user, assistant in convo_history:
        conversation.extend([{"role": "user", "content": user}, {"role": "assistant", "content": assistant}])

    conversation.append({"role":"user", "content":diagnostic_question})
    print('\nHF Conversation passed through chat_history in: ', conversation, "\n")

    # tokenize inputs
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    conversation_token = tokenizer.apply_chat_template(conversation, return_tensors="pt").to(device)
    ground_truth_answers = tokenizer(ground_truth_answers, return_tensors='pt')["input_ids"].to(device)
    concat_history_truth = torch.concat([conversation_token, ground_truth_answers], dim=-1).to(device)  

    # pass through model, get hidden state
    outputs = model(concat_history_truth, output_hidden_states=True) 
    #check hidden state shape 
    hiddens_diag_response = outputs.hidden_states[-1][:, -1+(-1*ground_truth_answers.shape[-1]):-1]

    logits  = model.lm_head(hiddens_diag_response)

    # calculate loss
    loss_fct = CrossEntropyLoss(reduction="mean")
    print('Logits Shape: ', logits.view(-1, logits.size(-1)).shape)
    print('Ground Truth Answers Shape: ', ground_truth_answers.view(-1).shape)
    loss = loss_fct(logits.view(-1, logits.size(-1)),ground_truth_answers.view(-1)) # (n,c) n shape required

    print('Loss Calculation', loss)

    return loss.item(), conversation