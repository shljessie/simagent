# diagnostic.py
import torch
from torch.nn import CrossEntropyLoss
from sentence_transformers import SentenceTransformer
import torch.nn.functional as F


modelsent = SentenceTransformer('paraphrase-MiniLM-L6-v2')
BOT_PERSONA = """
[SYSTEM]
You are Rohan a grad student at Stanford studying Material Science. You are 22 years old. Respond with one sentence only.
[/SYSTEM]
Respond with one sentence only.
"""

@torch.no_grad()
def calculate_loss(model: modelsent, tokenizer, convo_history, bot1_diag_response, ground_truth_answers):
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

    model = modelsent

    conversation = []
    full_system_prompt = BOT_PERSONA
    conversation.append({"role": "system", "content": full_system_prompt})

    for user, assistant in convo_history:
        conversation.extend([{"role": "user", "content": user}, {"role": "assistant", "content": assistant}])

    print('\nHF Conversation passed through chat_history in: ', conversation, "\n")

    # tokenize inputs
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inputs = tokenizer.apply_chat_template(conversation, return_tensors="pt").to(device)
    bot1_diag_response = tokenizer(bot1_diag_response, return_tensors='pt')["input_ids"].to(device)
    ground_truth_answers = tokenizer(ground_truth_answers, return_tensors='pt')["input_ids"].to(device)
    diag_question_response = torch.concat([inputs, bot1_diag_response], dim=-1).to(device) # add tensors together

    # calculate loss
    response_length = bot1_diag_response.shape[-1]
    ground_truth_length = ground_truth_answers.shape[1]

    if response_length > ground_truth_length:
        padding_size = response_length - ground_truth_length
        padded_ground_truth_answers = F.pad(ground_truth_answers, (0, padding_size), "constant", 0).to(device)
    elif response_length < ground_truth_length:
        padded_ground_truth_answers = ground_truth_answers[:, :response_length].to(device)
    else:
        padded_ground_truth_answers = ground_truth_answers

    loss_fct = CrossEntropyLoss(reduction="mean")
    loss = loss_fct(logits.view(-1, logits.size(-1)), padded_ground_truth_answers.view(-1))

    return loss.item(), conversation

"""
input must be a tensor of shape [N,C] where N is the batch size and C is the number of classes
target must be a tensor of shape [N]
"""