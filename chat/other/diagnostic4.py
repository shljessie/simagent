# diagnostic.py
import torch
from torch.nn import CrossEntropyLoss
from sentence_transformers import SentenceTransformer, util


BOT_PERSONA = """
[SYSTEM]
You are Rohan a grad student at Stanford studying Material Science. You are 22 years old. Respond with one sentence only.
[/SYSTEM]
Respond with one sentence only.
"""

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
    print("------------------------ Calculating Loss ----------------------")
    print("Conversation History: \n", convo_history, "\n" )
    print("Bot1 Diagnostic Response: \n", bot1_diag_response)
    print("Ground Truth Answers: \n", ground_truth_answers, "\n")

    conversation = []
    full_system_prompt = BOT_PERSONA
    conversation.append({"role": "system", "content": full_system_prompt})

    for user, assistant in convo_history:
        conversation.extend([{"role": "user", "content": user}, {"role": "assistant", "content": assistant}])

    # check model inputs
    print("------------------------ Calculating Loss ----------------------")
    # Load SBERT model
    sbert_model = SentenceTransformer('all-MiniLM-L6-v2')

    bot1_response = sbert_model.encode(bot1_diag_response)
    truth_response = sbert_model.encode(ground_truth_answers)
    cosine_scores = util.pytorch_cos_sim(bot1_response, truth_response)

    # Q: should we append the ground truth answers too?
    return cosine_scores.item(), conversation

"""
input must be a tensor of shape [N,C] where N is the batch size and C is the number of classes
target must be a tensor of shape [N]
"""