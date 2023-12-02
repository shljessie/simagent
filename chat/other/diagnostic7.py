import os
import dotenv
import torch
from torch.nn import CrossEntropyLoss
from sentence_transformers import SentenceTransformer, util
import numpy as np

BOT_PERSONA = """
[SYSTEM]
You are Rohan a grad student at Stanford studying Material Science. You are 22 years old. Respond with one sentence only.
[/SYSTEM]
Respond with one sentence only.
"""
# Function to calculate loss
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
    conversation = []
    full_system_prompt = BOT_PERSONA
    conversation.append({"role": "system", "content": full_system_prompt})

    for user, assistant in convo_history:
        conversation.extend([{"role": "user", "content": user}, {"role": "assistant", "content": assistant}])

    # check model inputs
    print("Calculate Loss \n")
    print("Conversation History: \n", convo_history )
    print("Bot1 Diagnostic Response: \n", bot1_diag_response)
    print("Ground Truth Answers: \n", ground_truth_answers, "\n")

    # sentence bert embeddings
    sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
    inputs = sbert_model.encode(convo_history)
    bot1_diag_response = sbert_model.encode(bot1_diag_response)
    ground_truth_answers = sbert_model.encode(ground_truth_answers)
    print('inputs', inputs.shape)
    print('bot1diag', bot1_diag_response.shape)


    diag_question_response = np.vstack((inputs, bot1_diag_response))
    diag_question_response = torch.from_numpy(diag_question_response)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    diag_question_response = diag_question_response.to(device)

    # pass through model, get hidden state
    outputs = model(diag_question_response, output_hidden_states=True) 
    hiddens_diag_response = outputs.hidden_states[-1][:, -1+(-1*bot1_diag_response.shape[-1]):-1]

    # calculate loss
    logits  = model.lm_head(hiddens_diag_response) #compare model output against actual tokens
    loss_fct = CrossEntropyLoss(reduction="mean")
    loss = loss_fct(logits.squeeze(), ground_truth_answers.squeeze()) # get the logits probabilities bot1_diag_response and ground_truth answers

    # Q: should we append the ground truth answers too?
    return loss.item() , conversation

# for testing
if __name__ == "__main__":
    history = "Prompt: Your name is Jack and you are from California. You are an introvert that likes to meditate. "
    questions = "What is your name?"
    answers = "Jack"
    bot1_output = "Jack"

    calculate_loss(model, tokenizer, history+questions, answers, bot1_output )