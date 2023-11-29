import torch
import evaluate
from evaluate import load

BOT_PERSONA = """
[SYSTEM]
You are Rohan a grad student at Stanford studying Material Science. You are 22 years old. Respond with one sentence only.
[/SYSTEM]
Respond with one sentence only.
"""


@torch.no_grad()
def calculate_loss(model, tokenizer, convo_history, bot1_diag_response, ground_truth_answers):
    """Calculate BERT score between bot1's response and ground truth answers.

    Parameters: 
    model -- model to use. This should be identical to the model used in the chat.
    tokenizer -- This should be identical to the model used in the chat.
    convo_history -- The conversation history of bot1 and bot2.
    bot1_diag_response -- Bot1 response to diagnostic question 
    ground_truth_answers -- Ground truth answers to diagnostic question
    """
    bertscore = load("bertscore")
    conversation = []
    full_system_prompt = BOT_PERSONA
    conversation.append({"role": "system", "content": full_system_prompt})

    for user, assistant in convo_history:
        conversation.extend([{"role": "user", "content": user}, {"role": "assistant", "content": assistant}])

    bertscore = bertscore.compute(predictions=bot1_diag_response, references=ground_truth_answers)

    return bertscore, conversation

