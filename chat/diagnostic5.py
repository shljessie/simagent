import torch
import nltk
from nltk.translate.bleu_score import sentence_bleu

nltk.download('punkt')

BOT_PERSONA = """
[SYSTEM]
You are Rohan a grad student at Stanford studying Material Science. You are 22 years old. Respond with one sentence only.
[/SYSTEM]
Respond with one sentence only.
"""


@torch.no_grad()
def calculate_loss(model, tokenizer, convo_history, bot1_diag_response, ground_truth_answers):
    """Calculate BERTScore and BLEU score between bot1's response and ground truth answers.

    Parameters: 
    model -- model to use. This should be identical to the model used in the chat.
    tokenizer -- This should be identical to the model used in the chat.
    convo_history -- The conversation history of bot1 and bot2.
    bot1_diag_response -- Bot1 response to diagnostic question 
    ground_truth_answers -- Ground truth answers to diagnostic question
    """

    conversation = []
    full_system_prompt = BOT_PERSONA
    conversation.append({"role": "system", "content": full_system_prompt})

    for user, assistant in convo_history:
        conversation.extend([{"role": "user", "content": user}, {"role": "assistant", "content": assistant}])

    # Tokenize sentences for BLEU calculation
    reference = [tokenizer.tokenize(ground_truth_answer) for ground_truth_answer in ground_truth_answers]
    candidate = tokenizer.tokenize(bot1_diag_response)

    # Calculate BLEU score
    bleu_score = sentence_bleu(reference, candidate)

    return bleu_score , conversation

