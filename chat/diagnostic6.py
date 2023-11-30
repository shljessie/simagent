# diagnostic.py
import torch
from torch.nn import CrossEntropyLoss
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

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

    # check model inputs
    print("------------------------ Calculating Loss ----------------------")
    print("Conversation History: \n", convo_history, "\n" )
    print("Bot1 Diagnostic Response: \n", bot1_diag_response)
    print("Ground Truth Answers: \n", ground_truth_answers, "\n")

    conversation = []
    full_system_prompt = BOT_PERSONA
    conversation.append({"role": "system", "content": full_system_prompt})

    # for user, assistant in convo_history:
    #     conversation.extend([{"role": "user", "content": user}, {"role": "assistant", "content": assistant}])

    print('\nHF Conversation passed through chat_history in: ', conversation, "\n")

    # tokenize inputs
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inputs = tokenizer.apply_chat_template(conversation, return_tensors="pt").to(device)
    bot1_diag_response = tokenizer(bot1_diag_response, return_tensors='pt')["input_ids"].to(device)
    ground_truth_answers = tokenizer(ground_truth_answers, return_tensors='pt')["input_ids"].to(device)
    diag_question_response = torch.concat([inputs, bot1_diag_response], dim=-1).to(device) # add tensors together


    # Check token lengths
    print("Input tokens length:", inputs.size(), "\n")
    print("Bot1 response tokens length:", bot1_diag_response.size(), "\n")
    print("Ground truth tokens length:", ground_truth_answers.size(), "\n")
    print("Concat size: ", diag_question_response.size(), "\n")
    
    # check tokenized inputs
    check = tokenizer.decode(diag_question_response[0])

    # pass through model, get hidden state
    outputs = model(diag_question_response, output_hidden_states=True) 
    # get hidden state of response to diagnostic output
    hiddens_diag_response = outputs.hidden_states[-1][:, -1+(-1*bot1_diag_response.shape[-1]):-1]
    print("hiddens_diag_response size: ", hiddens_diag_response.size(), "\n")

    # calculate loss
    logits  = model.lm_head(hiddens_diag_response) #compare model output against actual tokens
    padded_ground_truth_answers = torch.nn.functional.pad(
        ground_truth_answers, (0, bot1_diag_response.size(1) - ground_truth_answers.size(1)), 'constant', 0
    )
    logits = logits[:, :padded_ground_truth_answers.size(1), :].contiguous()

    print('size of logits: ', logits)
    print('size of padded ground truth', padded_ground_truth_answers)

    loss_fct = CrossEntropyLoss(reduction="mean")
    print('final logits:', logits.view(-1, logits.size(-1)))
    print('final logits: ',logits.view(-1, logits.size(-1)).shape )
    print('final ground truth,', padded_ground_truth_answers.view(-1) )
    print('final ground truth,', padded_ground_truth_answers.view(-1).shape )
    loss = loss_fct(logits.view(-1, logits.size(-1)), padded_ground_truth_answers.view(-1))

    return loss.item(), conversation

"""
input must be a tensor of shape [N,C] where N is the batch size and C is the number of classes
target must be a tensor of shape [N]
"""