import torch
from torch.nn import CrossEntropyLoss

BOT_PERSONA = """
[SYSTEM]
You are Rohan, a grad student at Stanford studying Material Science. I like cocoa almond spread.
[/SYSTEM]
Limit your response to one sentence.
Respond in this format: 

What is your xx?
My xx is 
"""

@torch.no_grad()
def calculate_loss(model, tokenizer, convo_history, bot1_diag_response, ground_truth_answers):
    print("------------------------ Calculating Loss ----------------------")
    print("Conversation History: \n", convo_history, "\n")
    print("Bot1 Diagnostic Response: \n", bot1_diag_response)
    print("Ground Truth Answers: \n", ground_truth_answers, "\n")

    # Prepare conversation history
    conversation = [{"role": "system", "content": BOT_PERSONA}]
    for user, assistant in convo_history:
        conversation.extend([{"role": "user", "content": user}, {"role": "assistant", "content": assistant}])

    print('\nHF Conversation passed through chat_history in: ', conversation, "\n")

    # Tokenize inputs
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inputs = tokenizer.apply_chat_template(conversation, return_tensors="pt").to(device)
    bot1_diag_response = tokenizer(bot1_diag_response, return_tensors='pt')["input_ids"].to(device)
    ground_truth_answers = tokenizer(ground_truth_answers, return_tensors='pt')["input_ids"].to(device)
    diag_question_response = torch.concat([inputs, bot1_diag_response], dim=-1).to(device)

    print("Input tokens length:", inputs.size(), "\n")
    print("Bot1 response tokens length:", bot1_diag_response.size(), "\n")
    print("Ground truth tokens length:", ground_truth_answers.size(), "\n")
    print("Concat size: ", diag_question_response.size(), "\n")

    check = tokenizer.decode(diag_question_response[0])
    print("Check decoded tokenizer: \n", check, "\n")

    # Model output
    outputs = model(diag_question_response, output_hidden_states=True)
    hiddens_diag_response = outputs.hidden_states[-1][:, -1+(-1*bot1_diag_response.shape[-1]):-1]
    print("hiddens_diag_response size: ", hiddens_diag_response.size(), "\n")

    # Loss calculation with mask
    logits = model.lm_head(hiddens_diag_response)
    loss_fct = CrossEntropyLoss(reduction="none")
    padded_ground_truth_answers = torch.nn.functional.pad(
        ground_truth_answers, (0, bot1_diag_response.size(1) - ground_truth_answers.size(1)), 'constant', 0
    )

    # Create a mask for non-padding tokens
    mask = ground_truth_answers != tokenizer.pad_token_id
    loss = loss_fct(logits.view(-1, logits.size(-1)), padded_ground_truth_answers.view(-1))
    masked_loss = torch.sum(loss * mask.view(-1)) / torch.sum(mask)

    return masked_loss.item(), conversation

# Note: Ensure that the tokenizer and model are properly initialized before calling this function.
