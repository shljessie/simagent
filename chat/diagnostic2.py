import torch
from torch.nn import CrossEntropyLoss

BOT_PERSONA = """
[SYSTEM]
You are Rohan a grad student at Stanford studying Material Science. I like cocoalmond spread.
[/SYSTEM]
Limit your response to one sentence.
"""

@torch.no_grad()
def calculate_loss(model, tokenizer, convo_history, bot1_diag_response, ground_truth_answers, debug=False):
    """Calculate the cross entropy loss of the diagnostic responses and ground_truth answers."""

    if debug:
        print("------------------------ Calculating Loss ----------------------")
        print("Conversation History: \n", convo_history, "\n")
        print("Bot1 Diagnostic Response: \n", bot1_diag_response)
        print("Ground Truth Answers: \n", ground_truth_answers, "\n")

    conversation = []
    full_system_prompt = BOT_PERSONA
    conversation.append({"role": "system", "content": full_system_prompt})

    for user, assistant in convo_history:
        conversation.extend([{"role": "user", "content": user}, {"role": "assistant", "content": assistant}])

    if debug:
        print('\nConversation passed to tokenizer: ', conversation, "\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # tokenize and concatenate inputs
    inputs = tokenizer.apply_chat_template(conversation, return_tensors="pt").to(device)
    response_tokens = tokenizer(bot1_diag_response, return_tensors='pt')["input_ids"].to(device)
    total_length = inputs.input_ids.size(1) + response_tokens.size(1)
    max_length = model.config.max_position_embeddings

    if total_length > max_length:
        raise ValueError(f"Total token length ({total_length}) exceeds the model's maximum length ({max_length}).")

    diag_question_response = torch.concat([inputs.input_ids, response_tokens], dim=-1).to(device)

    if debug:
        print("Input tokens length:", inputs.input_ids.size(), "\n")
        print("Bot1 response tokens length:", response_tokens.size(), "\n")
        check = tokenizer.decode(diag_question_response[0])
        print("Check decoded tokenizer: \n", check, "\n")

    # pass through model, get hidden state
    outputs = model(diag_question_response, output_hidden_states=True) 
    hiddens_diag_response = outputs.hidden_states[-1][:, -1*response_tokens.shape[-1]:]

    # calculate loss
    logits = model.lm_head(hiddens_diag_response)
    ground_truth_tokens = tokenizer(ground_truth_answers, return_tensors='pt')["input_ids"].to(device)
    padded_ground_truth = torch.nn.functional.pad(
        ground_truth_tokens, (0, response_tokens.size(1) - ground_truth_tokens.size(1)), 'constant', 0
    )
    logits = logits[:, :padded_ground_truth.size(1), :].contiguous()
    loss_fct = CrossEntropyLoss(reduction="mean")
    loss = loss_fct(logits.view(-1, logits.size(-1)), padded_ground_truth.view(-1))

    return loss.item(), conversation
