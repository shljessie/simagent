import subprocess
import os
from config import ConfigProfile7b, ConfigProfile13b

consistency = "profile7b"

if consistency == 'profile7b':
    config = ConfigProfile7b
elif consistency == 'profile13b':
    config = ConfigProfile13b
else:
    raise ValueError("Invalid Consistency Category")

def run_script(script_name, args):
    try:
        subprocess.run(['python', script_name] + args, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running {script_name}: {e}")
        raise

if __name__ == "__main__":
    # Define parameters for finetune.py
    finetune_args = ['--config', 'profile7b', '--rounds', '16']
    # Run finetune.py
    run_script('chat/finetune.py', finetune_args)

    # Retrieve the model name saved by finetune.py
    model_name = config.model_name

    # backproploss run
    # if model name is specificed means I am using finetuned model
    # python3 chat/chat.py --config profile7b --rounds 25 --finetune_model ./backprop_llama2_48_0.0001
    chat_args = ['--config', 'profile7b', '--rounds', '30', '--finetune_model', model_name]
    # Run chat.py
    run_script('chat/chat.py', chat_args)


