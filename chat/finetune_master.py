import subprocess
import os
from config import ConfigProfile

consistency = "profile"

if consistency == 'profile':
    config = ConfigProfile
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
    finetune_args = ['--config', 'profile', '--rounds', '5']
    # Run finetune.py
    run_script('finetune.py', finetune_args)

    # Retrieve the model name saved by finetune.py
    model_name = config.model_name

    # backproploss run
    # if model name is specificed means I am using finetuned model
    chat_args = ['--config', 'profile', '--rounds', '5', '--finetune_model', model_name]
    # Run chat.py
    run_script('chat.py', chat_args)
