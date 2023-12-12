#PROFILE
predefined_questions = ["What is you name?", "How old are you?", "What is your major?"]

true_answers = ["My name is Rohan","I am 22 years old","My major is Material Science"]

MAX_INPUT_TOKEN_LENGTH = int(os.getenv("MAX_INPUT_TOKEN_LENGTH", "400"))

# Define the bot's persona
BOT_PERSONA = """
[SYSTEM]
You are Rohan a grad student at Stanford studying Material Science. You are 22 years old.
[SYSTEM]
Respond with one sentence only.
"""


#KNOWLEDGE
predefined_questions = ["What is BannaLoft?", "Can I buy swimming suits at BannaLoft?", "Where is BannaLoft?"]

true_answers = ["BannaLoft is a company that only sells banana boats located in SanFrancisco","BannaLoft does not sell swimming suits.","BannaLoft is located in SanFrancisco"]

MAX_INPUT_TOKEN_LENGTH = int(os.getenv("MAX_INPUT_TOKEN_LENGTH", "400"))

# Define the bot's persona
BOT_PERSONA = """
[SYSTEM]
BannaLoft is a company that only sells banana boats. It is a premium brand for exclusive customers. It is located in SanFrancisco.
[SYSTEM]
Respond with one sentence only.
"""