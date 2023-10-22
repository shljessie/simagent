from typing import Any, Dict, List, Optional
from pydantic import root_validator
from langchain.memory.chat_memory import BaseMemory
from langchain.memory.utils import get_prompt_input_key

from langchain.llms import LlamaCpp
from langchain import PromptTemplate
from langchain.chains import ConversationChain
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler


class LlamaConversationMemory(BaseMemory):
    """Buffer for storing conversation memory."""

    human_prefix: str = ""
    ai_prefix: str = ""
    """Prefix to use for AI generated responses."""
    buffer: str = ""
    output_key: Optional[str] = None
    input_key: Optional[str] = None
    memory_key: str = "history"  #: :meta private:

    def validate_chains(cls, values: Dict) -> Dict:
        """Validate that return messages is not True."""
        if values.get("return_messages", False):
            raise ValueError(
                "return_messages must be False for LlamaConversationMemory"
            )
        return values

    def memory_variables(self) -> List[str]:
        """Will always return list of memory variables.
        :meta private:
        """
        return [self.memory_key]

    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, str]:
        """Return history buffer."""
        return {self.memory_key: self.buffer}

    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, str]) -> None:
        """Save context from this conversation to buffer."""
        if self.input_key is None:
            prompt_input_key = get_prompt_input_key(inputs, self.memory_variables)
        else:
            prompt_input_key = self.input_key
        if self.output_key is None:
            if len(outputs) != 1:
                raise ValueError(f"One output key expected, got {outputs.keys()}")
            output_key = list(outputs.keys())[0]
        else:
            output_key = self.output_key
        human = inputs[prompt_input_key]
        ai = outputs[output_key]
        self.buffer += " [/INST] ".join([human, ai])
        self.buffer += " </s><s>[INST] "

    def clear(self) -> None:
        """Clear memory contents."""
        self.buffer = ""

human = "Theodore"
robot = "Samantha"
template = f"""<s>[INST] <<SYS>>
My name is {human}. Your name is {robot}. You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.
<</SYS>>

{{history}} {{input}} [/INST]"""

prompt = PromptTemplate.from_template(template)
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
memory = LlamaConversationMemory()

# Make sure the model path is correct for your system!
llm = LlamaCpp(
    model_path = "checkpoints/meta-llama/Llama-2-7b-chat-hf",
    n_gpu_layers = 64,
    max_tokens = 1024,
    n_ctx = 2048,
    callback_manager = callback_manager
)

chain = ConversationChain(
    llm=llm,
    prompt=prompt,
    verbose=True,
    memory=memory
)

def chat(message: str):
    chain.predict(input=message)