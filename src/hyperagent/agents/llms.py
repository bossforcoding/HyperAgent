import os
from groq import Groq
from vllm import LLM as vLLM
from abc import ABC, abstractmethod
from openai import OpenAI, AzureOpenAI
from transformers import AutoTokenizer, AutoConfig
from hyperagent.constants import OLLAMA_URL


def truncate_tokens_hf(string: str, encoding_name: str) -> str:
    """Truncates a text string based on max number of tokens."""
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
    max_tokens = AutoConfig.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct").max_position_embeddings
    encoded_string = tokenizer.encode(string, return_tensors="pt")
    num_tokens = len(encoded_string[0])

    if num_tokens > max_tokens:
        string = tokenizer.decode(encoded_string[0][-max_tokens + 1000:])

    return string


class BaseLLM(ABC):
    def __init__(self, config):
        self.system_prompt = config["system_prompt"]
        self.config = config

    @abstractmethod
    def generate_response(self, prompt: str) -> str:
        """Generate response from the LLM"""
        pass

    def __call__(self, prompt: str) -> str:
        """Make the class callable"""
        return self.generate_response(prompt)


class GroqLLM(BaseLLM):
    def __init__(self, config):
        super().__init__(config)
        self.client = Groq(
            api_key=os.environ["GROQ_API_KEY"],
        )

    def generate_response(self, prompt: str):
        response = self.client.chat.completions.create(
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt},
            ],
            model=self.config["model"],
        )
        return response.choices[0].message.content


class LocalLLM(BaseLLM):
    def __init__(self, config):
        super().__init__(config)
        openai_api_key = os.environ["TOGETHER_API_KEY"]
        openai_api_base = "https://api.together.xyz"
        # openai_api_base = "http://localhost:8004/v1"
        # openai_api_key="token-abc123"

        self.client = OpenAI(
            api_key=openai_api_key,
            base_url=openai_api_base,
        )

    def generate_response(self, prompt: str):
        prompt = truncate_tokens_hf(prompt, encoding_name=self.config["model"])
        response = self.client.chat.completions.create(
            temperature=0,
            model=self.config["model"],
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt},
            ],
            max_tokens=None
        )
        return response.choices[0].message.content


class OpenAILLM(BaseLLM):
    def __init__(self, config):
        super().__init__(config)
        if "openai_api_key" in config:
            openai_api_key = config["openai_api_key"]
        elif "OPENAI_API_KEY" in os.environ:
            openai_api_key = os.environ["OPENAI_API_KEY"]
        else:
            assert False, "OpenAI API key not found"
        self.client = OpenAI(
            api_key=openai_api_key,
        )

    def generate_response(self, prompt: str):
        # The line `prompt = truncate_tokens(prompt, encoding_name=self.config["model"],
        # max_length=self.config["max_tokens"])` is calling a function named `truncate_tokens` with
        # three arguments: `prompt`, `encoding_name`, and `max_length`. This function is likely used
        # to truncate the input `prompt` to a specified maximum length based on the model being used
        # and the maximum tokens allowed.
        # prompt = truncate_tokens(prompt, encoding_name=self.config["model"], max_length=self.config["max_tokens"])
        response = self.client.chat.completions.create(
            temperature=0,
            model=self.config["model"],
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt},
            ]
        )
        return response.choices[0].message.content


class AzureLLM(BaseLLM):
    def __init__(self, config):
        super().__init__(config)
        if "openai_api_key" in config:
            openai_api_key = config["openai_api_key"]
        elif "OPENAI_API_KEY" in os.environ:
            openai_api_key = os.environ["OPENAI_API_KEY"]
        else:
            assert False, "OpenAI API key not found"

        self.client = AzureOpenAI(
            azure_endpoint=os.environ["AZURE_ENDPOINT_GPT35"] if "gpt35" in self.config["model"] else os.environ[
                "AZURE_ENDPOINT_GPT4"],
            api_key=openai_api_key,
            api_version=os.environ["API_VERSION"],
            azure_deployment="ai4code-research-gpt4o"
        )

    def generate_response(self, prompt: str):
        # prompt = truncate_tokens(prompt, encoding_name=self.config["model"], max_length=self.config["max_tokens"])
        response = self.client.chat.completions.create(
            temperature=0,
            model=self.config["model"],
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt},
            ]
        )
        return response.choices[0].message.content


class VLLM(BaseLLM):
    def __init__(self, config):
        super().__init__(config)
        self.client = vLLM(
            model=config["model"],
            tensor_parallel_size=2,
        )
        self.system_prompt = config["system_prompt"]

    def generate_response(self, prompt: str):
        composed_prompt = f"{self.system_prompt} {prompt}"
        response = self.client.generate(composed_prompt)
        return response[0].outputs[0].text


class OllamaLLM(BaseLLM):
    """Ollama LLM wrapper for local model inference"""

    def __init__(self, config, base_url=OLLAMA_URL):
        super().__init__(config)

        self.client = OpenAI(base_url=f"{base_url}/v1", api_key="ollama")

    def generate_response(self, prompt: str):
        response = self.client.chat.completions.create(
            temperature=0,
            model=self.config["model"],
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt},
            ]
        )
        return response.choices[0].message.content


def create_llm(config, llm_type="OLLAMA", **kwargs) -> BaseLLM:
    """
    Factory function to create LLM instances

    Args:
        config: LLM configuration
        llm_type: Type of LLM ('OLLAMA', etc.)
        **kwargs: Additional arguments for specific LLM types

    Returns:
        LLM instance

    Raises:
        ValueError: If llm_type is not supported
    """
    if llm_type.lower() == 'openai':
        return OpenAILLM(config)
    elif llm_type.lower() == 'azure':
        return AzureLLM(config)
    elif llm_type.lower() == 'groq':
        return GroqLLM(config)
    elif llm_type.lower() == 'vllm':
        return VLLM(config)
    elif llm_type.lower() == 'ollama':
        return OllamaLLM(config, OLLAMA_URL)
    elif llm_type.lower() == 'local':
        return LocalLLM(config)
    else:
        raise ValueError(f"Unsupported LLM type: {llm_type}")
