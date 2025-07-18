import json
import logging
from typing import List, Union, Dict, Any, Optional

import requests
from transformers import AutoTokenizer, AutoConfig


def truncate_tokens_hf(string: str, encoding_name: str) -> str:
    """Truncates a text string based on max number of tokens."""
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
    max_tokens = AutoConfig.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct").max_position_embeddings
    encoded_string = tokenizer.encode(string, return_tensors="pt")
    num_tokens = len(encoded_string[0])

    if num_tokens > max_tokens:
        string = tokenizer.decode(encoded_string[0][-max_tokens + 1000:])

    return string


logger = logging.getLogger(__name__)


class OllamaLLM:
    """Ollama LLM wrapper for local model inference"""

    def __init__(
            self,
            model_name: str = "llama2",
            base_url: str = "http://localhost:11434",
            temperature: float = 0.7,
            max_tokens: int = 2048,
            stop_sequences: Optional[List[str]] = None,
            **kwargs
    ):
        self.model_name = model_name
        self.base_url = base_url.rstrip('/')
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.stop_sequences = stop_sequences or []

        # Verifica connessione Ollama
        self._verify_connection()

        # Verifica che il modello sia disponibile
        self._verify_model()

    def _verify_connection(self):
        """Verifica che Ollama sia raggiungibile"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            response.raise_for_status()
            logger.info(f"Successfully connected to Ollama at {self.base_url}")
        except requests.exceptions.RequestException as e:
            error_msg = f"Cannot connect to Ollama at {self.base_url}. Make sure Ollama is running."
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e

    def _verify_model(self):
        """Verifica che il modello richiesto sia disponibile"""
        try:
            response = requests.get(f"{self.base_url}/api/tags")
            response.raise_for_status()
            available_models = [model["name"] for model in response.json().get("models", [])]

            if self.model_name not in available_models:
                logger.warning(
                    f"Model '{self.model_name}' not found. Available models: {available_models}"
                )
                logger.info(f"Pulling model '{self.model_name}'...")
                self._pull_model()
        except Exception as e:
            logger.error(f"Error verifying model: {e}")

    def _pull_model(self):
        """Scarica un modello se non disponibile"""
        try:
            response = requests.post(
                f"{self.base_url}/api/pull",
                json={"name": self.model_name},
                stream=True
            )
            response.raise_for_status()

            for line in response.iter_lines():
                if line:
                    data = json.loads(line)
                    if "status" in data:
                        logger.info(f"Pulling {self.model_name}: {data['status']}")

            logger.info(f"Successfully pulled model '{self.model_name}'")
        except Exception as e:
            logger.error(f"Failed to pull model '{self.model_name}': {e}")
            raise

    def generate(
            self,
            prompt: Union[str, List[str]],
            temperature: Optional[float] = None,
            max_tokens: Optional[int] = None,
            stop_sequences: Optional[List[str]] = None,
            **kwargs
    ) -> Union[str, List[str]]:
        """
        Generate text using Ollama

        Args:
            prompt: String or list of strings to generate from
            temperature: Override default temperature
            max_tokens: Override default max tokens
            stop_sequences: Override default stop sequences

        Returns:
            Generated text (string if single prompt, list if multiple)
        """
        single_prompt = isinstance(prompt, str)
        prompts = [prompt] if single_prompt else prompt

        results = []
        for p in prompts:
            result = self._generate_single(
                p,
                temperature or self.temperature,
                max_tokens or self.max_tokens,
                stop_sequences or self.stop_sequences
            )
            results.append(result)

        return results[0] if single_prompt else results

    def _generate_single(
            self,
            prompt: str,
            temperature: float,
            max_tokens: int,
            stop_sequences: List[str]
    ) -> str:
        """Generate text for a single prompt"""
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            }
        }

        if stop_sequences:
            payload["options"]["stop"] = stop_sequences

        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=300  # 5 minuti timeout per generazioni lunghe
            )
            response.raise_for_status()

            result = response.json()
            return result.get("response", "")

        except requests.exceptions.Timeout:
            logger.error(f"Timeout generating response for prompt: {prompt[:100]}...")
            raise
        except requests.exceptions.RequestException as e:
            logger.error(f"Error generating response: {e}")
            raise

    def chat(
            self,
            messages: List[Dict[str, str]],
            temperature: Optional[float] = None,
            max_tokens: Optional[int] = None,
            **kwargs
    ) -> str:
        """
        Chat completion using Ollama

        Args:
            messages: List of message dicts with 'role' and 'content'
            temperature: Override default temperature
            max_tokens: Override default max tokens

        Returns:
            Generated response
        """
        payload = {
            "model": self.model_name,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature or self.temperature,
                "num_predict": max_tokens or self.max_tokens,
            }
        }

        try:
            response = requests.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=300
            )
            response.raise_for_status()

            result = response.json()
            return result.get("message", {}).get("content", "")

        except Exception as e:
            logger.error(f"Error in chat completion: {e}")
            raise


# Alias per compatibilità
LocalLLM = OllamaLLM


# Factory function per creare LLM
def create_llm(config: Dict[str, Any]) -> OllamaLLM:
    """
    Create an Ollama LLM instance from configuration

    Args:
        config: Configuration dictionary

    Returns:
        OllamaLLM instance
    """
    return OllamaLLM(
        model_name=config.get("model", "llama2"),
        base_url=config.get("base_url", "http://localhost:11434"),
        temperature=config.get("temperature", 0.7),
        max_tokens=config.get("max_tokens", 2048),
        stop_sequences=config.get("stop_sequences", [])
    )


if __name__ == "__main__":
    config = {
        "model": "gradientai/Llama-3-8B-Instruct-Gradient-1048k",
        "system_prompt": "Being an helpful AI, I will help you with your queries. Please ask me anything."
    }
    llm = OllamaLLM()
    llm("How to create a new column in pandas dataframe?")
