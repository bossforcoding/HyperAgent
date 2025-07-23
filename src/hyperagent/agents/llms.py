import json
import logging
import os
from hyperagent import constants
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


class LLM:
    def __init__(self, config):
        self.system_prompt = config["system_prompt"]
        self.config = config

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        pass


class OllamaLLM(LLM):
    """Ollama LLM wrapper for local model inference"""

    def __init__(self, config, base_url: str = constants.OLLAMA_URL):
        super().__init__(config)
        self.model_name = self.config["model"]
        self.base_url = base_url.rstrip('/')
        # self.temperature = temperature
        self.max_tokens = self.config["max_tokens"]

        # Check Ollama connection
        self._verify_connection()

        # Check if the model is available
        self._verify_model()

    def _verify_connection(self):
        """Checks that Ollama is reachable"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            response.raise_for_status()
            logger.info(f"Successfully connected to Ollama at {self.base_url}")
        except requests.exceptions.RequestException as e:
            error_msg = f"Cannot connect to Ollama at {self.base_url}. Make sure Ollama is running."
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e

    def _verify_model(self):
        """Checks that the requested model is available"""
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
        """Downloads a model if not available"""
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

    def convert_ollama_to_openai(self, output):
        openai_output = {
            "choices": [
                {
                    "content_filter_results": {
                        "hate": {
                            "filtered": False,
                            "severity": "safe"
                        },
                        "self_harm": {
                            "filtered": False,
                            "severity": "safe"
                        },
                        "sexual": {
                            "filtered": False,
                            "severity": "safe"
                        },
                        "violence": {
                            "filtered": False,
                            "severity": "safe"
                        }
                    },
                    "finish_reason": "stop",
                    "index": 0,
                    "message": {
                        "content": str(output['message']['content']),
                        "role": "user"
                    }
                }
            ],
            "created": 1716105669,
            "id": "chatcmpl-9QVldRhe4q0z7qIz3uZ6oFq7E5lvw",
            "model": output['model'],
            "object": "chat.completion",
            "prompt_filter_results": [
                {
                    "prompt_index": 0,
                    "content_filter_results": {
                        "hate": {
                            "filtered": False,
                            "severity": "safe"
                        },
                        "self_harm": {
                            "filtered": False,
                            "severity": "safe"
                        },
                        "sexual": {
                            "filtered": False,
                            "severity": "safe"
                        },
                        "violence": {
                            "filtered": False,
                            "severity": "safe"
                        }
                    }
                }
            ],
            "system_fingerprint": None,
            "usage": {
                "completion_tokens": -1,
                "prompt_tokens": -1,
                "total_tokens": -1
            }
        }

        return openai_output

    def __call__(self, prompt: str):

        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt},
            ],
            "stream": False  # Set to False to receive the complete response at once
        }

        response = requests.post(
            # f"{self.base_url}/api/generate",
            f"{self.base_url}/api/chat",
            json=payload,
        ).json()

        message_content = response.get("message", {}).get("content", "No content received.")
        message_content_opeai_style = convert_ollama_to_openai = self.convert_ollama_to_openai(response)
        print("Ollama Message Content:", message_content)
        print("Ollama Message Content OpenAI Style:", message_content_opeai_style)
        return message_content
        # return message_content_opeai_style


# Alias for compatibility
LocalLLM = OllamaLLM
