import os

from autogen.agentchat.contrib.multimodal_conversable_agent import DEFAULT_MODEL
from dotenv import load_dotenv

load_dotenv()

DEFAULT_GH_TOKEN = os.environ.get("GITHUB_TOKEN", None)
DEFAULT_DEVICES = "0"
DEFAULT_CLONE_DIR = "data/repos"
SEMANTIC_CODE_SEARCH_DB_PATH = "/tmp/semantic_code_search_hyperagent/"
ZOEKT_CODE_SEARCH_INDEX_PATH = "/tmp/zoekt_code_search_hyperagent/"
DEFAULT_PATCHES_DIR = "/tmp/hyperagent/patches"
DEFAULT_WORKDIR_CLI = "/tmp/hyperagent/"
DEFAULT_PLANNER_TYPE = "static"
DEFAULT_VLLM_PORT = 5200
DEFAULT_LANGUAGE = "python"
DEFAULT_VERBOSE_LEVEL = 1
DEFAULT_TRAJECTORIES_PATH = "data/agent_trajectories/nav"
DO_NOT_SUMMARIZED_KEYS = ["python", "code_snippet"]
MODEL_TYPE = os.environ.get("MODEL_TYPE", "OLLAMA")
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
SUMMARIZER_MODEL = os.environ.get("SUMMARIZER_MODEL", "llama3.2:latest ")
REVIEWER_MODEL = os.environ.get("REVIEWER_MODEL", "llama3.2:latest")
NAV_MODEL = os.environ.get("NAV_MODEL", "llama3.2:latest")
EDIT_MODEL = os.environ.get("EDIT_MODEL", "llama3.2:latest")
EXEC_MODEL = os.environ.get("EXEC_MODEL", "llama3.2:latest")
PLAN_MODEL = os.environ.get("PLAN_MODEL", "llama3.2:latest")
FAULT_MODEL = os.environ.get("FAULT_MODEL", "llama3.2:latest")
DEFAULT_MODEL = os.environ.get("DEFAULT_MODEL", "llama3.2:latest")

DEFAULT_LLM_CONFIGS = {
    "name": "ollama",
    "nav": [{
        "model": NAV_MODEL,
        "api_key": "",
        "stop_sequences": ["\nObservation:"],
        "base_url": OLLAMA_HOST,
        "api_type": "ollama",
    }],
    "edit": [{
        "model": EDIT_MODEL,
        "api_key": "",
        "stop_sequences": ["\nObservation:"],
        "base_url": OLLAMA_HOST,
        "api_type": "ollama",
    }],
    "exec": [{
        "model": EXEC_MODEL,
        "api_key": "",
        "stop_sequences": ["\nObservation:"],
        "base_url": OLLAMA_HOST,
        "api_type": "ollama",
    }],
    "plan": [{
        "model": PLAN_MODEL,
        "api_key": "",
        "base_url": OLLAMA_HOST,
        "api_type": "ollama",
    }],
    "type": "patch"
}

DEFAULT_IMAGE_NAME = "python:3-slim"
D4J_FOLDER = "/datadrive5/huypn16/defects4j"
