import logging
from argparse import ArgumentParser, Namespace
from pathlib import Path

from hyperagent import HyperAgent

# Constants
DEFAULT_PROMPT = "How to add new memory efficient fine-tuning technique to the project?"

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logging.getLogger('hyperagent').setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


def parse():
    parser = ArgumentParser()
    parser.add_argument("--repo", type=str, required=True)
    parser.add_argument("--commit", type=str, default="")
    parser.add_argument("--language", type=str, default="python")
    parser.add_argument("--clone_dir", type=str, default="data/repos")

    # Mutually exclusive prompt options
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--prompt",
        type=str,
        help="Prompt text to query the codebase"
    )
    group.add_argument(
        "--prompt_file",
        type=str,
        help="Path to a .txt file containing the prompt"
    )

    return parser.parse_args()


def load_prompt(args: Namespace) -> str:
    """
    Load prompt from file, argument, or use default.
    Preserves all original whitespace, newlines, and formatting.

    Args:
        args: Parsed arguments from ArgumentParser.

    Returns:
        The prompt text with original formatting preserved.

    Raises:
        FileNotFoundError: If the prompt file doesn't exist.
        ValueError: If the prompt is empty or contains only whitespace.
    """
    if args.prompt_file:
        path = Path(args.prompt_file)
        if not path.exists():
            raise FileNotFoundError(f"Prompt file not found: {path}")
        if not path.is_file():
            raise ValueError(f"Path is not a file: {path}")

        logger.info(f"Loading prompt from file: {path}")
        # Read without stripping to preserve formatting
        prompt_text = path.read_text(encoding="utf-8")

    elif args.prompt:
        logger.info("Using prompt from command line argument")
        # Keep original formatting from command line
        prompt_text = args.prompt

    else:
        logger.info("No prompt provided, using default")
        prompt_text = DEFAULT_PROMPT

    # Validate that prompt is not empty or only whitespace
    if not prompt_text or not prompt_text.strip():
        raise ValueError("Prompt cannot be empty or contain only whitespace")

    logger.debug(f"Prompt loaded (length: {len(prompt_text)} chars, {prompt_text.count(chr(10))} newlines)")
    return prompt_text


if __name__ == "__main__":
    logger.info("Start!")
    args = parse()

    # Load prompt
    prompt = load_prompt(args)

    pilot = HyperAgent(args.repo, commit=args.commit, language=args.language, clone_dir=args.clone_dir)
    logger.info("Setup done!")

    print(pilot.query_codebase(prompt))
