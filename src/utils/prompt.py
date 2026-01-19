from pathlib import Path


# Prompt file paths
PROMPTS_DIR = Path(__file__).parent.parent.parent / "prompts"


def load_prompt(prompt_name: str) -> str:
    """
    Load prompt template from prompts/ directory.

    Args:
        prompt_name: Name of the prompt file (without .txt extension)

    Returns:
        Prompt template string
    """
    prompt_path = PROMPTS_DIR / f"{prompt_name}.txt"
    with open(prompt_path, "r", encoding="utf-8") as f:
        return f.read()
