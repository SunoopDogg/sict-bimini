import json
import logging
import re


logger = logging.getLogger(__name__)


def parse_json_response(response: str) -> dict:
    """
    Extract and parse JSON from LLM response.

    Args:
        response: Raw LLM response string

    Returns:
        Parsed JSON as dictionary

    Raises:
        ValueError: If no valid JSON found in response
    """
    # Try direct parse first (fastest path for clean JSON responses)
    stripped = response.strip()
    if stripped.startswith("{"):
        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            pass

    # Fallback: extract JSON block from surrounding text
    json_match = re.search(r'\{[\s\S]*\}', response)
    if json_match:
        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError as e:
            logger.warning(f"JSON parse error: {e}")
            raise ValueError(f"Invalid JSON format: {e}")
    raise ValueError("No valid JSON found in response")
