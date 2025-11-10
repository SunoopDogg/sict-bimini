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
    # Try to find JSON block in the response
    json_match = re.search(r'\{[\s\S]*\}', response)
    if json_match:
        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError as e:
            logger.warning(f"JSON parse error: {e}")
            raise ValueError(f"유효하지 않은 JSON 형식: {e}")
    raise ValueError("응답에서 유효한 JSON을 찾을 수 없습니다")
