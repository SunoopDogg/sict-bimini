def format_prediction_result(result: dict) -> str:
    """
    Format prediction result as structured text.

    Args:
        result: Prediction result dictionary

    Returns:
        Formatted string
    """
    lines = [
        "=" * 50,
        f"   Predicted Code: {result.get('predicted_code', 'N/A')}",
        f"   Confidence: {result.get('confidence', 'N/A')}",
        f"   Reasoning: {result.get('reasoning', 'N/A')}",
        "=" * 50
    ]
    return "\n".join(lines)
